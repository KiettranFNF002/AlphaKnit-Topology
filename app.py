import streamlit as st
import torch
import numpy as np
import trimesh
import plotly.graph_objects as go
import sys
import os
import tempfile

# Add project root to path
sys.path.append(os.getcwd())

from src.alphaknit.inference import AlphaKnitPredictor
from src.alphaknit.compiler import KnittingCompiler
from src.alphaknit.simulator import ForwardSimulator

# Page Config
st.set_page_config(
    page_title="AlphaKnit: 3D to Knitting Pattern",
    page_icon="🧶",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar & Configuration
# ---------------------------------------------------------------------
st.sidebar.title("🧶 AlphaKnit Config")

checkpoint_path = st.sidebar.text_input(
    "Model Checkpoint Path",
    value="checkpoints/best_model_colab_v6.6F.pt",
    help="v6.6F model trained on Colab with WebDataset shards"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: `{device}`")

beam_width = st.sidebar.slider(
    "Beam Width",
    min_value=1, max_value=7, value=1, step=2,
    help="1 = greedy (fast). 3 or 5 = compile-guided beam search (better patterns, slower)."
)

# ---------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------
@st.cache_resource
def load_predictor(path, device_str):
    if not os.path.exists(path):
        return None
    try:
        predictor = AlphaKnitPredictor.load(path, device_str=device_str)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor = load_predictor(checkpoint_path, device)

if predictor is None:
    st.sidebar.warning("⚠️ Model not found. Train usage or check path.")
else:
    st.sidebar.success("✅ Model Loaded!")

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def plot_point_cloud(points, title="Point Cloud", color="blue"):
    """Visualizes a 3D point cloud using Plotly."""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=points[:, 2],  # Color by Z-axis
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def process_uploaded_file(uploaded_file):
    """Parses .npy or .obj files into a point cloud."""
    if uploaded_file.name.endswith('.npy'):
        points = np.load(uploaded_file)
        # Ensure correct shape (N, 3)
        if points.shape[1] != 3 and points.shape[0] == 3:
            points = points.T
        if points.shape[1] != 3:
            st.error(f"Invalid .npy shape: {points.shape}. Expected (N, 3).")
            return None
        return points

    elif uploaded_file.name.endswith('.obj') or uploaded_file.name.endswith('.ply'):
        # Save to temp file for trimesh to load
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            mesh = trimesh.load(tmp_path)
            # Sample points if mesh
            if hasattr(mesh, 'sample'):
                points, _ = trimesh.sample.sample_surface(mesh, 1024) # Sample enough points
            elif hasattr(mesh, 'vertices'):
                points = mesh.vertices
            else:
                st.error("Could not extract points from mesh.")
                return None
                
            # Normalize to unit sphere/box like training data?
            # Creating dataset usually involves centering and scaling.
            # Center
            points = points - np.mean(points, axis=0)
            # Scale
            scale = np.max(np.linalg.norm(points, axis=1))
            if scale > 0:
                points = points / scale
            
            return points
        except Exception as e:
            st.error(f"Error processing mesh: {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    return None

# ---------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------
st.title("🧶 AlphaKnit: Neural Knitting Generator")
st.markdown("Upload a 3D shape (Point Cloud or Mesh) to generate a knitting pattern for it.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload 3D File (.npy, .obj)", type=['npy', 'obj'])

if uploaded_file and predictor:
    points = process_uploaded_file(uploaded_file)
    
    if points is not None:
        # Resample to exactly 256 for model if needed, or model handles it via max pooling?
        # Model expects input (B, 3, N) or (B, N, 3)?
        # PointNetEncoder expects (B, 3, N).
        # And usually fixed N is better for batching, but max pool handles variable N.
        # However, data generation used 256 points. Let's resample to 256 for consistency.
        
        if len(points) > 256:
            indices = np.random.choice(len(points), 256, replace=False)
            points_input = points[indices]
        elif len(points) < 256:
            indices = np.random.choice(len(points), 256, replace=True)
            points_input = points[indices]
        else:
            points_input = points

        # Visualize Input
        with col1:
            st.subheader("Input 3D Shape")
            st.plotly_chart(plot_point_cloud(points, "Input Point Cloud"))

        # Predict Button
        if st.button("🧶 Generate Pattern", type="primary"):
            with st.spinner("Thinking like a knitter..."):
                # Run Inference — predict() returns a dict
                try:
                    result = predictor.predict(points_input, beam_width=beam_width)
                    tokens = result["tokens"]
                    graph  = result["graph"]
                    valid  = result["valid"]
                    errors = result["errors"]
                    
                    # Display Results in Col2
                    with col2:
                        st.subheader("Generated Pattern")
                        
                        # Metrics
                        col2a, col2b = st.columns(2)
                        col2a.metric("Compile Status", "✅ Valid" if valid else "❌ Invalid")
                        col2b.metric("Tokens", len(tokens))
                        if graph is not None:
                            st.metric("Stitch Count", graph.size)
                        if errors:
                            st.warning(f"Validation issues: {errors[:3]}")
                            
                            
                        # Format Pattern (Phase 9A)
                        with st.expander("📝 Human-Readable Pattern", expanded=True):
                            try:
                                from src.alphaknit.formatter import PatternFormatter
                                formatter = PatternFormatter()
                                formatted_text = formatter.format_tokens(tokens)
                                st.markdown(formatted_text)
                            except Exception as fe:
                                st.warning(f"Formatting failed: {fe}")
                                
                        # Raw Tokens
                        with st.expander("🔍 Raw Tokens"):
                            st.text_area("Tokens", " ".join(tokens), height=100)
                        
                        if graph is not None:
                            # Simulate the pattern back to 3D
                            st.subheader("Simulated Output")
                            try:
                                sim = ForwardSimulator()
                                mesh = sim.build_mesh(graph)
                                # Apply smoothing for visual (Phase 9A)
                                mesh = sim.apply_laplacian_smoothing(mesh)
                                
                                if hasattr(mesh, 'vertices'):
                                    sim_points = np.array(mesh.vertices)
                                    st.plotly_chart(
                                        plot_point_cloud(sim_points, "Simulated Result")
                                    )
                                else:
                                    st.plotly_chart(
                                        plot_point_cloud(np.array(mesh), "Simulated Result (PC)")
                                    )
                            except Exception as sim_err:
                                st.info(f"Simulation skipped: {sim_err}")
                            
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.exception(e)

elif not predictor:
    st.info("Please verify the model checkpoint path in the sidebar.")
else:
    st.info("Upload a file to get started.")
