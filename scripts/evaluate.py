import sys, os, argparse, json, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.alphaknit.inference import AlphaKnitPredictor
from src.alphaknit.evaluator import Evaluator
from src.alphaknit.knitting_dataset import KnittingDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_full.pt")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="checkpoints/eval_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset", type=str, default="data/processed/dataset_5k")
    args = parser.parse_args()

    # Load trained model
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        # Fallback to best_model.pt if full not found
        if os.path.exists("checkpoints/best_model.pt"):
            print("Falling back to checkpoints/best_model.pt")
            args.checkpoint = "checkpoints/best_model.pt"
        else:
            sys.exit(1)

    predictor = AlphaKnitPredictor.load(args.checkpoint, device_str=args.device)
    evaluator = Evaluator()

    # Dataset
    dataset = KnittingDataset(args.dataset, n_points=256, max_seq_len=300)

    # Evaluate
    print(f"Evaluating on {args.samples} samples...")
    summary = evaluator.evaluate_dataset(
        predictor.model, dataset, n_samples=args.samples,
        device=torch.device(args.device)
    )

    # Save
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Evaluation Summary ({summary['n_evaluated']} samples) ===")
    print(f"Token accuracy:       {summary['mean_token_acc']:.4f}")
    print(f"Edit distance:        {summary['mean_edit_dist']:.2f}")
    if summary['mean_sc_mae']:
        print(f"Stitch count MAE:     {summary['mean_sc_mae']:.2f}")
    if summary['mean_chamfer']:
        print(f"Chamfer distance:     {summary['mean_chamfer']:.4f}")
    print(f"Compile success rate: {summary['compile_success_rate']*100:.1f}%")
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
