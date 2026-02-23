"""
Post-training auto-evaluation script.
Run this after phase8_train.py finishes, or manually at any point.

Usage:
    python scripts/eval_phase8.py
    python scripts/eval_phase8.py --checkpoint checkpoints/best_model_phase8.pt --samples 200
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.alphaknit.inference import AlphaKnitPredictor
from src.alphaknit.knitting_dataset import KnittingDataset
from src.alphaknit.compiler import KnittingCompiler, CompileError
from src.alphaknit import config


def eval_model(predictor, dataset, n_samples=200, beam_widths=(1, 3)):
    """
    Evaluate a predictor on n_samples from dataset.
    Returns dict with compile_success_rate per beam_width, top confusions, val_loss proxy.
    """
    compiler = KnittingCompiler()
    results = {}

    for bw in beam_widths:
        compile_ok = 0
        valid_ok = 0
        total = 0
        confusion = {}   # (pred_tok, true_tok) → count

        for idx in range(min(n_samples, len(dataset))):
            pc_npy, src_ids, tgt_ids = dataset[idx]
            pc = pc_npy.numpy()

            result = predictor.predict(pc, beam_width=bw)
            tokens = result["tokens"]

            # Compile check
            try:
                compiler.compile(tokens)
                compile_ok += 1
            except (CompileError, Exception):
                pass

            if result["valid"]:
                valid_ok += 1

            # Confusion on last few tokens vs ground truth
            gt_tokens = [config.ID_TO_TOKEN.get(i, "<UNK>") for i in tgt_ids.tolist()]
            pred_tokens = tokens + ["<PAD>"] * max(0, len(gt_tokens) - len(tokens))
            for p, g in zip(pred_tokens[:len(gt_tokens)], gt_tokens):
                if g == "<PAD>" or g == "<SOS>":
                    continue
                key = (p, g)
                confusion[key] = confusion.get(key, 0) + 1

            total += 1

            if (idx + 1) % 20 == 0:
                print(f"  beam_width={bw}: {idx+1}/{n_samples} — compile={compile_ok/(idx+1)*100:.1f}%")

        # Top confusions (pred ≠ true)
        errors = {k: v for k, v in confusion.items() if k[0] != k[1]}
        top_conf = sorted(errors.items(), key=lambda x: -x[1])[:8]

        results[f"beam_{bw}"] = {
            "n_evaluated": total,
            "compile_success_rate": round(compile_ok / max(total, 1), 4),
            "valid_rate": round(valid_ok / max(total, 1), 4),
            "top_confusions": [
                {"pred": p, "true": g, "count": cnt}
                for (p, g), cnt in top_conf
            ]
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model_phase8.pt")
    parser.add_argument("--dataset", default="data/debug/dataset_5k")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", default="checkpoints/eval_phase8.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Fallback to test checkpoint if phase8 not ready
    if not os.path.exists(args.checkpoint):
        fallback = "checkpoints/best_model_phase8_test.pt"
        if os.path.exists(fallback):
            print(f"Phase8 checkpoint not found, using smoke test checkpoint: {fallback}")
            args.checkpoint = fallback
        else:
            print(f"No checkpoint found at {args.checkpoint}")
            sys.exit(1)

    print(f"Loading: {args.checkpoint}")
    predictor = AlphaKnitPredictor.load(args.checkpoint, device_str=args.device)

    print(f"Loading dataset: {args.dataset}")
    dataset = KnittingDataset(args.dataset, n_points=config.N_POINTS, max_seq_len=config.MAX_SEQ_LEN)

    print(f"\nEvaluating {args.samples} samples with beam_width=[1, 3]...")
    results = eval_model(predictor, dataset, n_samples=args.samples, beam_widths=(1, 3))

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)
    for mode, res in results.items():
        print(f"\n{mode.upper().replace('_', ' ')}:")
        print(f"  Compile success rate : {res['compile_success_rate']*100:.1f}%")
        print(f"  Valid rate           : {res['valid_rate']*100:.1f}%")
        print(f"  Top confusions:")
        for c in res["top_confusions"][:5]:
            print(f"    pred={c['pred']:<6} true={c['true']:<6} count={c['count']}")

    print(f"\nResults saved to: {args.output}")

    # Key insight
    b1 = results.get("beam_1", {})
    b3 = results.get("beam_3", {})
    if b1 and b3:
        delta = b3["compile_success_rate"] - b1["compile_success_rate"]
        print(f"\nBeam search improvement: {delta*100:+.1f}% compile rate (greedy→beam=3)")


if __name__ == "__main__":
    main()
