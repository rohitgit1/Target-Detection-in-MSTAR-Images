"""
Unified Command-Line Interface (CLI) for MSTAR SAR Automatic Target Recognition.
"""

from typing import Optional
import argparse
import os
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from mstar_atr.constants import (
    CLASSES,
    DEFAULT_CROP_SIZE,
    TARGET_METADATA,
)
from mstar_atr.data.dataset import SARTransform
from mstar_atr.data.downloader import prepare_mstar_dataset
from mstar_atr.filters.speckle import lee_filter
from mstar_atr.interpretation.gradcam import GradCAM, overlay_gradcam_on_sar
from mstar_atr.training.trainer import (
    evaluate_model,
    load_checkpoint,
    train_mstar_model,
)


def cmd_prepare_data(args: argparse.Namespace) -> None:
    print(f"[MSTAR-ATR] Preparing dataset in {args.dir}...")
    prepare_mstar_dataset(target_dir=args.dir, force_synthetic=args.synthetic)


def cmd_train(args: argparse.Namespace) -> None:
    print(f"[MSTAR-ATR] Training {args.model} on {args.data} for {args.epochs} epochs...")
    prepare_mstar_dataset(target_dir=args.data, verbose=False)
    train_mstar_model(
        data_dir=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    from mstar_atr.data.dataset import get_mstar_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MSTAR-ATR] Loading model from {args.checkpoint}...")
    model, ckpt = load_checkpoint(args.checkpoint, device=device)

    _, test_loader = get_mstar_dataloaders(data_dir=args.data, batch_size=32)
    loss, acc, preds, targets = evaluate_model(model, test_loader, device=device)

    print(f"[MSTAR-ATR] Evaluation on {args.data}:")
    print(f"  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {acc * 100:.2f}%")

    try:
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(targets, preds, target_names=CLASSES[:len(np.unique(targets))]))
    except Exception:
        pass


def cmd_predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device=device)

    with Image.open(args.image) as img:
        img_gray = img.convert("L")
        raw_arr = np.array(img_gray, dtype=np.float32)

    transform = SARTransform(output_size=DEFAULT_CROP_SIZE, is_training=False)
    input_tensor = transform(raw_arr).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    top_idx = int(np.argmax(probs))
    pred_class = CLASSES[top_idx]
    confidence = probs[top_idx]

    meta = TARGET_METADATA.get(pred_class, {})
    print("\n" + "=" * 50)
    print("MSTAR SAR RADAR CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"Target Image:      {args.image}")
    print(f"Predicted Class:   {pred_class} ({meta.get('full_name', 'Unknown')})")
    print(f"Role / Category:   {meta.get('category', 'N/A')}")
    print(f"Confidence:        {confidence * 100:.2f}%")
    print(f"Radar Signature:   {meta.get('radar_signature', 'N/A')}")
    print("\nTop-3 Predictions:")
    top3_indices = np.argsort(probs)[::-1][:3]
    for rank, idx in enumerate(top3_indices, 1):
        print(f"  {rank}. {CLASSES[idx]:<10} : {probs[idx] * 100:.2f}%")
    print("=" * 50)

    if args.cam:
        cam_engine = GradCAM(model)
        heatmap, _, _ = cam_engine.generate_heatmap(input_tensor, target_class=top_idx)
        overlay = overlay_gradcam_on_sar(raw_arr, heatmap)
        output_cam_path = os.path.splitext(args.image)[0] + "_gradcam.png"
        Image.fromarray(overlay).save(output_cam_path)
        print(f"Grad-CAM Attention Map saved to: {output_cam_path}")


def cmd_demo(args: argparse.Namespace) -> None:
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(args.port)]
    print(f"[MSTAR-ATR] Launching Streamlit Tactical Radar App on port {args.port}...")
    subprocess.run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mstar-atr",
        description="MSTAR SAR Automatic Target Recognition (ATR) Suite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare-data
    p_prep = subparsers.add_parser("prepare-data", help="Download or generate MSTAR data")
    p_prep.add_argument("--dir", default="data/mstar", help="Directory for MSTAR data")
    p_prep.add_argument("--synthetic", action="store_true", help="Force synthetic chip generation")
    p_prep.set_defaults(func=cmd_prepare_data)

    # train
    p_train = subparsers.add_parser("train", help="Train a deep learning model")
    p_train.add_argument("--model", default="aconvnet", choices=["aconvnet", "resnet18", "legacy_cnn"])
    p_train.add_argument("--data", default="data/mstar", help="Path to MSTAR dataset folder")
    p_train.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    p_train.add_argument("--output-dir", default="checkpoints", help="Directory to save checkpoints")
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate a model checkpoint")
    p_eval.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p_eval.add_argument("--data", default="data/mstar", help="Path to MSTAR dataset folder")
    p_eval.set_defaults(func=cmd_eval)

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict target class on an image")
    p_pred.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p_pred.add_argument("--image", required=True, help="Path to SAR image file")
    p_pred.add_argument("--cam", action="store_true", help="Generate Grad-CAM visualization")
    p_pred.set_defaults(func=cmd_predict)

    # demo
    p_demo = subparsers.add_parser("demo", help="Launch interactive Streamlit Radar web app")
    p_demo.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
