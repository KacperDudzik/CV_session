"""predict(image_path) -> class probabilities (M5, Session 10).

  python predict.py path/to/dermoscopic.jpg                       # 11-class leaderboard model
  python predict.py path/to/dermoscopic.jpg --model models/model_m4.pt

The image goes through data.load_image (RGB, PIL resize to 224x224) and session7.prep
(ImageNet mean/std): exactly the validation transform, no random step.
"""

import argparse
from pathlib import Path

import config
import data
from models import load_checkpoint, predict_proba

_loaded = {}  # checkpoint path -> (model, checkpoint), so a folder of images loads the model once


def predict(image_path, model_path=config.MODEL_11) -> dict[str, float]:
    key = str(Path(model_path).resolve())
    if key not in _loaded:
        _loaded[key] = load_checkpoint(model_path)
    model, ckpt = _loaded[key]
    probs = predict_proba(model, data.load_image(image_path).unsqueeze(0))[0]
    return dict(zip(ckpt["classes"], probs.tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default=str(config.MODEL_11))
    args = parser.parse_args()
    probs = predict(args.image, args.model)
    for name, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"{name:8s} {p:.3f}")
