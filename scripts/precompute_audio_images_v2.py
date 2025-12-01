"""
Precompute text images (PNG) for train/eval using the new multimodal
renderer in image_test/txt_img.py.

Text:
  - Input classes: data/classes_cifar100.csv (en, ko, slug)
  - Output: data/text_img_v2/<slug>/text.png (RGB 224x224)

Run:
  python scripts/precompute_audio_images_v2.py

Deps:
  - python-dotenv, pillow, numpy, torch
  - sentence-transformers (for text)
"""
from pathlib import Path
import sys
import os

import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv

# ensure project root (parent of scripts/) is on sys.path so we can import image_test.*
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from image_test.txt_img import StableTextureFactory as TxtTextureFactory
from image_test.txt_img import build_text_multimodal_image
from sentence_transformers import SentenceTransformer


IMAGE_SIZE = 224

CLASSES_CSV = Path("data/classes_cifar100.csv")

OUT_TEXT_ROOT = Path("data/text_img_v2")
SENTENCE_MODEL = "sentence-transformers/static-similarity-mrl-multilingual-v1"


def save_tensor_as_png(tensor: torch.Tensor, out_path: Path):
    arr = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def read_classes(csv_path: Path):
    import csv

    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = (row.get("en") or row.get("\ufeffen") or "").strip()
            if not en:
                continue
            ko = (row.get("ko") or en).strip()
            slug = en.lower().replace(" ", "_")
            rows.append((en, ko, slug))
    return rows


def precompute_text_images(classes, device="cuda"):
    print(f"[text_v2] Using sentence model: {SENTENCE_MODEL}")
    texture_factory = TxtTextureFactory(image_size=IMAGE_SIZE // 4)
    model = SentenceTransformer(SENTENCE_MODEL, device=device)

    for en, ko, slug in classes:
        out_path = OUT_TEXT_ROOT / slug / "text.png"
        if out_path.exists():
            continue
        try:
            img = build_text_multimodal_image(
                en,
                texture_factory=texture_factory,
                semantic_model=model,
                device=device,
                patch_size=IMAGE_SIZE // 4,
            )
            # txt_img main() grayscale+RGB 변환을 따라감
            img = Image.Image.convert(img, "L").convert("RGB")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Failed text '{en}' ({slug}): {e}")


def main():
    load_dotenv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classes = read_classes(CLASSES_CSV)
    print(f"Loaded {len(classes)} classes for text images.")
    precompute_text_images(classes, device=device)
    print("Done (v2 text precompute).")


if __name__ == "__main__":
    main()
