# VisualByte: Zero‑Shot‑Ready Multimodal Retrieval with a Single Vision Backbone

VisualByte is a personal research prototype that explores a simple but powerful idea:

> **Turn every modality into an image, and let a single ConvNeXt vision backbone learn a shared latent space for text, audio, and images.**

Instead of running separate text, audio, and vision encoders (BERT + Whisper + ViT), VisualByte renders text and audio into texture/spectrogram images and feeds everything through one ConvNeXt‑Tiny. This makes the system:

- **Lightweight** enough for edge devices and robots,
- **Simple** to integrate (one encoder, one latent space),
- **Naturally multimodal**, supporting text→image, audio→image, and image→image retrieval.

The current experiments are CIFAR‑100–scale, but the architecture is compatible with CLIP‑style zero‑shot scaling if trained on larger web‑scale data.

---

## 1. Features

- Single **ConvNeXt‑Tiny** backbone for all modalities  
- Text rendered as **semantic textures** (`text_img_v2`)  
- Audio rendered as **log‑mel / Whisper spectrogram images** (`audio_img_*`)  
- CLIP‑style contrastive training:
  - image ↔ text, image ↔ audio, audio ↔ text  
- Class‑balanced batch sampler: each batch contains (image, text, audio) triplets per class  
- Evaluation:
  - Text→Image and Audio→Image Recall@K  
  - t‑SNE visualization of the cross‑modal latent space  
  - Blog‑ready figures: retrieval grid, t‑SNE, similarity matrix

---

## 2. Repository Layout (important files)

- `visual_byte_unfreeze_train_v5.py` – **recommended training script** (v5, text v2 + audio v1)  
- `visual_byte_unfreeze_eval_v5.py` – evaluation + retrieval/t‑SNE visualizations for v5  
- `scripts/collect_data.py` – build dataset: CIFAR‑100 images + OpenAI TTS audio per class  
- `scripts/precompute_audio_images.py` – precompute **audio images (v1)** from WAVs  
  - outputs: `data/audio_img_train_1`, `data/audio_img_eval_1`  
- `scripts/precompute_audio_images_v2.py` – precompute **text images (v2)** only  
  - outputs: `data/text_img_v2/<slug>/text.png`  
- `scripts/generate_blog_visuals_v5.py` – generate three blog figures:
  - `blog_simple_v5.png` – one class, text/audio/image queries → image result (non‑experts)  
  - `blog_tsne_v5.png` – multimodal t‑SNE (experts)  
  - `blog_confusion_v5.png` – text‑to‑image similarity matrix (experts)

Older versions (`visual_byte_unfreeze_train_v2/v3/v4.py`, etc.) are kept for reference, but **v5 is the recommended path**.

---

## 3. Quickstart (v5 pipeline)

### 3.1 Environment

Python 3.10+ recommended. The core dependencies are:

```bash
pip install torch torchvision timm matplotlib scikit-learn \
            pillow numpy soundfile torchaudio python-dotenv \
            sentence-transformers
```

For TTS data generation you will also need:

- `openai` Python client  
- An `OPENAI_API_KEY` in `.env` or your environment.

### 3.2 Step 1 – Collect images & audio (CIFAR‑100 + OpenAI TTS)

This script:
- reads `data/classes_cifar100.csv` (`en, ko` per class),  
- saves CIFAR‑100 images per class under `data/image/<slug>`,  
- uses OpenAI TTS to synthesize **training audio** per class under `data/audio/<slug>/en/*.wav`.

```bash
python scripts/collect_data.py \
  --classes-csv data/classes_cifar100.csv \
  --images-per-class 8 \
  --image-source cifar100 \
  --cifar-use-test \
  --tts-model gpt-4o-mini-tts
```

> Note: You can adjust `--images-per-class` and the TTS model/voices in `collect_data.py`.  
> This step requires a valid `OPENAI_API_KEY`.

### 3.3 Step 2 – Precompute audio images (v1, log‑mel / Whisper)

For the current best configuration, **audio uses the original v1 spectrogram images**.  
Run:

```bash
python scripts/precompute_audio_images.py
```

This will create:
- `data/audio_img_train_1/<class>/en/*.png` – train audio images  
- `data/audio_img_eval_1/<class>/en/*.png` – eval audio images  

These are used by v5 as the audio modality.

### 3.4 Step 3 – Precompute text images (v2, semantic textures)

Text uses the **new v2 texture‑style renderer** (SentenceTransformer + 4×4 grid).  
Run:

```bash
python scripts/precompute_audio_images_v2.py
```

This script now only generates:
- `data/text_img_v2/<slug>/text.png`

These text images are used as the text modality in v5.

### 3.5 Step 4 – Train v5 (recommended)

`visual_byte_unfreeze_train_v5.py` is configured to:
- use `data/audio_img_train_1` for audio images (auto‑detected),  
- use `data/text_img_v2` for text images,  
- cap per class = 8 samples,  
- ConvNeXt‑Tiny backbone + projector + classifier,  
- CLIP‑style losses (image↔text, image↔audio, audio↔text).

Run:

```bash
python visual_byte_unfreeze_train_v5.py
```

During training, the script also evaluates R@1/R@5 on a held‑out eval set each epoch and saves:

- `best_visual_byte_train_unfreeze_convnext_v5.pth` – best checkpoint by score  
- `last_visual_byte_train_unfreeze_convnext_v4.pth` – last epoch checkpoint

### 3.6 Step 5 – Evaluate and visualize (v5)

To generate retrieval/t‑SNE figures for v5:

```bash
python visual_byte_unfreeze_eval_v5.py
```

This produces:
- `retrieval_grid_improved.png` – detailed retrieval grid (for analysis),
- `tsne_improved.png` – t‑SNE latent space visualization.

For blog‑ready, simpler visuals:

```bash
python scripts/generate_blog_visuals_v5.py
```

This produces three images:

- `blog_simple_v5.png`  
  - One “hero” class.  
  - Rows show **Text**, **Audio**, and **Image** queries, each mapped to the same **image memory**.  
  - Designed so non‑experts can see: *“문자로 묻든, 소리로 묻든, 실제 이미지로 묻든 같은 결과를 찾는다.”*

- `blog_tsne_v5.png`  
  - t‑SNE plot of image/text/audio points.  
  - Colors = classes, markers = modalities.  
  - Good for papers and technical blogs.

- `blog_confusion_v5.png`  
  - Similarity matrix (heatmap) between text query vectors and image‑class centroids for a subset of classes.  
  - Shows how sharply each class is separated in the latent space.

---

## 4. Minimal Retrieval Example (Python)

Here is a minimal snippet showing how to run text→image retrieval with the trained v5 model:

```python
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from visual_byte_unfreeze_eval_v5 import CONFIG, VisualByteNet, EvalDataset

device = CONFIG["device"]

def load_classes(csv_path: Path):
    import csv
    classes = []
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            en = (row.get("en") or row.get("\ufeffen") or "").strip()
            if not en:
                continue
            ko = (row.get("ko") or "").strip()
            slug = en.lower().replace(" ", "_")
            classes.append((en, ko, slug))
    return classes

classes = load_classes(Path(CONFIG["classes_csv"]))

eval_ds = EvalDataset(
    classes,
    CONFIG["audio_eval_root"],
    image_size=CONFIG["image_size"],
    cap_per_class=CONFIG["cap_per_class"],
    audio_method=CONFIG["audio_image_convert_method"],
    audio_image_root=None,                 # auto audio_img_eval_{method}
    text_image_root=CONFIG["text_image_root"],
)
eval_loader = DataLoader(eval_ds, batch_size=CONFIG["batch_size"], shuffle=False)

model = VisualByteNet(num_classes=len(classes)).to(device)
state_dict = torch.load(CONFIG["model_path"], map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

batch = next(iter(eval_loader))
imgs = batch["image"].to(device)
txts = batch["text"].to(device)

with torch.no_grad():
    z_img, _ = model(imgs)
    z_txt, _ = model(txts)

z_img_n = F.normalize(z_img, dim=-1)
z_txt_n = F.normalize(z_txt, dim=-1)
sim = z_txt_n @ z_img_n.T  # [B_txt, B_img]
top1 = sim.argmax(dim=-1)
print("Top‑1 retrieved image indices for each text:", top1.cpu().tolist())
```

You can adapt this pattern for:
- audio→image (use `batch["audio"]`),  
- image→image (use two image sets),  
or offline retrieval against a larger image database by precomputing all `z_img` vectors.

---

## 5. How This Relates to Existing Work

VisualByte is inspired by, but different from, several existing multimodal models:

- **DINOv2 Meets Text / Talk2DINO**  
  Align a DINO/DINOv2 vision encoder with a separate text encoder (LiT‑style).  
  Vision and language towers remain distinct; audio is not part of the core design.

- **ImageBind**  
  Jointly aligns image, text, audio, depth, IMU, thermal into one space, but still uses **separate encoders per modality** and large‑scale training.

- **Audio Spectrogram Transformer**  
  Focuses on audio understanding from spectrograms with an audio‑specific transformer, not a shared vision backbone.

In contrast, VisualByte:

- Uses **no text or audio encoder at inference time**.  
  Text and audio are rendered into images and passed through the same ConvNeXt as real photos.
- Pushes the “everything becomes an image” idea to its logical extreme, which makes it:
  - extremely simple to deploy on edge devices and robots,  
  - a natural fit for existing vision‑only inference stacks.

The current implementation is still CIFAR‑scale and uses synthetic TTS, but the architecture is compatible with CLIP‑style zero‑shot scaling if trained on web‑scale multimodal data.

---

## 6. Known Limitations

To keep expectations grounded:

- **Dataset scale** – All experiments so far are on CIFAR‑100 + synthetic OpenAI TTS.  
  This is far smaller and cleaner than the noisy web‑scale data used by CLIP/Whisper.

- **Audio remains the weakest modality** – Even with regularization and loss weighting, audio→image is consistently below text→image.  
  Early attempts to encode audio as complex texture‑style 2×2 tiles looked nice but hurt accuracy; the current log‑mel/Whisper images are a pragmatic compromise.

- **Heavy use of precompute** – The design leans on off‑line pipelines (TTS, spectrograms, text textures).  
  This is acceptable for many edge scenarios, but not yet a raw‑waveform / raw‑token end‑to‑end system.

Despite these limitations, VisualByte is a working proof‑of‑concept that a **single small vision model** can behave like a multimodal encoder if we design the inputs and training flow carefully.  
If you build on this repo, I’d love to see what you do with it—especially on real robotic or AR platforms.

