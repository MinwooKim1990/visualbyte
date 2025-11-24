import os
import random
import csv
import hashlib
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import timm
from torchvision import datasets, transforms
import math

warnings.filterwarnings("ignore")

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    import wave
    _HAS_SF = False

# Optional libs for audio->image conversions
try:
    import torchaudio
    _HAS_TA = True
except ImportError:
    _HAS_TA = False

try:
    import transformers  # for whisper encoder
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

CONFIG = {
    "seed": 42,
    "epochs": 70,
    "batch_size": 16,
    "lr": 7e-4,
    "image_size": 224,
    "classes_csv": "data/classes_cifar100.csv",
    "image_root": "data/image",
    "audio_train_root": "data/audio",
    "audio_eval_root": "data/audio_eval_tts",
    # if precomputed audio images exist (see scripts/precompute_audio_images.py),
    # set this to data/audio_img_train_1 or _2 automatically at runtime
    "audio_image_root": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_path": "best_visual_byte_train_unfreeze_convnext_v3_lambda1_8cap.pth",
    "cap_per_class": 8,
    # 1: whisper encoder map (recommended), 2: simple log-mel
    "audio_image_convert_method": 1,
    # audio loss weighting
    "audio_loss_weight": 0.8,   # base lambda for audio losses
    "warmup_audio": False,       # if True, use 3-phase schedule over epochs
}


class StableTextureFactory:
    def __init__(self, image_size=518):
        self.image_size = image_size

    def _get_hash(self, text: str) -> int:
        key = f"{text}".encode("utf-8")
        return int(hashlib.md5(key).hexdigest(), 16)

    def generate(self, text: str, variation: int = 0) -> torch.Tensor:
        h = self._get_hash(text)
        img = Image.new("RGB", (self.image_size, self.image_size), color="black")
        draw = ImageDraw.Draw(img)
        r = (h >> 16) & 255
        g = (h >> 8) & 255
        b = h & 255
        base_color = (r, g, b)
        pattern_type = h % 8
        stride = 14 + (h % 25)

        if pattern_type == 0:
            for x in range(0, self.image_size, stride):
                draw.rectangle([x, 0, x + stride // 2, self.image_size], fill=base_color)
        elif pattern_type == 1:
            for y in range(0, self.image_size, stride):
                draw.rectangle([0, y, self.image_size, y + stride // 2], fill=base_color)
        elif pattern_type == 2:
            for y in range(0, self.image_size, stride):
                for x in range(0, self.image_size, stride):
                    if ((x // stride) + (y // stride)) % 2 == 0:
                        draw.rectangle([x, y, x + stride, y + stride], fill=base_color)
        elif pattern_type == 3:
            r_dot = stride // 2
            for y in range(0, self.image_size, stride):
                for x in range(0, self.image_size, stride):
                    draw.ellipse([x, y, x + r_dot, y + r_dot], fill=base_color)
        elif pattern_type == 4:
            width = stride // 2
            for i in range(-self.image_size, self.image_size * 2, stride):
                draw.line([(i, 0), (i + self.image_size, self.image_size)], fill=base_color, width=width)
        elif pattern_type == 5:
            for x in range(0, self.image_size, stride):
                draw.line([(x, 0), (x, self.image_size)], fill=base_color, width=stride // 4)
            for y in range(0, self.image_size, stride):
                draw.line([(0, y), (self.image_size, y)], fill=base_color, width=stride // 4)
        elif pattern_type == 6:
            center = self.image_size // 2
            for r_c in range(self.image_size, 0, -stride):
                draw.ellipse([center - r_c, center - r_c, center + r_c, center + r_c], outline=base_color, width=stride // 2)
        else:
            local_seed = h
            for _ in range(100):
                local_seed = (local_seed * 1103515245 + 12345) & 0x7FFFFFFF
                x = local_seed % self.image_size
                local_seed = (local_seed * 1103515245 + 12345) & 0x7FFFFFFF
                y = local_seed % self.image_size
                w = (local_seed % stride) + 5
                draw.rectangle([x, y, x + w, y + w], fill=base_color)

        img_np = np.array(img)
        if variation > 0:
            if (h % 2) == 0:
                img_np = 255 - img_np
            noise = np.random.randint(0, 60, img_np.shape, dtype=np.int16)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np).filter(ImageFilter.GaussianBlur(radius=1.5))
        else:
            img = Image.fromarray(img_np)
        return torch.tensor(np.array(img).transpose(2, 0, 1) / 255.0, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Audio to image (method 1: whisper encoder map, method 2: log-mel simple)
# ---------------------------------------------------------------------------
def load_audio(path: Path):
    if _HAS_SF:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    else:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32) / 32768.0
    return audio.astype(np.float32), sr


def audio_to_image_whisper(path: Path, image_size: int) -> torch.Tensor:
    if not _HAS_TRANSFORMERS:
        return audio_to_image_logmel(path, image_size)
    try:
        from transformers import WhisperFeatureExtractor, WhisperModel
        feat_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        model = WhisperModel.from_pretrained("openai/whisper-small")
        audio, sr = load_audio(path)
        inputs = feat_extractor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            encoder_outputs = model.encoder(inputs.input_features)
        hidden = encoder_outputs.last_hidden_state  # (1, time, dim)
        # map to 2D: time x dim -> resize
        hmap = hidden.squeeze(0).cpu()  # (T, D)
        # pad/crop to square-ish
        hmap = hmap.T.unsqueeze(0).unsqueeze(0)  # 1x1xD xT
        hmap = F.interpolate(hmap, size=(image_size, image_size), mode="bilinear", align_corners=False)
        img = hmap.squeeze(0).squeeze(0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img.repeat(3, 1, 1)
    except Exception:
        return audio_to_image_logmel(path, image_size)


def audio_to_image_logmel(path: Path, image_size: int) -> torch.Tensor:
    audio, sr = load_audio(path)
    audio_t = torch.tensor(audio, dtype=torch.float32)
    if _HAS_TA:
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)(audio_t)
        mel_spec = torch.log1p(mel_spec)
    else:
        n_fft = 1024
        window = torch.hann_window(n_fft)
        spec = torch.stft(audio_t, n_fft=n_fft, hop_length=256, win_length=n_fft, window=window, return_complex=True)
        mag = torch.log1p(spec.abs())
        mel_spec = mag
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
    mel_spec = mel_spec.unsqueeze(0)  # 1xF xT
    mel_spec = F.interpolate(mel_spec.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
    return mel_spec.repeat(3, 1, 1)


def audio_to_image(path: Path, image_size: int, method: int):
    if method == 1:
        return audio_to_image_whisper(path, image_size)
    else:
        return audio_to_image_logmel(path, image_size)


class VisualByteDataset(Dataset):
    def __init__(
        self,
        classes,
        image_root,
        audio_root,
        train=True,
        image_size=518,
        cap_per_class=4,
        audio_method=1,
        audio_image_root=None,
    ):
        self.classes = classes
        self.train = train
        self.image_size = image_size
        self.audio_root = Path(audio_root)
        self.texture_factory = StableTextureFactory(image_size)
        self.audio_method = audio_method
        self.audio_image_root = Path(audio_image_root) if audio_image_root else None
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.cifar = datasets.CIFAR100(root="./data", train=train, download=True, transform=transform)

        self.cifar_to_local_idx = {}
        for i, c_name in enumerate(self.cifar.classes):
            for local_idx, (en, ko, slug) in enumerate(classes):
                if c_name == slug or c_name == en.lower().replace(" ", "_"):
                    self.cifar_to_local_idx[i] = local_idx
                    break

        counts = {i: 0 for i in range(len(classes))}
        self.valid_indices = []
        self.valid_labels = []
        for i, target in enumerate(self.cifar.targets):
            if target in self.cifar_to_local_idx:
                local_idx = self.cifar_to_local_idx[target]
                if counts[local_idx] < cap_per_class:
                    self.valid_indices.append(i)
                    self.valid_labels.append(local_idx)
                    counts[local_idx] += 1

        print(f"✔ Dataset Filtered (cap {cap_per_class}/class): {len(self.valid_indices)} samples.")

        self.audio_dict = {}
        if self.audio_root.exists():
            for local_idx, (en, ko, slug) in enumerate(classes):
                for lang in ["en", "ko", ""]:
                    p = self.audio_root / slug / lang
                    if p.exists():
                        files = list(p.glob("*.wav")) + list(p.glob("**/*.wav"))
                        if files:
                            self.audio_dict[local_idx] = files[:cap_per_class]

        # optional precomputed audio images (PNG) from scripts/precompute_audio_images.py
        self.audio_image_dict = {}
        if self.audio_image_root and self.audio_image_root.exists():
            for local_idx, (en, ko, slug) in enumerate(classes):
                p = self.audio_image_root / slug
                if p.exists():
                    imgs = list(p.rglob("*.png"))
                    if imgs:
                        self.audio_image_dict[local_idx] = imgs[:cap_per_class]
            print(f"✔ Using precomputed audio images from {self.audio_image_root} ({len(self.audio_image_dict)} classes)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img, cifar_label = self.cifar[real_idx]
        local_idx = self.cifar_to_local_idx[cifar_label]
        class_slug = self.classes[local_idx][2]

        text_img = self.texture_factory.generate(class_slug, variation=0)
        text_img = (text_img - self.mean) / self.std
        has_audio = 0
        audio_img = torch.zeros(3, self.image_size, self.image_size)

        # 1) precomputed PNG if available
        if local_idx in self.audio_image_dict:
            try:
                audio_path = random.choice(self.audio_image_dict[local_idx])
                with Image.open(audio_path).convert("RGB") as im:
                    im = im.resize((self.image_size, self.image_size))
                    audio_img = torch.tensor(np.array(im).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
                    audio_img = (audio_img - self.mean) / self.std
                    has_audio = 1
            except Exception:
                pass

        # 2) fall back to wav -> image on the fly
        if has_audio == 0 and local_idx in self.audio_dict:
            try:
                audio_path = random.choice(self.audio_dict[local_idx])
                audio_img = audio_to_image(audio_path, self.image_size, self.audio_method)
                audio_img = (audio_img - self.mean) / self.std
                has_audio = 1
            except Exception:
                pass
        return {"image": img, "text": text_img, "audio": audio_img, "label": torch.tensor(local_idx), "has_audio": torch.tensor(has_audio)}


class ClassBalancedBatchSampler(Sampler):
    """
    Sample batches where each class appears at most once per batch (image/text/audio triplet per class).
    Works with datasets that expose .valid_indices and .valid_labels aligned.
    """

    def __init__(self, labels, batch_size, num_classes):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        # build index list per class
        self.class_to_indices = {c: [] for c in range(num_classes)}
        for idx, lbl in enumerate(labels):
            self.class_to_indices[lbl].append(idx)
        # number of batches per epoch
        self.num_batches = math.ceil(len(labels) / batch_size)

    def __iter__(self):
        rng = random.Random()
        classes = list(self.class_to_indices.keys())
        for _ in range(self.num_batches):
            rng.shuffle(classes)
            batch_classes = classes[: self.batch_size]
            batch = []
            for c in batch_classes:
                if not self.class_to_indices[c]:
                    continue
                batch.append(rng.choice(self.class_to_indices[c]))
            if batch:
                yield batch

    def __len__(self):
        return self.num_batches


class VisualByteNet(nn.Module):
    def __init__(self, num_classes=100, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = True
        feat_dim = self.backbone.num_features
        self.projector = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.0)
        self.temperature = 0.07

    def forward(self, x):
        feat = self.backbone(x)
        latent = self.projector(feat)
        latent = F.normalize(latent, p=2, dim=-1)
        logits = self.classifier(latent) / self.temperature
        return latent, logits


def audio_lambda_schedule(epoch_idx: int, total_epochs: int, base_lambda: float, use_warmup: bool) -> float:
    """
    3-phase schedule when warmup is enabled:
      0 - 1/3: audio losses off (0.0)
      1/3 - 2/3: ramp from 0 -> base_lambda * 1.2 (fast catch-up)
      2/3 - 1: ramp down to base_lambda (settle)
    """
    if not use_warmup or total_epochs <= 0:
        return base_lambda
    p = (epoch_idx + 1) / total_epochs  # 1-based epoch fraction
    mid_lambda = base_lambda * 1.2
    if p < 1 / 3:
        return 0.0
    elif p < 2 / 3:
        # linear ramp 0 -> mid_lambda
        alpha = (p - 1 / 3) / (1 / 3)
        return mid_lambda * alpha
    else:
        # linear ramp mid_lambda -> base_lambda
        alpha = (p - 2 / 3) / (1 / 3)
        return mid_lambda + (base_lambda - mid_lambda) * alpha


def train_epoch(model, loader, optimizer, scaler, criterion_ce, device, epoch):
    model.train()
    total_loss = 0
    steps = 0
    use_amp = device == "cuda"
    lam_audio = audio_lambda_schedule(epoch, CONFIG["epochs"], CONFIG["audio_loss_weight"], CONFIG.get("warmup_audio", False))

    for batch in loader:
        imgs = batch["image"].to(device)
        texts = batch["text"].to(device)
        audios = batch["audio"].to(device)
        labels = batch["label"].to(device)
        has_audio = batch["has_audio"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda" if use_amp else "cpu", enabled=use_amp, dtype=torch.float16 if use_amp else torch.bfloat16):
            lat_i, log_i = model(imgs)
            lat_t, log_t = model(texts)

            loss_cls = (criterion_ce(log_i, labels) + criterion_ce(log_t, labels)) / 2

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * lat_i @ lat_t.t()
            logits_per_text = logits_per_image.t()
            bs = imgs.size(0)
            gt = torch.arange(bs, dtype=torch.long, device=device)
            loss_clip = (criterion_ce(logits_per_image, gt) + criterion_ce(logits_per_text, gt)) / 2

            loss_audio = 0
            loss_audio_text = 0
            if (has_audio > 0).any():
                lat_a_valid = model(audios[has_audio > 0])[0]
                lat_i_valid = lat_i[has_audio > 0]
                logits_per_audio = logit_scale * lat_a_valid @ lat_i_valid.t()
                valid_bs = len(lat_a_valid)
                loss_audio = (criterion_ce(logits_per_audio, torch.arange(valid_bs, device=device)) + criterion_ce(logits_per_audio.t(), torch.arange(valid_bs, device=device))) / 2
                lat_t_valid = lat_t[has_audio > 0]
                logits_at = logit_scale * lat_a_valid @ lat_t_valid.t()
                loss_audio_text = (criterion_ce(logits_at, torch.arange(valid_bs, device=device)) + criterion_ce(logits_at.t(), torch.arange(valid_bs, device=device))) / 2

            loss = loss_cls + loss_clip + lam_audio * (loss_audio + loss_audio_text)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        steps += 1
        if steps % 50 == 0:
            ce_val = loss_cls.item()
            clip_val = loss_clip.item()
            audio_val = loss_audio.item() if torch.is_tensor(loss_audio) else loss_audio
            audio_text_val = loss_audio_text.item() if torch.is_tensor(loss_audio_text) else loss_audio_text
            scale_val = logit_scale.item()
            total_val = loss.item()
            print(f"  [Epoch {epoch}] Step {steps}")
            print(f"    CE: {ce_val:.2f} | CLIP: {clip_val:.2f} | Audio: {audio_val:.2f} | Audio-Text: {audio_text_val:.2f} | λ_audio: {lam_audio:.2f}")
            print(f"    Scale: {scale_val:.2f} | Total: {total_val:.4f}")

    return total_loss / steps


if __name__ == "__main__":
    from PIL import Image
    print(f"🚀 Running Unfreeze Training v3 on {CONFIG['device']}")
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    classes: List[Tuple[str, str, str]] = []
    if Path(CONFIG["classes_csv"]).exists():
        with open(CONFIG["classes_csv"], encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                en = row.get("en", "").strip()
                if not en:
                    continue
                ko = row.get("ko", "").strip()
                slug = en.lower().replace(" ", "_")
                classes.append((en, ko, slug))
    else:
        print("⚠ CSV missing. Using CIFAR-100.")
        temp = datasets.CIFAR100(root="./data", train=True, download=True)
        classes = [(c, c, c.replace(" ", "_")) for c in temp.classes]

    # choose precomputed audio image root if present
    auto_audio_image_root = CONFIG["audio_image_root"]
    guess_root = Path(f"data/audio_img_train_{CONFIG['audio_image_convert_method']}")
    if auto_audio_image_root is None and guess_root.exists():
        auto_audio_image_root = str(guess_root)

    train_ds = VisualByteDataset(
        classes,
        CONFIG["image_root"],
        CONFIG["audio_train_root"],
        train=True,
        image_size=CONFIG["image_size"],
        cap_per_class=CONFIG["cap_per_class"],
        audio_method=CONFIG["audio_image_convert_method"],
        audio_image_root=auto_audio_image_root,
    )
    # class-balanced batch sampler: one sample per class per batch (up to batch_size classes)
    sampler = ClassBalancedBatchSampler(
        labels=train_ds.valid_labels, batch_size=CONFIG["batch_size"], num_classes=len(classes)
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=4 if CONFIG["device"] == "cuda" else 0)

    model = VisualByteNet(num_classes=len(classes)).to(CONFIG["device"])
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": CONFIG["lr"] * 0.1},
            {"params": head_params, "lr": CONFIG["lr"]},
        ],
        weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if CONFIG["device"] == "cuda" else None

    best_loss = float("inf")
    for epoch in range(CONFIG["epochs"]):
        avg_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, CONFIG["device"], epoch + 1)
        scheduler.step()
        curr_lr = optimizer.param_groups[1]["lr"]
        print(f"📊 Epoch {epoch+1} Avg Loss: {avg_loss:.4f} | LR: {curr_lr:.6f}")
        if avg_loss < best_loss:
            print(f"✅ Saving Best Model: {avg_loss:.4f}")
            best_loss = avg_loss
            torch.save(model.state_dict(), CONFIG["save_path"])

    print("✨ All Done. Run visual_byte_unfreeze_eval_v3.py for evaluation.")
