import random
import csv
import hashlib
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageDraw, ImageFilter
import timm
from torchvision import datasets, transforms

warnings.filterwarnings("ignore")

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    import wave
    _HAS_SF = False

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
    "model_path": "best_visual_byte_train_unfreeze_convnext_v5.pth",
    "classes_csv": "data/classes_cifar100.csv",
    "audio_eval_root": "data/audio_eval_tts",
    # precomputed audio images root (None => auto from audio_image_convert_method)
    "audio_image_root": None,
    "text_image_root": "data/text_img_v2",
    "image_size": 224,
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cap_per_class": 1,
    # 1: whisper encoder map, 2: log-mel simple
    "audio_image_convert_method": 1,
    # visualization subset (metrics use all classes)
    "vis_num_classes": 10,
    "seed": 42,
}


class StableTextureFactory:
    def __init__(self, image_size=224):
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
        hidden = encoder_outputs.last_hidden_state
        hmap = hidden.squeeze(0).cpu()  # T x D
        hmap = hmap.T.unsqueeze(0).unsqueeze(0)
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
        mel_spec = torch.log1p(spec.abs())
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-6)
    mel_spec = mel_spec.unsqueeze(0)
    mel_spec = F.interpolate(mel_spec.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
    return mel_spec.repeat(3, 1, 1)


def audio_to_image(path: Path, image_size: int, method: int):
    if method == 1:
        return audio_to_image_whisper(path, image_size)
    else:
        return audio_to_image_logmel(path, image_size)


class EvalDataset(Dataset):
    def __init__(self, classes, audio_eval_root, image_size=224, cap_per_class=4, audio_method=1, audio_image_root=None, text_image_root=None):
        self.classes = classes
        self.image_size = image_size
        self.audio_root = Path(audio_eval_root)
        self.texture_factory = StableTextureFactory(image_size)
        self.audio_method = audio_method
        self.audio_image_root = Path(audio_image_root) if audio_image_root else None
        self.text_image_root = Path(text_image_root) if text_image_root else None
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.cifar_test = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

        self.cifar_to_local_idx = {}
        for i, c_name in enumerate(self.cifar_test.classes):
            for local_idx, (en, ko, slug) in enumerate(classes):
                if c_name == slug or c_name == en.lower().replace(" ", "_"):
                    self.cifar_to_local_idx[i] = local_idx
                    break

        self.audio_map = {}
        if self.audio_root.exists():
            for local_idx, (en, ko, slug) in enumerate(classes):
                p = self.audio_root / slug
                if p.exists():
                    files = list(p.glob("*.wav")) + list(p.glob("**/*.wav"))
                    if files:
                        self.audio_map[local_idx] = files[:cap_per_class]

        # optional precomputed audio images (PNG)
        self.audio_image_map = {}
        if self.audio_image_root and self.audio_image_root.exists():
            for local_idx, (en, ko, slug) in enumerate(classes):
                p = self.audio_image_root / slug
                if p.exists():
                    imgs = list(p.rglob("*.png"))
                    if imgs:
                        self.audio_image_map[local_idx] = imgs[:cap_per_class]
            print(f"✔ Using precomputed audio images from {self.audio_image_root} ({len(self.audio_image_map)} classes)")
        # optional precomputed text images
        self.text_image_map = {}
        if self.text_image_root and self.text_image_root.exists():
            for local_idx, (en, ko, slug) in enumerate(classes):
                p = self.text_image_root / slug / "text.png"
                if p.exists():
                    self.text_image_map[local_idx] = p
            print(f"✔ Using precomputed text images from {self.text_image_root} ({len(self.text_image_map)} classes)")

        per_class_counts = {i: 0 for i in range(len(classes))}
        self.samples = []
        for i, target in enumerate(self.cifar_test.targets):
            if target in self.cifar_to_local_idx:
                local_idx = self.cifar_to_local_idx[target]
                if per_class_counts[local_idx] >= cap_per_class:
                    continue
                self.samples.append(i)
                per_class_counts[local_idx] += 1

        print(f"Eval dataset: {len(self.samples)} samples ({len(self.audio_map)} classes with audio, cap {cap_per_class}/class)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cifar_idx = self.samples[idx]
        img, cifar_label = self.cifar_test[cifar_idx]
        local_idx = self.cifar_to_local_idx[cifar_label]
        class_slug = self.classes[local_idx][2]

        if local_idx in self.text_image_map:
            try:
                with Image.open(self.text_image_map[local_idx]).convert("RGB") as im:
                    im = im.resize((self.image_size, self.image_size))
                    text_img = torch.tensor(np.array(im).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
            except Exception:
                text_img = self.texture_factory.generate(class_slug, variation=0)
        else:
            text_img = self.texture_factory.generate(class_slug, variation=0)
        text_img = (text_img - self.mean) / self.std
        has_audio = 0
        audio_img = torch.zeros(3, self.image_size, self.image_size)

        # prefer precomputed PNG
        if local_idx in self.audio_image_map:
            try:
                audio_path = random.choice(self.audio_image_map[local_idx])
                with Image.open(audio_path).convert("RGB") as im:
                    im = im.resize((self.image_size, self.image_size))
                    audio_img = torch.tensor(np.array(im).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
                    audio_img = (audio_img - self.mean) / self.std
                    has_audio = 1
            except Exception:
                pass

        # fallback to wav -> image
        if has_audio == 0 and local_idx in self.audio_map:
            try:
                audio_path = random.choice(self.audio_map[local_idx])
                audio_img = audio_to_image(audio_path, self.image_size, self.audio_method)
                audio_img = (audio_img - self.mean) / self.std
                has_audio = 1
            except Exception:
                audio_img = torch.zeros(3, self.image_size, self.image_size)

        return {"image": img, "text": text_img, "audio": audio_img, "label": torch.tensor(local_idx), "has_audio": torch.tensor(has_audio)}


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


def visualize_retrieval_grid_improved(model, loader, device, classes, vis_class_subset=None, save_path="retrieval_grid_improved.png"):
    model.eval()
    print("\n🔍 Generating Improved Retrieval Grid...")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def denorm(img_t):
        x = img_t * std + mean
        x = x.clamp(0, 1)
        return x

    gallery_feats = []
    gallery_imgs = []
    gallery_lbls = []

    all_txt = []
    all_txt_lbl = []
    all_aud = []
    all_aud_lbl = []
    all_img = []
    all_img_lbl = []

    # for visualization: first occurrence per class
    query_samples = {}

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            texts = batch["text"].to(device)
            audios = batch["audio"].to(device)
            lbls = batch["label"].to(device)
            has_audio = batch["has_audio"]

            lat_i, _ = model(imgs)
            lat_t, _ = model(texts)
            lat_a_full = None
            if has_audio.any():
                lat_a_full, _ = model(audios)
            gallery_feats.append(lat_i.cpu())
            gallery_imgs.append(imgs.cpu())
            gallery_lbls.append(lbls.cpu())

            all_img.append(lat_i.cpu())
            all_img_lbl.append(lbls.cpu())
            all_txt.append(lat_t.cpu())
            all_txt_lbl.append(lbls.cpu())
            # audio latents for samples that have audio
            if has_audio.any():
                mask = has_audio > 0
                all_aud.append(lat_a_full[mask].cpu())
                all_aud_lbl.append(lbls[mask].cpu())

            for i in range(len(imgs)):
                l_idx = lbls[i].item()
                if l_idx not in query_samples:
                    query_samples[l_idx] = {
                        "text": texts[i].cpu(),
                        "audio": audios[i].cpu() if has_audio[i] > 0 else None,
                        "image": imgs[i].cpu(),
                        "name": classes[l_idx][0],
                    }

    gallery_feats = torch.cat(gallery_feats)
    gallery_imgs = torch.cat(gallery_imgs)
    gallery_lbls = torch.cat(gallery_lbls)
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(gallery_feats.numpy())

    # aggregate for global metrics
    def recall_at_k(query_feats, query_lbls, k=1):
        if len(query_feats) == 0:
            return 0, 0
        feats = torch.cat(query_feats).numpy()
        lbls = torch.cat(query_lbls).numpy()
        dists, idxs = nn_model.kneighbors(feats, n_neighbors=min(k, len(gallery_feats)))
        preds = gallery_lbls[idxs].numpy()
        hits = (preds == lbls[:, None]).any(axis=1)
        return hits.sum(), len(hits)

    t_hit1, t_tot = recall_at_k(all_txt, all_txt_lbl, k=1)
    t_hit5, _ = recall_at_k(all_txt, all_txt_lbl, k=5)
    a_hit1, a_tot = recall_at_k(all_aud, all_aud_lbl, k=1)
    a_hit5, _ = recall_at_k(all_aud, all_aud_lbl, k=5)
    i_hit1, i_tot = recall_at_k(all_img, all_img_lbl, k=1)
    # i_hit5 not used

    # visualization subset
    vis_keys = list(query_samples.keys())
    if vis_class_subset is not None:
        vis_keys = [k for k in vis_keys if k in vis_class_subset]
    random.shuffle(vis_keys)
    vis_keys = vis_keys[:10]

    rows = len(vis_keys)
    if rows == 0:
        print("⚠ No samples to visualize")
        return

    fig = plt.figure(figsize=(20, 3.4 * rows))
    gs = fig.add_gridspec(rows, 6, hspace=0.45, wspace=0.25)

    text_correct = audio_correct = image_correct = 0
    text_total = audio_total = image_total = 0
    text_top5 = audio_top5 = 0

    with torch.no_grad():
        for r, k in enumerate(vis_keys):
            s = query_samples[k]
            name = s["name"]

            ax = fig.add_subplot(gs[r, 0])
            ax.imshow(denorm(s["text"]).permute(1, 2, 0).numpy())
            ax.set_title(f"Text: {name}", fontsize=11, fontweight="bold")
            ax.axis("off")

            t_vec, _ = model(s["text"].unsqueeze(0).to(device))
            dists, idxs = nn_model.kneighbors([t_vec[0].cpu().numpy()], n_neighbors=5)
            idx = idxs[0][0]
            pred_lbl = gallery_lbls[idx].item()
            is_correct = pred_lbl == k
            text_correct += is_correct
            text_total += 1
            if k in gallery_lbls[idxs[0]].tolist():
                text_top5 += 1
            ax = fig.add_subplot(gs[r, 1])
            ax.imshow(denorm(gallery_imgs[idx]).permute(1, 2, 0).numpy())
            color = "green" if is_correct else "red"
            ax.set_title(f"Text→Pred: {classes[pred_lbl][0]}", color=color, fontsize=10, fontweight="bold")
            ax.axis("off")

            ax = fig.add_subplot(gs[r, 2])
            if s["audio"] is not None:
                a_vis = denorm(s["audio"]).permute(1, 2, 0).numpy()
                a_vis = (a_vis - a_vis.min()) / (a_vis.max() - a_vis.min() + 1e-8)
                ax.imshow(a_vis)
                ax.set_title(f"Audio: {name}", fontsize=11, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No Audio", ha="center", va="center", fontsize=10)
            ax.axis("off")

            ax = fig.add_subplot(gs[r, 3])
            if s["audio"] is not None:
                a_vec, _ = model(s["audio"].unsqueeze(0).to(device))
                dists, idxs = nn_model.kneighbors([a_vec[0].cpu().numpy()], n_neighbors=5)
                idx = idxs[0][0]
                pred_lbl = gallery_lbls[idx].item()
                is_correct = pred_lbl == k
                audio_correct += is_correct
                audio_total += 1
                if k in gallery_lbls[idxs[0]].tolist():
                    audio_top5 += 1
                ax.imshow(denorm(gallery_imgs[idx]).permute(1, 2, 0).numpy())
                color = "green" if is_correct else "red"
                ax.set_title(f"Audio→Pred: {classes[pred_lbl][0]}", color=color, fontsize=10, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No Audio", ha="center", va="center", fontsize=10)
            ax.axis("off")

            ax = fig.add_subplot(gs[r, 4])
            ax.imshow(denorm(s["image"]).permute(1, 2, 0).numpy())
            ax.set_title(f"Image: {name}", fontsize=11, fontweight="bold")
            ax.axis("off")

            ax = fig.add_subplot(gs[r, 5])
            img_vec, _ = model(s["image"].unsqueeze(0).to(device))
            dists, idxs = nn_model.kneighbors([img_vec[0].cpu().numpy()], n_neighbors=5)
            idx = idxs[0][0]
            pred_lbl = gallery_lbls[idx].item()
            is_correct = pred_lbl == k
            image_correct += is_correct
            image_total += 1
            ax.imshow(denorm(gallery_imgs[idx]).permute(1, 2, 0).numpy())
            color = "green" if is_correct else "red"
            ax.set_title(f"Image→Pred: {classes[pred_lbl][0]}", color=color, fontsize=10, fontweight="bold")
            ax.axis("off")

    def ratio(c, t):
        return (c / t * 100) if t > 0 else 0.0

    text_acc = ratio(text_correct, text_total)
    audio_acc = ratio(audio_correct, audio_total)
    img_acc = ratio(image_correct, image_total)
    text_r5 = ratio(text_top5, text_total)
    audio_r5 = ratio(audio_top5, audio_total)

    # global metrics (all samples)
    text_acc_global = ratio(t_hit1, t_tot)
    text_r5_global = ratio(t_hit5, t_tot)
    audio_acc_global = ratio(a_hit1, a_tot)
    audio_r5_global = ratio(a_hit5, a_tot)
    img_acc_global = ratio(i_hit1, i_tot)

    fig.suptitle(
        f"Text→Image R@1 {text_acc_global:.1f}% (vis {text_acc:.1f}%) | R@5 {text_r5_global:.1f}% (vis {text_r5:.1f}%) | "
        f"Audio→Image R@1 {audio_acc_global:.1f}% (vis {audio_acc:.1f}%) | R@5 {audio_r5_global:.1f}% (vis {audio_r5:.1f}%) | "
        f"Image→Image R@1 {img_acc_global:.1f}% (vis {img_acc:.1f}%)",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved {save_path}")


def visualize_tsne_improved(model, loader, device, classes, save_path="tsne_improved.png", max_classes=10, vis_class_subset=None):
    print("\n🎨 Generating Improved t-SNE...")
    model.eval()
    lats = {"image": [], "text": [], "audio": []}
    lbls = {"image": [], "text": [], "audio": []}

    with torch.no_grad():
        for b in loader:
            li, _ = model(b["image"].to(device))
            lt, _ = model(b["text"].to(device))
            la, _ = model(b["audio"].to(device))
            lats["image"].append(li.cpu())
            lats["text"].append(lt.cpu())
            lats["audio"].append(la.cpu())
            lbls["image"].append(b["label"])
            lbls["text"].append(b["label"])
            lbls["audio"].append(b["label"])

    all_vecs = []
    all_lbls = []
    all_types = []
    for k in lats:
        v = torch.cat(lats[k]).numpy()
        l = torch.cat(lbls[k]).numpy()
        if len(v) > 500:
            idx = np.random.choice(len(v), 500, replace=False)
            v = v[idx]
            l = l[idx]
        all_vecs.append(v)
        all_lbls.append(l)
        all_types.extend([k] * len(v))

    X = np.concatenate(all_vecs)
    y = np.concatenate(all_lbls)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 12))
    markers = {"image": "o", "text": "x", "audio": "^"}
    marker_names = {"image": "Image", "text": "Text", "audio": "Audio"}
    cmap = plt.get_cmap("tab20")
    unique_classes = np.unique(y)
    if vis_class_subset is not None:
        chosen = set(vis_class_subset)
    else:
        chosen = set(np.random.choice(unique_classes, max_classes, replace=False)) if len(unique_classes) > max_classes else set(unique_classes)

    types_arr = np.array(all_types)
    for cls_idx in unique_classes:
        color = cmap(cls_idx % 20)
        for mod in ["image", "text", "audio"]:
            mask = (y == cls_idx) & (types_arr == mod)
            if mask.any():
                label = f"{classes[cls_idx][0]} ({marker_names[mod]})" if cls_idx in chosen else None
                ax.scatter(
                    tsne[mask, 0],
                    tsne[mask, 1],
                    color=color,
                    marker=markers[mod],
                    label=label,
                    alpha=0.7,
                    s=40,
                    edgecolors="black",
                    linewidths=0.5,
                )

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, ncol=2)
    ax.set_title("Cross-Modal Latent Space (t-SNE)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved {save_path}")


if __name__ == "__main__":
    print(f"🚀 Loading model from {CONFIG['model_path']}...")
    random.seed(CONFIG["seed"])
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
        temp = datasets.CIFAR100(root="./data", train=False, download=True)
        classes = [(c, c, c.replace(" ", "_")) for c in temp.classes]

    vis_subset = None
    if CONFIG["vis_num_classes"] and CONFIG["vis_num_classes"] < len(classes):
        vis_subset = random.sample(list(range(len(classes))), CONFIG["vis_num_classes"])
        print(f"Visualization subset: {len(vis_subset)} classes (metrics use all classes)")

    # pick precomputed audio image root if present
    auto_audio_image_root = CONFIG["audio_image_root"]
    guess_root = Path(f"data/audio_img_eval_{CONFIG['audio_image_convert_method']}")
    if auto_audio_image_root is None and guess_root.exists():
        auto_audio_image_root = str(guess_root)

    eval_ds = EvalDataset(
        classes,
        CONFIG["audio_eval_root"],
        image_size=CONFIG["image_size"],
        cap_per_class=CONFIG["cap_per_class"],
        audio_method=CONFIG["audio_image_convert_method"],
        audio_image_root=auto_audio_image_root,
        text_image_root=CONFIG.get("text_image_root"),
    )
    eval_loader = DataLoader(eval_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = VisualByteNet(num_classes=len(classes)).to(CONFIG["device"])
    state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    # handle classifier shape mismatch when evaluating on a subset of classes
    clf_w = state_dict.get("classifier.weight")
    clf_b = state_dict.get("classifier.bias")
    if clf_w is not None and clf_w.shape[0] != model.classifier.weight.shape[0]:
        state_dict.pop("classifier.weight", None)
        state_dict.pop("classifier.bias", None)
        print(f"⚠ Classifier shape mismatch (ckpt {clf_w.shape[0]} vs eval {model.classifier.weight.shape[0]}). Reinitializing classifier and loading rest.")
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully!")

    vis_ids = vis_subset if vis_subset is not None else None
    visualize_retrieval_grid_improved(model, eval_loader, CONFIG["device"], classes, vis_class_subset=vis_ids)
    visualize_tsne_improved(model, eval_loader, CONFIG["device"], classes, vis_class_subset=vis_ids)
    print("\n✨ All Done!")
