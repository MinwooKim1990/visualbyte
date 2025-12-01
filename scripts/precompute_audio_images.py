"""
Precompute audio -> image (PNG) for train/eval to avoid on-the-fly conversion.

Hardcoded paths:
- Train wavs: data/audio/<class>/en/*.wav
- Eval wavs:  data/audio_eval_tts/<class>/en/*.wav
- Outputs:
    method 1 (whisper encoder map, fallback to log-mel):
        data/audio_img_train_1/<class>/en/*.png
        data/audio_img_eval_1/<class>/en/*.png
    method 2 (simple log-mel):
        data/audio_img_train_2/<class>/en/*.png
        data/audio_img_eval_2/<class>/en/*.png

Image size: 224 (matches v3 train/eval config)

Preprocess for consistency:
- Trim leading/trailing silence
- Force fixed duration (center-crop or pad) to TARGET_DURATION_SEC
- Then convert to image (Whisper or log-mel)

Deps: python-dotenv, pillow, numpy, torch
- method 1 uses transformers + torchaudio if available; otherwise falls back to log-mel.
"""
import os
import wave
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from dotenv import load_dotenv

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
    import librosa  # optional for better silence trim
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:
    import transformers  # noqa: F401
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


IMAGE_SIZE = 224
TARGET_DURATION_SEC = 2.0  # after trim, pad/crop to this duration for consistent visuals
SILENCE_TOP_DB = 40        # silence trim threshold (used if librosa available)

TRAIN_AUDIO_ROOT = Path("data/audio")
EVAL_AUDIO_ROOT = Path("data/audio_eval_tts")
OUT_TRAIN_1 = Path("data/audio_img_train_1")
OUT_EVAL_1 = Path("data/audio_img_eval_1")
OUT_TRAIN_2 = Path("data/audio_img_train_2")
OUT_EVAL_2 = Path("data/audio_img_eval_2")


def trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    # Prefer librosa's trim if available
    if _HAS_LIBROSA:
        trimmed, _ = librosa.effects.trim(audio, top_db=SILENCE_TOP_DB)
        if trimmed.size > 0:
            return trimmed.astype(np.float32)
    # Fallback: simple energy threshold
    thr = max(0.01, 0.05 * float(np.max(np.abs(audio)) + 1e-8))
    idx = np.where(np.abs(audio) > thr)[0]
    if idx.size == 0:
        return audio.astype(np.float32)
    return audio[idx[0] : idx[-1] + 1].astype(np.float32)


def load_audio(path: Path):
    if _HAS_SF:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    else:
        with wave.open(str(path), "rb") as wf:  # type: ignore[name-defined]
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32) / 32768.0
    audio = audio.astype(np.float32)
    # 1) trim leading/trailing silence
    audio = trim_silence(audio, sr)
    # 2) normalize length: time-stretch/compress to fixed duration (no zero padding to avoid black bands)
    target_len = int(sr * TARGET_DURATION_SEC)
    if audio.shape[0] < 2:
        audio = np.zeros(target_len, dtype=np.float32)
    else:
        xs = np.linspace(0, audio.shape[0] - 1, num=target_len)
        audio = np.interp(xs, np.arange(audio.shape[0]), audio).astype(np.float32)
    return audio, sr


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
        hidden = encoder_outputs.last_hidden_state  # (1, T, D)
        hmap = hidden.squeeze(0).cpu()  # (T, D)
        hmap = hmap.T.unsqueeze(0).unsqueeze(0)  # 1x1xD xT
        hmap = F.interpolate(hmap, size=(image_size, image_size), mode="bilinear", align_corners=False)
        img = hmap.squeeze(0).squeeze(0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img.repeat(3, 1, 1)
    except Exception:
        return audio_to_image_logmel(path, image_size)


def audio_to_image(path: Path, image_size: int, method: int) -> torch.Tensor:
    if method == 1:
        return audio_to_image_whisper(path, image_size)
    else:
        return audio_to_image_logmel(path, image_size)


def save_tensor_as_png(tensor: torch.Tensor, out_path: Path):
    arr = tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def convert_root(audio_root: Path, out_root: Path, method: int):
    wavs = list(audio_root.rglob("*.wav"))
    print(f"[method {method}] {audio_root} -> {out_root}, files: {len(wavs)}")
    for wav in wavs:
        # only use en subfolders
        if "en" not in wav.parts:
            continue
        rel = wav.relative_to(audio_root)
        out_path = out_root / rel.with_suffix(".png")
        if out_path.exists():
            continue
        try:
            img_tensor = audio_to_image(wav, IMAGE_SIZE, method)
            save_tensor_as_png(img_tensor, out_path)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Failed {wav}: {e}")


def main():
    load_dotenv()
    # method 1
    convert_root(TRAIN_AUDIO_ROOT, OUT_TRAIN_1, method=1)
    convert_root(EVAL_AUDIO_ROOT, OUT_EVAL_1, method=1)
    # method 2
    convert_root(TRAIN_AUDIO_ROOT, OUT_TRAIN_2, method=2)
    convert_root(EVAL_AUDIO_ROOT, OUT_EVAL_2, method=2)
    print("Done.")


if __name__ == "__main__":
    main()
