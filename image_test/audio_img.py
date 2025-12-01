# audio_2x2_img.py
# 오디오를 2x2 타일 이미지(스펙 + attention + embedding-hash)로 변환

import wave
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

# ----------------------------------------------------
# 선택적 의존성
# ----------------------------------------------------
try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

try:
    import torchaudio
    _HAS_TA = True
except ImportError:
    _HAS_TA = False


# ----------------------------------------------------
# 0. 오디오 로딩
# ----------------------------------------------------
def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """WAV 로드 -> mono float32 [-1,1]."""
    if _HAS_SF:
        audio, sr = sf.read(str(path))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    else:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            audio = np.frombuffer(
                wf.readframes(n_frames),
                dtype=np.int16,
            ).astype(np.float32) / 32768.0
    return audio.astype(np.float32), int(sr)


# ----------------------------------------------------
# 1. 무음 + 약한 노이즈 트리밍 (waveform 기준)
# ----------------------------------------------------
def trim_silence(
    audio_t: torch.Tensor,
    sr: int,
    top_db: float = 40.0,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    """
    앞/뒤 무음 + 아주 약한 노이즈 구간을 잘라낸다.
    frame RMS dB 기준으로 max_db - top_db 아래는 잘라냄.
    """
    if audio_t.numel() == 0:
        return audio_t

    x = audio_t.unsqueeze(0).unsqueeze(0)  # (1,1,T)

    window = torch.ones(1, 1, frame_length, device=audio_t.device) / frame_length
    power = F.conv1d(x ** 2, window, stride=hop_length)  # (1,1,N_frames)
    rms = torch.sqrt(power + 1e-8)
    db = 20.0 * torch.log10(rms + 1e-8).squeeze()  # (N_frames,)

    if torch.isneginf(db).all():
        return audio_t

    max_db = db.max()
    mask = db > (max_db - top_db)

    if not mask.any():
        return audio_t

    idx = torch.nonzero(mask, as_tuple=False).squeeze()
    if idx.dim() == 0:
        idx = idx.unsqueeze(0)

    first_frame = int(idx[0].item())
    last_frame = int(idx[-1].item())

    start_sample = max(0, first_frame * hop_length)
    end_sample = min(audio_t.shape[-1], last_frame * hop_length + frame_length)

    trimmed = audio_t[start_sample:end_sample]

    if trimmed.numel() < frame_length:
        return audio_t

    return trimmed


# ----------------------------------------------------
# 2. log-mel 스펙트로그램
# ----------------------------------------------------
def wav_to_logmel(
    audio_t: torch.Tensor,
    sr: int,
    n_mels: int = 80,
) -> torch.Tensor:
    """
    (F, T) = (mel_bins, time_frames)
    """
    if _HAS_TA:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=n_mels,
            power=2.0,
        )
        mel = mel_transform(audio_t)  # (F,T)
        mel = torch.log1p(mel)
    else:
        n_fft = 1024
        hop_length = 256
        window = torch.hann_window(n_fft)
        spec = torch.stft(
            audio_t,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        mag = torch.log1p(spec.abs())  # (F_raw, T)
        mag = mag.unsqueeze(0).unsqueeze(0)  # (1,1,F_raw,T)
        mel = F.interpolate(
            mag,
            size=(n_mels, mag.shape[-1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (F,T)
    return mel


# ----------------------------------------------------
# 2.5. mel 상에서 활성 시간 구간만 crop
# ----------------------------------------------------
def crop_mel_active_region(
    mel: torch.Tensor,
    energy_ratio: float = 0.1,
    min_frames: int = 4,
) -> torch.Tensor:
    """
    mel(F,T) 에서 시간별 평균 에너지 기준으로
    실제 발화 구간만 남김.
    """
    energy = mel.mean(dim=0)  # (T,)
    max_e = energy.max()
    if max_e <= 0:
        return mel

    thr = max_e * energy_ratio
    mask = energy > thr
    if not mask.any():
        return mel

    idx = torch.nonzero(mask, as_tuple=False).squeeze()
    if idx.dim() == 0:
        idx = idx.unsqueeze(0)

    start = int(idx[0].item())
    end = int(idx[-1].item()) + 1

    if end - start < min_frames:
        return mel
    return mel[:, start:end]


# ----------------------------------------------------
# 3. 기본 바코드형 패턴 (0,0 타일)
# ----------------------------------------------------
def make_base_pattern(
    mel: torch.Tensor,
    patch_size: int,
    n_freq_bins: int = 24,
    n_time_bins: int = 32,
    n_levels: int = 4,
) -> torch.Tensor:
    """
    log-mel(F,T) -> coarse grid(F',T') -> 양자화 -> patch_size x patch_size
    출력 shape: (1, patch_size, patch_size) [0~1]
    """
    mel_min = mel.min()
    mel_max = mel.max()
    mel_norm = (mel - mel_min) / (mel_max - mel_min + 1e-6)  # (F,T)

    x = mel_norm.unsqueeze(0).unsqueeze(0)  # (1,1,F,T)
    pooled = F.adaptive_avg_pool2d(x, output_size=(n_freq_bins, n_time_bins)).squeeze(0).squeeze(0)  # (F',T')

    q = torch.clamp(
        torch.round(pooled * (n_levels - 1)),
        0,
        n_levels - 1,
    )
    q = q / (n_levels - 1)  # (F',T')

    q = q.unsqueeze(0).unsqueeze(0)  # (1,1,F',T')
    img = F.interpolate(q, size=(patch_size, patch_size), mode="nearest").squeeze(0).squeeze(0)  # (H,W)
    return img.unsqueeze(0)  # (1,H,W)


# ----------------------------------------------------
# 4. segment 임베딩 & attention 패턴 (0,1 / 1,0 타일)
# ----------------------------------------------------
def compute_segment_embeddings(
    mel: torch.Tensor,
    n_segments: int = 4,
) -> torch.Tensor:
    """
    mel(F,T)을 시간축으로 n_segments 등분해서
    각 segment에 대해 간단한 mean-pool 임베딩 추출.
    반환: (n_segments, F)
    """
    F_dim, T = mel.shape
    seg_len = max(1, T // n_segments)
    embs = []
    for i in range(n_segments):
        start = i * seg_len
        end = T if i == n_segments - 1 else min(T, (i + 1) * seg_len)
        if start >= end:
            seg = mel
        else:
            seg = mel[:, start:end]
        emb = seg.mean(dim=1)  # (F,)
        embs.append(emb)
    embs_t = torch.stack(embs, dim=0)  # (N,F)
    return embs_t


def make_attention_patch(
    seg_embs: torch.Tensor,
    patch_size: int,
    upper: bool = True,
) -> torch.Tensor:
    """
    segment 임베딩들로 N x N cosine similarity matrix 만든 뒤,
    이를 patch_size x patch_size 이미지로 확장.
    upper=True이면 상삼각을 강조, False이면 하삼각을 강조하는 식으로
    살짝 다르게 만들 수 있음.
    출력: (1, patch_size, patch_size)
    """
    # seg_embs: (N,F)
    N, F_dim = seg_embs.shape
    norm = seg_embs / (seg_embs.norm(dim=1, keepdim=True) + 1e-8)
    sim = norm @ norm.T  # (N,N), -1~1

    # 0~1로 정규화
    sim_norm = (sim + 1.0) / 2.0  # (N,N)

    if upper:
        mask = torch.triu(torch.ones_like(sim_norm), diagonal=1)
    else:
        mask = torch.tril(torch.ones_like(sim_norm), diagonal=-1)

    # 삼각 부분만 남기고 나머지 0
    sim_tri = sim_norm * mask

    sim_tri = sim_tri.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
    img = F.interpolate(sim_tri, size=(patch_size, patch_size), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)  # (H,W)
    return img.unsqueeze(0)  # (1,H,W)


# ----------------------------------------------------
# 5. 임베딩 기반 hash 패턴 (1,1 타일)
# ----------------------------------------------------
def make_embedding_hash_patch(
    mel: torch.Tensor,
    patch_size: int,
    hash_grid: Tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """
    mel(F,T) -> 전역 임베딩 (F,) -> random projection + sign -> bit grid (H,W)
    -> patch_size x patch_size 로 확대.
    """
    F_dim, T = mel.shape
    # 간단한 전역 임베딩: 시간 평균
    global_emb = mel.mean(dim=1)  # (F,)

    # 랜덤 projection (고정 seed로 deterministic)
    H_bits, W_bits = hash_grid
    K = H_bits * W_bits

    g = torch.Generator(device=mel.device).manual_seed(0)
    W = torch.randn(K, F_dim, generator=g, device=mel.device)  # (K,F)

    proj = W @ global_emb  # (K,)
    bits = (proj >= 0).float()  # 0/1

    grid = bits.view(H_bits, W_bits)  # (H_bits, W_bits)

    grid = grid.unsqueeze(0).unsqueeze(0)  # (1,1,H_bits,W_bits)
    img = F.interpolate(grid, size=(patch_size, patch_size), mode="nearest").squeeze(0).squeeze(0)  # (H,W)
    return img.unsqueeze(0)  # (1,H,W)


# ----------------------------------------------------
# 6. 전체 2x2 타일 이미지 구성
# ----------------------------------------------------
def audio_to_2x2_image_tensor(
    path: Path,
    image_size: int = 224,
    n_mels: int = 80,
    trim_top_db: float = 40.0,
) -> torch.Tensor:
    """
    최종 출력: (3, image_size, image_size), 0~1 범위
      [0,0] : base barcoded mel pattern
      [0,1] : attention upper tri
      [1,0] : attention lower tri
      [1,1] : embedding-hash pattern
    """
    patch_size = image_size // 2

    # 1) 오디오 로드 & 무음 트림
    audio, sr = load_audio(path)
    audio_t = torch.tensor(audio, dtype=torch.float32)
    audio_t = trim_silence(audio_t, sr=sr, top_db=trim_top_db)

    # 2) log-mel 및 활성 구간 crop
    mel = wav_to_logmel(audio_t, sr=sr, n_mels=n_mels)  # (F,T)
    mel = crop_mel_active_region(mel, energy_ratio=0.1, min_frames=4)  # (F,T_active)

    # 3) 타일별 이미지 생성 (1채널)
    base_patch = make_base_pattern(mel, patch_size=patch_size)                # (1,H,W)
    seg_embs = compute_segment_embeddings(mel, n_segments=4)                  # (4,F)
    attn_upper = make_attention_patch(seg_embs, patch_size=patch_size, upper=True)
    attn_lower = make_attention_patch(seg_embs, patch_size=patch_size, upper=False)
    hash_patch = make_embedding_hash_patch(mel, patch_size=patch_size)

    # 4) 캔버스에 배치
    canvas = torch.zeros(1, image_size, image_size)  # (1,H,W)

    # 좌상 (0,0)
    canvas[:, 0:patch_size, 0:patch_size] = base_patch

    # 우상 (0,1)
    canvas[:, 0:patch_size, patch_size:image_size] = attn_upper

    # 좌하 (1,0)
    canvas[:, patch_size:image_size, 0:patch_size] = attn_lower

    # 우하 (1,1)
    canvas[:, patch_size:image_size, patch_size:image_size] = hash_patch

    # 1채널 -> 3채널 grayscale
    img_3c = canvas.repeat(3, 1, 1)  # (3,H,W)
    img_3c = img_3c.clamp(0.0, 1.0)

    return img_3c


# ----------------------------------------------------
# 7. 외부에서 쓸 API
# ----------------------------------------------------
def audio_to_image(
    path: Path,
    image_size: int = 224,
) -> torch.Tensor:
    return audio_to_2x2_image_tensor(path, image_size=image_size)


# ----------------------------------------------------
# 8. 단독 실행 테스트
# ----------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    wav_path = Path("tts_1.wav")  # 여기에 테스트 wav 파일 이름 넣기

    img = audio_to_image(wav_path, image_size=224)  # (3,224,224)

    # PNG로 저장
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save("audio_2x2_output.png")
    print("saved: audio_2x2_output.png")

    # 화면에 표시
    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    plt.show()