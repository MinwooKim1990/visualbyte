import argparse
import hashlib
from typing import List, Tuple
from PIL import ImageOps 

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from sentence_transformers import SentenceTransformer


# ------------------------------
# StableTextureFactory (원래 쓰던 버전, image_size만 patch_size로 바꿔서 사용)
# ------------------------------
class StableTextureFactory:
    def __init__(self, image_size: int = 518):
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
                draw.ellipse(
                    [center - r_c, center - r_c, center + r_c, center + r_c],
                    outline=base_color,
                    width=stride // 2,
                )
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


# ------------------------------
# 유틸 함수들
# ------------------------------
def split_into_n_parts(text: str, n: int = 4) -> List[str]:
    """문자열을 길이 기준으로 n등분 (최대한 균등하게)"""
    length = len(text)
    if length == 0:
        return ["" for _ in range(n)]

    base = length // n
    rem = length % n
    parts = []
    start = 0
    for i in range(n):
        part_len = base + (1 if i < rem else 0)
        if part_len > 0:
            parts.append(text[start : start + part_len])
        else:
            parts.append("")
        start += part_len
    return parts


def tensor_to_pil(img_tensor: torch.Tensor, size: int = 56) -> Image.Image:
    img_np = img_tensor.detach().cpu().numpy()
    img_np = np.clip(img_np.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return img


def create_barcode_patch(value: float, patch_size: int = 56) -> Image.Image:
    """
    softmax attention 값(value ∈ [0,1])을 받아
    6자리 정밀도 바코드 → 가로/세로 스트라이프를 섞은 체크무늬 패치로 만든다.

    예) value = 0.123456
        int_val = 123456
        hi = 123  -> 가로 스트라이프 설정
        lo = 456  -> 세로 스트라이프 설정
    """
    # 1) 값 클램프 및 0~999999 정수로 스케일링
    v = max(0.0, min(1.0, float(value)))
    int_val = int(round(v * 999_999))  # 0 ~ 999999

    hi = int_val // 1000  # 상위 3자리 (0~999)
    lo = int_val % 1000   # 하위 3자리 (0~999)

    # 2) 가로/세로 줄 개수 결정 (3~12개 정도로 제한)
    num_h = 3 + (hi % 10)  # horizontal stripes
    num_v = 3 + (lo % 10)  # vertical stripes

    # 3) 기본 캔버스 생성
    img = Image.new("RGB", (patch_size, patch_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 4) 가로 스트라이프 그리기 (회색 계열)
    stripe_h = max(1, patch_size // (num_h * 2))  # stripe/spacing 비율 1:1
    y = 0
    for k in range(num_h):
        # hi 값과 k를 섞어서 intensity 변화
        intensity = 80 + ((hi + k * 17) % 176)  # 80~255
        y_end = min(patch_size, y + stripe_h)
        draw.rectangle([0, y, patch_size, y_end], fill=(intensity, intensity, intensity))
        y += stripe_h * 2
        if y >= patch_size:
            break

    # 5) 세로 스트라이프 그리기 (보라 계열) → 가로와 겹치면서 체크무늬 느낌
    stripe_v = max(1, patch_size // (num_v * 2))
    x = 0
    for k in range(num_v):
        intensity = 80 + ((lo + k * 29) % 176)  # 80~255
        x_end = min(patch_size, x + stripe_v)
        # 세로줄은 (R=intensity, B=intensity)로 색을 넣어서 가로줄과 시각적으로 섞이게 함
        draw.rectangle([x, 0, x_end, patch_size], fill=(intensity, 0, intensity))
        x += stripe_v * 2
        if x >= patch_size:
            break

    return img



def compute_part_attention(parts: List[str], model: SentenceTransformer, device: str) -> np.ndarray:
    with torch.no_grad():
        emb = model.encode(parts, convert_to_tensor=True, normalize_embeddings=True, device=device)
        scores = torch.matmul(emb, emb.T)  # (4,4)
        attn = torch.softmax(scores, dim=-1)
        return attn.cpu().numpy()


def create_semantic_hash_patches(
    text: str,
    model: SentenceTransformer,
    texture_factory: StableTextureFactory,
    device: str,
    num_patches: int = 6,
    patch_size: int = 56,
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    sentence embedding → num_patches개 segment로 나눔 → 각 segment를 해시 텍스처로 변환.
    디버그를 위해 segment 값도 같이 반환.
    """
    with torch.no_grad():
        emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True, device=device)  # (D,)

    segments = np.array_split(emb, num_patches)

    patch_images: List[Image.Image] = []
    for idx, seg in enumerate(segments):
        seg_str = ",".join(f"{v:.4f}" for v in seg)
        key = f"sem_chunk_{idx}:{seg_str}"
        tex_tensor = texture_factory.generate(key)
        patch_img = tensor_to_pil(tex_tensor, size=patch_size)
        patch_images.append(patch_img)

    return patch_images, segments


# ------------------------------
# 메인: 텍스트 → 224x224 이미지 생성 + 디버그 출력
# ------------------------------
def build_text_multimodal_image(
    text: str,
    texture_factory: StableTextureFactory,
    semantic_model: SentenceTransformer,
    device: str,
    patch_size: int = 56,
) -> Image.Image:
    grid_size = 4
    img_size = patch_size * grid_size
    canvas = Image.new("RGB", (img_size, img_size), color="black")

    # 1. 텍스트 4파트 분해
    parts = split_into_n_parts(text, n=4)

    print("\n==============================")
    print(f"[TEXT] '{text}'")
    print("  [4 parts]")
    for i, p in enumerate(parts):
        print(f"    part[{i}] (diagonal patch ({i},{i})) = '{p}' (len={len(p)})")

    # 2. 대각선 4패치: 각 파트 해시 이미지
    for i, part in enumerate(parts):
        tex_tensor = texture_factory.generate(part)
        patch_img = tensor_to_pil(tex_tensor, size=patch_size)
        left = i * patch_size
        top = i * patch_size
        canvas.paste(patch_img, (left, top))

        h_val = texture_factory._get_hash(part)
        print(f"    -> diagonal ({i},{i}) uses hash={h_val} for text='{part}'")

    # 3. attention-like matrix 계산 (4x4)
    attn = compute_part_attention(parts, semantic_model, device=device)  # (4,4)
    print("\n  [Attention matrix 4x4] (row i, col j)")
    with np.printoptions(precision=4, suppress=True):
        print(attn)

    # 4. 상단 삼각형 (i < j): barcode 패치
    print("\n  [Upper triangle patches: attention-based barcodes]")
    for i in range(grid_size):
        for j in range(i + 1, grid_size):
            value = float(attn[i, j])
            barcode_patch = create_barcode_patch(value, patch_size=patch_size)
            left = j * patch_size
            top = i * patch_size
            canvas.paste(barcode_patch, (left, top))

            print(
                f"    patch({i},{j}) for parts[{i}]='{parts[i]}' & parts[{j}]='{parts[j]}' "
                f"attn={value:.6f}"
            )

    # 5. 하단 삼각형 (i > j): sentence embedding 기반 semantic hash 패치 6개
    lower_coords: List[Tuple[int, int]] = [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
    semantic_patches, sem_segments = create_semantic_hash_patches(
        text, semantic_model, texture_factory, device=device, num_patches=len(lower_coords), patch_size=patch_size
    )

    print("\n  [Lower triangle patches: semantic-hash (sentence embedding segments)]")
    for idx, (coord, patch_img) in enumerate(zip(lower_coords, semantic_patches)):
        row, col = coord
        left = col * patch_size
        top = row * patch_size
        canvas.paste(patch_img, (left, top))

        seg = sem_segments[idx]
        seg_min = float(np.min(seg))
        seg_max = float(np.max(seg))
        seg_mean = float(np.mean(seg))
        print(
            f"    patch({row},{col}) <- semantic segment[{idx}] "
            f"len={len(seg)}, min={seg_min:.4f}, max={seg_max:.4f}, mean={seg_mean:.4f}"
        )

    print("==============================\n")
    return canvas


# ------------------------------
# main()
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Text → VisualByte-style multimodal texture image generator (debug)")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="입력 문자열",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="저장할 PNG 파일 이름 (기본값: output.png)",
    )
    parser.add_argument(
        "--semantic_model",
        type=str,
        default="sentence-transformers/static-similarity-mrl-multilingual-v1",
        help="SentenceTransformer 모델 이름 "
             "(예: sentence-transformers/static-similarity-mrl-multilingual-v1, intfloat/multilingual-e5-small)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=56,
        help="각 그리드 패치 크기 (기본 56 → 전체 224x224)",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    texture_factory = StableTextureFactory(image_size=args.patch_size)
    semantic_model = SentenceTransformer(args.semantic_model, device=device)

    img = build_text_multimodal_image(
        args.text,
        texture_factory=texture_factory,
        semantic_model=semantic_model,
        device=device,
        patch_size=args.patch_size,
    )

    img = ImageOps.grayscale(img).convert("RGB")

    img.save(args.output)
    print(f"Saved image for text '{args.text}' → {args.output}")


if __name__ == "__main__":
    main()
