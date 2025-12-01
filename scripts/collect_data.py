"""
Data collector for CIFAR-100-style labels.

Features:
- Reads class names from a CSV (`en,ko` columns). Korean can be left blank; English will be reused.
- Generates multi-voice TTS audio (EN/KO) via OpenAI Audio API; prepends a small silence to avoid clipped leading phonemes.
- Images: default source is torchvision CIFAR-100 (stable, no rate limit). Optional web search via DuckDuckGo, Brave API, or headless Selenium+BS4 scrape.

Outputs (per class):
- data/image/<class_slug>/img_1.jpg ... img_N.jpg
- data/audio/<class_slug>/en/<voice>.wav
- data/audio/<class_slug>/ko/<voice>.wav

Prereqs (install):
    pip install openai python-dotenv pillow imagehash requests numpy soundfile
    # Optional for CIFAR-100 images
    pip install torch torchvision
    # Optional for web/Brave search
    pip install duckduckgo-search
    # Optional for Selenium scrape
    pip install selenium beautifulsoup4

Env:
    OPENAI_API_KEY must be set (e.g., via .env).
"""
import argparse
import csv
from io import BytesIO
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import requests
import soundfile as sf
from openai import OpenAI
from PIL import Image, UnidentifiedImageError
import imagehash
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "image"
AUDIO_DIR = DATA_DIR / "audio"
DEFAULT_CLASSES_CSV = DATA_DIR / "classes_cifar100.csv"

SUPPORTED_VOICES = ["nova", "shimmer", "echo", "onyx", "fable", "alloy", "ash", "sage", "coral"]
DEFAULT_EN_VOICES = ["alloy", "fable", "onyx", "shimmer"]
DEFAULT_KO_VOICES = ["alloy", "fable", "onyx", "shimmer"]
DEFAULT_IMAGES_PER_CLASS = 5
DEFAULT_TTS_MODEL = "tts-1"
DEFAULT_TTS_PRE_SILENCE_MS = 400  # prepend silence to avoid clipped leading phoneme
IMAGE_SOURCES = ["cifar100", "web", "brave", "selenium"]


def slugify(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def display_text(label: str) -> str:
    """Make TTS-friendly text (replace underscores)."""
    return label.replace("_", " ").strip()


def load_classes(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Class CSV not found: {csv_path}")
    rows = []
    # Use utf-8-sig to strip potential BOM from files edited on Windows.
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize fieldnames in case BOM snuck into the header.
            en = (row.get("en") or row.get("\ufeffen") or "").strip()
            ko = row.get("ko", "").strip()
            if not en:
                continue
            if not ko:
                ko = en
            rows.append({"en": en, "ko": ko, "slug": slugify(en)})
    if not rows:
        raise ValueError("No classes loaded from CSV.")
    return rows


def ensure_dirs(label_slug: str) -> None:
    (IMAGE_DIR / label_slug).mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / label_slug / "en").mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / label_slug / "ko").mkdir(parents=True, exist_ok=True)


def synthesize_tts(
    client: OpenAI,
    text: str,
    voice: str,
    out_path: Path,
    model: str = DEFAULT_TTS_MODEL,
    pre_silence_ms: int = DEFAULT_TTS_PRE_SILENCE_MS,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[TTS] {out_path}")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav",
    ) as resp:
        resp.stream_to_file(tmp_path)

    # Prepend silence to avoid clipped leading phoneme
    audio, sr = sf.read(tmp_path)
    if pre_silence_ms > 0:
        pad = np.zeros((int(sr * pre_silence_ms / 1000),) + (() if audio.ndim == 1 else (audio.shape[1],)), dtype=audio.dtype)
        audio = np.concatenate([pad, audio], axis=0)

    sf.write(out_path, audio, sr)
    tmp_path.unlink(missing_ok=True)


def download_image(url: str, out_path: Path) -> Optional[Image.Image]:
    headers = {"User-Agent": "data-collector/0.1"}
    try:
        resp = requests.get(url, headers=headers, timeout=(5, 15))
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG")
        return img
    except (requests.RequestException, UnidentifiedImageError) as exc:
        print(f"[image] skip {url} ({exc})")
        return None


def collect_images_for_label(
    label_en: str,
    label_slug: str,
    max_images: int,
    query_suffix: str,
) -> None:
    from duckduckgo_search import DDGS  # Lazy import; used only for web mode.
    from duckduckgo_search.exceptions import RatelimitException

    query = f"{display_text(label_en)} {query_suffix}".strip()
    print(f"[image] Searching '{query}'")
    max_results = max_images * 6  # fetch extra to survive filtering
    saved = 0
    hashes: Set[str] = set()

    try:
        with DDGS() as ddgs:
            for result in ddgs.images(
                query,
                max_results=max_results,
                safesearch="moderate",
                size="Medium",
                type_image="photo",
            ):
                url = result.get("image")
                if not url:
                    continue
                img_path = IMAGE_DIR / label_slug / f"img_{saved + 1}.jpg"
                img = download_image(url, img_path)
                if img is None:
                    continue
                ph = imagehash.phash(img)
                if ph in hashes:
                    img_path.unlink(missing_ok=True)
                    continue
                hashes.add(ph)
                saved += 1
                if saved >= max_images:
                    break
    except RatelimitException as exc:
        print(f"[image] rate limited for query '{query}' ({exc}); skipping this class.")
        time.sleep(1.0)

    print(f"[image] {label_en}: saved {saved}/{max_images}")


def collect_images_brave(
    label_en: str,
    label_slug: str,
    max_images: int,
    query_suffix: str,
    api_key: str,
) -> None:
    """Use Brave Search Images API (requires BRAVE_API_KEY)."""
    query = f"{display_text(label_en)} {query_suffix}".strip()
    print(f"[image/brave] Searching '{query}'")
    url = "https://api.search.brave.com/res/v1/images/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params = {
        "q": query,
        "count": max_images * 5,
        "search_lang": "en",
        "country": "us",
        "safesearch": "moderate",
        "size": "medium",
    }

    saved = 0
    hashes: Set[str] = set()

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=(5, 15))
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"[image/brave] request failed: {exc}")
        return

    results = data.get("results", [])
    for item in results:
        img_url = item.get("properties", {}).get("url") or item.get("url")
        if not img_url:
            continue
        img_path = IMAGE_DIR / label_slug / f"img_{saved + 1}.jpg"
        img = download_image(img_url, img_path)
        if img is None:
            continue
        ph = imagehash.phash(img)
        if ph in hashes:
            img_path.unlink(missing_ok=True)
            continue
        hashes.add(ph)
        saved += 1
        if saved >= max_images:
            break

    print(f"[image/brave] {label_en}: saved {saved}/{max_images}")


def collect_images_selenium(
    label_en: str,
    label_slug: str,
    max_images: int,
    query_suffix: str,
    driver_path: Optional[str] = None,
) -> None:
    """
    Use Selenium (headless Chrome) + BeautifulSoup to scrape Bing Images without API keys.
    Requires: selenium, bs4, a Chrome/Edge browser + matching driver in PATH or --selenium-driver.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from bs4 import BeautifulSoup
    from urllib.parse import quote
    import time

    query = f"{display_text(label_en)} {query_suffix}".strip()
    print(f"[image/selenium] Searching '{query}'")

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    service = Service(executable_path=driver_path) if driver_path else Service()

    try:
        driver = webdriver.Chrome(service=service, options=opts)
    except Exception as exc:
        print(f"[image/selenium] Failed to start Chrome driver ({exc}); skipping.")
        return

    try:
        search_url = f"https://www.bing.com/images/search?q={quote(query)}&form=HDRSC2&first=1"
        driver.get(search_url)
        # Scroll to load more thumbs
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.0)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        imgs = soup.find_all("img")

        saved = 0
        hashes: Set[str] = set()

        for img in imgs:
            url = img.get("src") or img.get("data-src") or img.get("data-lazy")
            if not url or not url.startswith("http"):
                continue
            img_path = IMAGE_DIR / label_slug / f"img_{saved + 1}.jpg"
            pil_img = download_image(url, img_path)
            if pil_img is None:
                continue
            ph = imagehash.phash(pil_img)
            if ph in hashes:
                img_path.unlink(missing_ok=True)
                continue
            hashes.add(ph)
            saved += 1
            if saved >= max_images:
                break

        print(f"[image/selenium] {label_en}: saved {saved}/{max_images}")
    finally:
        driver.quit()


def collect_images_from_cifar100(
    label_to_slug: Dict[str, str],
    max_images: int,
    use_test: bool = True,
) -> None:
    """
    Sample images directly from torchvision CIFAR-100 train (and optionally test) sets.
    Avoids web rate limits and licensing ambiguity.
    """
    import torch
    from torchvision import datasets, transforms

    target_dir = IMAGE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR100(root=DATA_DIR / "torchvision", train=True, download=True, transform=tfm)
    datasets_list = [train_ds]
    if use_test:
        test_ds = datasets.CIFAR100(root=DATA_DIR / "torchvision", train=False, download=True, transform=tfm)
        datasets_list.append(test_ds)

    counts = {slug: 0 for slug in label_to_slug.values()}
    need = max_images
    # Build mapping from dataset class name to slug
    name_to_slug = label_to_slug

    for dataset in datasets_list:
        for img_tensor, target in dataset:
            class_name = dataset.classes[target]
            slug = name_to_slug.get(class_name)
            if slug is None:
                continue
            if counts[slug] >= need:
                if all(v >= need for v in counts.values()):
                    break
                continue
            # img_tensor is CxHxW in [0,1]; convert to PIL
            pil_img = transforms.ToPILImage()(img_tensor)
            out_path = target_dir / slug / f"img_{counts[slug] + 1}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(out_path, format="JPEG")
            counts[slug] += 1
        # If all done, stop outer loop
        if all(v >= need for v in counts.values()):
            break

    for cls, cnt in counts.items():
        print(f"[cifar100] {cls}: saved {cnt}/{need}")


def collect_tts_for_label(
    client: OpenAI,
    label_en: str,
    label_ko: str,
    label_slug: str,
    voices_en: Sequence[str],
    voices_ko: Sequence[str],
    model: str,
    pre_silence_ms: int,
) -> None:
    # Filter unsupported voices to avoid 400 errors
    voices_en = [v for v in voices_en if v in SUPPORTED_VOICES]
    voices_ko = [v for v in voices_ko if v in SUPPORTED_VOICES]

    text_en = display_text(label_en)
    text_ko = label_ko

    for voice in voices_en:
        out_path = AUDIO_DIR / label_slug / "en" / f"{voice}.wav"
        if out_path.exists():
            continue
        synthesize_tts(client, text_en, voice, out_path, model=model, pre_silence_ms=pre_silence_ms)

    for voice in voices_ko:
        out_path = AUDIO_DIR / label_slug / "ko" / f"{voice}.wav"
        if out_path.exists():
            continue
        synthesize_tts(client, text_ko, voice, out_path, model=model, pre_silence_ms=pre_silence_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect images and TTS audio per class.")
    parser.add_argument(
        "--classes-csv",
        type=Path,
        default=DEFAULT_CLASSES_CSV,
        help="CSV with 'en,ko' columns.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=DEFAULT_IMAGES_PER_CLASS,
        help="Number of images to download per class.",
    )
    parser.add_argument(
        "--image-source",
        choices=IMAGE_SOURCES,
        default="cifar100",
        help="Where to get images: 'cifar100' (torchvision), 'web' (DuckDuckGo), 'brave' (Brave API, needs BRAVE_API_KEY), or 'selenium' (headless Bing scrape).",
    )
    parser.add_argument(
        "--cifar-use-test",
        action="store_true",
        help="When using CIFAR-100 source, also sample from the test split.",
    )
    parser.add_argument(
        "--image-query-suffix",
        default="photo",
        help="Extra words appended to the image search query (web source only).",
    )
    parser.add_argument(
        "--voices-en",
        nargs="+",
        default=DEFAULT_EN_VOICES,
        help="Voices for English TTS (supported: nova, shimmer, echo, onyx, fable, alloy, ash, sage, coral).",
    )
    parser.add_argument(
        "--voices-ko",
        nargs="+",
        default=DEFAULT_KO_VOICES,
        help="Voices for Korean TTS (supported: nova, shimmer, echo, onyx, fable, alloy, ash, sage, coral).",
    )
    parser.add_argument(
        "--tts-model",
        default=DEFAULT_TTS_MODEL,
        help="OpenAI TTS model (e.g., tts-1, gpt-4o-mini-tts).",
    )
    parser.add_argument(
        "--tts-pre-silence-ms",
        type=int,
        default=DEFAULT_TTS_PRE_SILENCE_MS,
        help="Milliseconds of leading silence to prepend to synthesized audio (to avoid clipped leading phoneme).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image download.",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS synthesis.",
    )
    parser.add_argument(
        "--selenium-driver",
        default=None,
        help="Path to Chrome/Edge driver executable (optional if driver is on PATH).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    classes = load_classes(args.classes_csv)
    client = None if args.skip_tts else OpenAI()

    print(f"Loaded {len(classes)} classes from {args.classes_csv}")

    label_to_slug = {item["en"]: item["slug"] for item in classes}

    if not args.skip_images and args.image_source == "cifar100":
        collect_images_from_cifar100(
            label_to_slug=label_to_slug,
            max_images=args.images_per_class,
            use_test=args.cifar_use_test,
        )

    for idx, item in enumerate(classes, 1):
        label_en = item["en"]
        label_ko = item["ko"]
        slug = item["slug"]

        print(f"\n[{idx}/{len(classes)}] {label_en} ({slug})")
        ensure_dirs(slug)

        if not args.skip_images and args.image_source == "web":
            collect_images_for_label(
                label_en,
                slug,
                max_images=args.images_per_class,
                query_suffix=args.image_query_suffix,
            )
        if not args.skip_images and args.image_source == "brave":
            api_key = os.environ.get("BRAVE_API_KEY")
            if not api_key:
                print("[image/brave] Missing BRAVE_API_KEY; skipping images for this class.")
            else:
                collect_images_brave(
                    label_en,
                    slug,
                    max_images=args.images_per_class,
                    query_suffix=args.image_query_suffix,
                    api_key=api_key,
                )
        if not args.skip_images and args.image_source == "selenium":
            collect_images_selenium(
                label_en,
                slug,
                max_images=args.images_per_class,
                query_suffix=args.image_query_suffix,
                driver_path=args.selenium_driver,
            )

        if not args.skip_tts and client is not None:
            collect_tts_for_label(
                client,
                label_en,
                label_ko,
                slug,
                voices_en=args.voices_en,
                voices_ko=args.voices_ko,
                model=args.tts_model,
                pre_silence_ms=args.tts_pre_silence_ms,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
