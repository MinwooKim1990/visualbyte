# Visual Byte: Texture-Biased Vision Backbone for Cross-Modal Alignment of Text, Audio, and Images

## Abstract
We propose **Visual Byte**, a single vision-backbone approach for aligning text, audio, and images without language or speech encoders at inference time. Text and audio are rendered into texture-like images exploiting the texture bias of modern convolutional vision transformers (e.g., ConvNeXt). A class-balanced batching scheme enforces co-occurrence of modalities per class, and consistent normalization across modalities stabilizes training. Experiments on CIFAR-100-derived multimodal data show that image–text alignment remains strong while audio–image alignment improves as audio quality and preprocessing are refined. The pipeline is edge-friendly: audio-to-image conversion can be precomputed offline, and a quantized ConvNeXt-tiny fits mobile/robot NPU constraints. We release scripts for data synthesis, quiet-audio repair, and precomputation.

## 1. Introduction
Edge devices (phones/robots) have limited compute and memory, making LLM or large audio models impractical. We ask: **Can a single vision backbone align text and audio with images if non-visual modalities are converted into texture-like images?** Prior work indicates CNNs and ViT variants exhibit texture bias (Geirhos et al., 2019). We leverage this inductive bias by rendering text as geometric barcodes and audio as spectrogram-like images, and train a unified ConvNeXt encoder with contrastive and classification objectives.

## 2. Data Construction
### 2.1 Classes
- CIFAR-100 label set (100 classes); CSV with columns `en, ko`; slug = lowercase + underscores.
- Optional small subsets (e.g., 5–10 classes) for rapid iteration.

### 2.2 Images
- Source: CIFAR-100 train/test; cap per class (5–8) for balance.
- Stored at `data/image/<class_slug>/img_*.jpg`.

### 2.3 Text → Image (Texture Rendering)
- Deterministic hash-based patterns (stripes, grids, checker, dots, concentric, noise); color derived from hash of class string.
- Optional perturbation (invert, noise, blur).
- Output: 224×224 RGB; normalized with ImageNet mean/std.

### 2.4 Audio → Image
1) **TTS**: OpenAI `gpt-4o-mini-tts` (voices alloy, fable, onyx, shimmer; coral extra). Evaluation uses distinct voices/prompts (OpenAI/Gemini possible).
2) **Preprocess**: trim silence; resample/time-warp to fixed 2.0 s; optional 16 kHz resample for Whisper.
3) **Backends**:  
   - Method 1: Whisper encoder map (16 kHz) → interpolate to 224×224 → normalize.  
   - Method 2: Log-mel spectrogram (torchaudio/STFT) → log1p → interpolate → normalize.
4) **Precompute caches**: `data/audio_img_train_1/2`, `data/audio_img_eval_1/2` (PNG) to avoid on-the-fly conversion.
5) **Quiet-audio repair**: detect peak < threshold; re-synthesize TTS; delete stale PNGs; regenerate.

### 2.5 Normalization
All modalities (image, text-render, audio-image) share the same normalization: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]. Even without pretrained weights, consistent scaling is critical for stable contrastive alignment.

## 3. Model
- Backbone: ConvNeXt-Tiny (pretrained optional; ~28M params).
- Projector: Linear → GELU → Linear → L2 normalize.
- Classifier: Linear(num_classes); trainable temperature for logits; learnable `logit_scale` for contrastive heads.

## 4. Objectives
- `loss_cls`: CE for image logits + CE for text logits (classification).
- `loss_clip`: Symmetric CE for image↔text similarity (CLIP-style).
- `loss_audio`: CE for audio↔image similarity (only when audio exists).
- `loss_audio_text`: CE for audio↔text similarity.
- Default weights = 1; optionally downweight audio losses (λ<1) or warm-start them if text alignment deteriorates.

### Loss Flow (schematic)
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="620" height="320">
  <rect x="20" y="40" width="140" height="50" fill="#f2f2f2" stroke="#666"/><text x="90" y="70" text-anchor="middle" font-size="13">Image</text>
  <rect x="20" y="135" width="140" height="50" fill="#f2f2f2" stroke="#666"/><text x="90" y="165" text-anchor="middle" font-size="13">Text-render</text>
  <rect x="20" y="230" width="140" height="50" fill="#f2f2f2" stroke="#666"/><text x="90" y="260" text-anchor="middle" font-size="13">Audio-image</text>
  <rect x="190" y="40" width="140" height="240" fill="#d9e8ff" stroke="#3366cc"/><text x="260" y="165" text-anchor="middle" font-size="13">ConvNeXt</text>
  <rect x="360" y="40" width="140" height="240" fill="#fce5cd" stroke="#cc6600"/><text x="430" y="165" text-anchor="middle" font-size="13">Projector</text>
  <rect x="530" y="135" width="70" height="50" fill="#ffe599" stroke="#d4aa00"/><text x="565" y="165" text-anchor="middle" font-size="13">CE</text>
  <!-- arrows -->
  <defs><marker id="a" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="#333"/></marker></defs>
  <line x1="160" y1="65" x2="190" y2="65" stroke="#333" marker-end="url(#a)"/>
  <line x1="160" y1="160" x2="190" y2="160" stroke="#333" marker-end="url(#a)"/>
  <line x1="160" y1="255" x2="190" y2="255" stroke="#333" marker-end="url(#a)"/>
  <line x1="330" y1="65" x2="360" y2="65" stroke="#333" marker-end="url(#a)"/>
  <line x1="330" y1="160" x2="360" y2="160" stroke="#333" marker-end="url(#a)"/>
  <line x1="330" y1="255" x2="360" y2="255" stroke="#333" marker-end="url(#a)"/>
  <line x1="500" y1="160" x2="530" y2="160" stroke="#333" marker-end="url(#a)"/>
  <text x="430" y="25" text-anchor="middle" font-size="12">CLIP: img ↔ txt</text>
  <text x="430" y="305" text-anchor="middle" font-size="12">CLIP: aud ↔ img, aud ↔ txt</text>
</svg>
```

## 5. Batching
**ClassBalancedBatchSampler**: each batch selects up to `batch_size` distinct classes, one sample per class (image, text, audio). This enforces co-occurrence of positive pairs for all modalities, stabilizing contrastive training, especially when audio presence is sparse or uneven.

## 6. Training
- Input: 224×224.
- Optimizer: AdamW; Cosine LR; weight decay 0.01.
- AMP: torch.amp autocast + GradScaler.
- Caps: typically 5–8 samples per class; batches use the balanced sampler.
- Epochs: 60+; when LR ~1e-6 and loss plateaus, restart with fresh schedule instead of extending at a vanishing LR.
- Logging: per-step CE/CLIP/Audio/Audio-Text losses and logit_scale to diagnose imbalance.

## 7. Evaluation
### 7.1 Retrieval
- Metrics: R@1/R@5 for Text→Image, Audio→Image; Image→Image as sanity.
- Visualization: retrieval grid with query (text/audio/image), predicted nearest image; color-coded correctness. R values displayed in title.

### 7.2 t-SNE
- Latent embeddings from all modalities; markers by modality, colors by class.
- Subset option (`eval_num_classes`, default 10) to sample classes for clearer plots; only those classes appear in legend/points.

### 7.3 Subset Evaluation
- Random subset with fixed seed for reproducibility; applies to metrics and plots.

## 8. Implementation Artifacts
- `collect_data.py`: CIFAR/web images; OpenAI TTS with leading silence; saves to `data/image`, `data/audio`.
- `add_openai_tts_coral.py`: extra coral TTS with padding; overwrite allowed.
- `generate_eval_tts_openai.py`: eval TTS (distinct voices/prompts) with padding; overwrite allowed; no argparse (config in code).
- `precompute_audio_images.py`: trim → fixed-duration warp → Whisper/log-mel → PNG caches for train/eval; overwrite skipped unless PNG deleted.
- `fix_silent_tts.py`: detect quiet wavs (peak<thr), re-synthesize; does not touch PNGs.
- `fix_quiet_and_regen.py`: detect quiet wavs, re-synthesize, delete related PNGs, regenerate PNGs (method1/2).
- `list_quiet_tts.py`: report peak/duration and PNG existence.

## 9. Deployment Considerations
- ConvNeXt-Tiny FP32 ~50–60 MB; FP16/INT8 ~10–30 MB. Suitable for mobile/robot NPU with ONNX→TFLite/CoreML and NNAPI/NPU delegates.
- Precompute audio-images offline; on-device only runs the vision backbone.
- For higher accuracy: train larger, then distill/compress to a small backbone (MobileNet/EfficientNet-Lite/ConvNeXt-tiny).
- On-device preprocessing: prefer log-mel backend; Whisper encoder is heavier and best kept offline.

## 10. Failure Modes & Mitigations
- Quiet/empty audio → re-synthesize and regenerate PNGs via provided scripts.
- Text accuracy drop when audio improves → downweight audio/audio-text losses (e.g., 0.5) or warm-start them after initial epochs.
- Modal scale mismatch → enforce identical normalization across modalities; avoid mixed pipelines.
- Audio imbalance → balanced sampler ensures audio-present classes appear per batch; increase per-class audio cap if possible.

## 11. Pseudocode
```
for each class c:
    imgs_c = sample CIFAR images (cap)
    txt_c  = render_texture(hash(c))
    aud_c  = TTS(c); trim; warp to 2s; to_image(method1/2)

train:
    batch = class_balanced_sampler()
    lat_i, log_i = f(img)
    lat_t, log_t = f(txt)
    if has_audio: lat_a = f(aud)
    loss = CE(img)+CE(txt) + CLIP(img,txt) + CLIP(aud,img) + CLIP(aud,txt)
    optimize(...)

eval:
    compute R@K; visualize grid; t-SNE on subset classes
```

## 12. Related Work
- ConvNeXt: Liu et al., “A ConvNet for the 2020s,” CVPR 2022.
- CLIP: Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML 2021.
- Texture bias: Geirhos et al., “ImageNet-trained CNNs are biased towards texture,” ICLR 2019.
- Audio spectrograms as images: Hershey et al., “CNN Architectures for Large-Scale Audio Classification,” ICASSP 2017.
- Whisper: Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” 2022.

## 13. Conclusion
Visual Byte shows that converting text and audio into texture-like images enables a single vision backbone to align three modalities without heavy language/audio models. Class-balanced batching, consistent normalization, and precomputed audio-images stabilize training. For edge deployment, a quantized ConvNeXt-tiny with offline audio preprocessing is practical; for higher accuracy, train larger and distill/compress to an edge-friendly backbone.
