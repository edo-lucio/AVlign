from pathlib import Path

import torch
import torchaudio
from PIL import Image
from transformers import (
    ASTFeatureExtractor,
    ASTModel,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    ClapModel,
    ClapProcessor,
    CLIPModel,
    CLIPProcessor,
    T5EncoderModel,
)


# ─── Recommended HF model IDs ────────────────────────────────────────────────

MODEL_IDS: dict[str, dict[str, str]] = {
    "clip":    {"small":  "openai/clip-vit-base-patch32",
                "medium": "openai/clip-vit-large-patch14"},
    "dinov2":  {"small":  "facebook/dinov2-small",
                "medium": "facebook/dinov2-base"},
    "clap":    {"medium": "laion/clap-htsat-unfused"},
    "ast":     {"medium": "MIT/ast-finetuned-audioset-10-10-0.4593"},
    "roberta": {"small":  "roberta-base",
                "medium": "roberta-large"},
    "t5":      {"small":  "google-t5/t5-small",
                "medium": "google-t5/t5-base"},
}


# ─── Input loaders ───────────────────────────────────────────────────────────

def _default_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_image(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    raise TypeError(f"unsupported image input: {type(x).__name__}")


def _load_audio(x, target_sr: int) -> torch.Tensor:
    """Return a mono float32 waveform resampled to target_sr."""
    if isinstance(x, torch.Tensor):
        wav, sr = x, target_sr   # caller is responsible for sample rate here
    elif isinstance(x, (str, Path)):
        wav, sr = torchaudio.load(str(x))           # (C, T)
    else:
        raise TypeError(f"unsupported audio input: {type(x).__name__}")
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(torch.float32)


def _features(out) -> torch.Tensor:
    """Unwrap CLIP/CLAP `get_*_features` outputs.

    transformers <5 returned a Tensor; transformers >=5 returns a
    BaseModelOutputWithPooling whose `.pooler_output` is the projected feature.
    """
    if isinstance(out, torch.Tensor):
        return out
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    raise TypeError(f"unexpected feature output type: {type(out).__name__}")


def _resolve_id(family: str, size: str | None, model_id: str | None) -> str:
    if model_id is not None:
        return model_id
    options = MODEL_IDS[family]
    if size is None:
        # pick the only entry, or the smallest if more than one
        return options.get("small") or next(iter(options.values()))
    if size not in options:
        raise KeyError(f"{family}: size must be one of {list(options)}, got {size!r}")
    return options[size]


# ─── Image + text dual towers ────────────────────────────────────────────────

class CLIPEncoder:
    """OpenAI CLIP — image and text towers in a shared projection space."""

    def __init__(self, size: str = "small", device: str | None = None,
                 model_id: str | None = None):
        self.device = _default_device(device)
        mid = _resolve_id("clip", size, model_id)
        self.model = CLIPModel.from_pretrained(mid).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(mid)
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode_image(self, images) -> torch.Tensor:
        imgs = [_load_image(x) for x in images]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        return _features(self.model.get_image_features(**inputs)).float().cpu()

    @torch.no_grad()
    def encode_text(self, texts) -> torch.Tensor:
        inputs = self.processor(text=list(texts), return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        return _features(self.model.get_text_features(**inputs)).float().cpu()


# ─── Image only ──────────────────────────────────────────────────────────────

class DINOv2Encoder:
    """DINOv2 — self-supervised ViT; image embedding from the [CLS] token."""

    def __init__(self, size: str = "small", device: str | None = None,
                 model_id: str | None = None):
        self.device = _default_device(device)
        mid = _resolve_id("dinov2", size, model_id)
        self.model = AutoModel.from_pretrained(mid).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(mid)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_image(self, images) -> torch.Tensor:
        imgs = [_load_image(x) for x in images]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        return out.last_hidden_state[:, 0].float().cpu()


# ─── Audio + text dual towers ────────────────────────────────────────────────

class CLAPEncoder:
    """LAION-CLAP — audio and text towers in a shared projection space."""

    SAMPLE_RATE = 48000

    def __init__(self, size: str = "medium", device: str | None = None,
                 model_id: str | None = None):
        self.device = _default_device(device)
        mid = _resolve_id("clap", size, model_id)
        self.model = ClapModel.from_pretrained(mid).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(mid)
        self.dim = self.model.config.projection_dim

    @torch.no_grad()
    def encode_audio(self, audios) -> torch.Tensor:
        wavs = [_load_audio(x, self.SAMPLE_RATE).numpy() for x in audios]
        inputs = self.processor(audio=wavs, sampling_rate=self.SAMPLE_RATE,
                                return_tensors="pt").to(self.device)
        return _features(self.model.get_audio_features(**inputs)).float().cpu()

    @torch.no_grad()
    def encode_text(self, texts) -> torch.Tensor:
        inputs = self.processor(text=list(texts), return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        return _features(self.model.get_text_features(**inputs)).float().cpu()


# ─── Audio only ──────────────────────────────────────────────────────────────

class ASTEncoder:
    """Audio Spectrogram Transformer — supervised AudioSet, ViT-style backbone."""

    SAMPLE_RATE = 16000

    def __init__(self, size: str = "medium", device: str | None = None,
                 model_id: str | None = None):
        self.device = _default_device(device)
        mid = _resolve_id("ast", size, model_id)
        self.model = ASTModel.from_pretrained(mid).to(self.device).eval()
        self.processor = ASTFeatureExtractor.from_pretrained(mid)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_audio(self, audios) -> torch.Tensor:
        wavs = [_load_audio(x, self.SAMPLE_RATE).numpy() for x in audios]
        inputs = self.processor(wavs, sampling_rate=self.SAMPLE_RATE,
                                return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        return out.pooler_output.float().cpu()


# ─── Text only ───────────────────────────────────────────────────────────────

def _mean_pool(hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class RoBERTaEncoder:
    """RoBERTa — sentence embedding via attention-masked mean pool."""

    def __init__(self, size: str = "small", device: str | None = None,
                 model_id: str | None = None):
        self.device = _default_device(device)
        mid = _resolve_id("roberta", size, model_id)
        self.model = AutoModel.from_pretrained(mid).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(mid)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode_text(self, texts) -> torch.Tensor:
        inputs = self.tokenizer(list(texts), return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        out = self.model(**inputs)
        return _mean_pool(out.last_hidden_state, inputs["attention_mask"]).float().cpu()


class T5Encoder:
    """T5 encoder-only — sentence embedding via attention-masked mean pool."""

    def __init__(self, size: str = "small", device: str | None = None,
                 model_id: str | None = None):
        self.device = _default_device(device)
        mid = _resolve_id("t5", size, model_id)
        self.model = T5EncoderModel.from_pretrained(mid).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(mid)
        self.dim = self.model.config.d_model

    @torch.no_grad()
    def encode_text(self, texts) -> torch.Tensor:
        inputs = self.tokenizer(list(texts), return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        out = self.model(**inputs)
        return _mean_pool(out.last_hidden_state, inputs["attention_mask"]).float().cpu()


# ─── Factory ─────────────────────────────────────────────────────────────────

ENCODERS = {
    "clip":    CLIPEncoder,
    "dinov2":  DINOv2Encoder,
    "clap":    CLAPEncoder,
    "ast":     ASTEncoder,
    "roberta": RoBERTaEncoder,
    "t5":      T5Encoder,
}


def build_encoder(name: str, **kwargs):
    """`build_encoder("clip", size="small", device="cuda")` etc."""
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder {name!r}, choose from {sorted(ENCODERS)}")
    return ENCODERS[name](**kwargs)
