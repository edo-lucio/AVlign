from .datasets import (
    ClothoDataset,
    Flickr8kDataset,
    download_clotho,
    download_flickr8k,
)
from .models import (
    ASTEncoder,
    CLAPEncoder,
    CLIPEncoder,
    DINOv2Encoder,
    MODEL_IDS,
    RoBERTaEncoder,
    T5Encoder,
    build_encoder,
)

__all__ = [
    # datasets
    "Flickr8kDataset",
    "ClothoDataset",
    "download_flickr8k",
    "download_clotho",
    # models
    "CLIPEncoder",
    "DINOv2Encoder",
    "CLAPEncoder",
    "ASTEncoder",
    "RoBERTaEncoder",
    "T5Encoder",
    "build_encoder",
    "MODEL_IDS",
]
# cka is imported as a submodule (`from fgw_validation.cka import ...`)
# rather than re-exported here, so `python -m fgw_validation.cka` doesn't
# trigger the runpy double-import warning.
