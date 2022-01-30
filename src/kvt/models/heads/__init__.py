from .dino import DINOHead
from .head import MultiHead
from .metric_learning import (
    AdaCos,
    AddMarginProduct,
    ArcMarginProduct,
    CurricularFace,
    SphereProduct,
)
from .multisample_dropout import MultiSampleDropout

__all__ = [
    "DINOHead",
    "MultiHead",
    "AdaCos",
    "AddMarginProduct",
    "ArcMarginProduct",
    "CurricularFace",
    "SphereProduct",
    "MultiSampleDropout",
]
