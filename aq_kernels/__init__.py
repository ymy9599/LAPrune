try:
    from .aq_lut import aq_lut_forward, aq_lut_gather_forward
except Exception:
    aq_lut_forward = None
    aq_lut_gather_forward = None

try:
    from .fused_linear_encode import fused_linear_encode_forward
except Exception:
    fused_linear_encode_forward = None

try:
    from .aq_lut_gqa import (
        aq_lut_gather as aq_lut_gather_gqa,
        aq_lut_gather_grouped,
        gqa_gather_kv,
        gqa_gather_kv_with_tail,
    )
except Exception:
    aq_lut_gather_gqa = None
    aq_lut_gather_grouped = None
    gqa_gather_kv = None
    gqa_gather_kv_with_tail = None
