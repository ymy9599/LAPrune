def get_modifier(method: str, model_type: str = None):
    """
    LAPrune open-source scope: **AQ only**.

    Supported:
    - aq-evaluation
    - aq-generation
    """
    if method == "aq-evaluation":
        from .aq_evaluation import AQEvaluation
        return None, AQEvaluation

    if method == "aq-generation":
        from .aq_generation import AQGeneration
        return None, AQGeneration

    raise NotImplementedError(
        f"Unsupported model_method='{method}'. This LAPrune package intentionally exposes AQ only."
    )