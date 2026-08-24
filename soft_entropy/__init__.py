from soft_entropy.accumulator import SoftEntropyAccumulator

__all__ = ["SoftEntropyAccumulator", "LLMInferrer"]


def __getattr__(name: str):
    if name == "LLMInferrer":
        from soft_entropy.llm import LLMInferrer

        return LLMInferrer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
