"""Thalamus-Decision: calibrated typed decisions (choice / yes-no / score) about text and expression profiles."""

__version__ = "0.1.0"


def __getattr__(name):  # the model stack (torch, transformers) loads only when it is used
    if name == "ThalamusDecision":
        from .api import ThalamusDecision
        return ThalamusDecision
    raise AttributeError(name)


__all__ = ["ThalamusDecision", "__version__"]
