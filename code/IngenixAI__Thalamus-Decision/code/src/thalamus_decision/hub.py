"""Locate a model: a local copy (an explicit folder or ``THALAMUS_DECISION_HOME``) or a Hugging Face repository.

A model repository holds the checkpoint, ``CALIBRATION.json``, ``decision.json``, and the same ``preprocessing/``
(gene panel, normalization) and ``depmap/`` (1,673 cell-line profiles) files as Thalamus-Reasoning-9B.
"""
import os
from pathlib import Path

REPO_ID = "ingenix/Thalamus-Decision-9B"
FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "*.safetensors", "model.safetensors.index.json", "CALIBRATION.json", "decision.json",
         "preprocessing/*", "depmap/*"]
EVAL_FILES = ["eval/*"]  # the evaluation subsets (about 0.5 GB): fetched only by eval/benchmark.py


def model_path(source: str | os.PathLike | None = None) -> Path:
    """``source`` (or ``$THALAMUS_DECISION_HOME``) if it is a local folder, else the Hub snapshot of ``source`` (a repo
    id ``owner/name``) or of ``REPO_ID``, limited to the files the engine reads."""
    if not source and os.environ.get("THALAMUS_DECISION_HOME"):
        source = os.environ["THALAMUS_DECISION_HOME"]
        if not Path(source).expanduser().is_dir():
            raise FileNotFoundError(f"THALAMUS_DECISION_HOME={source} is not a folder")
    if source:
        local = Path(source).expanduser()
        if local.is_dir():
            return local
        if local.exists() or str(source).startswith((".", "/", "~")) or str(source).count("/") != 1:
            raise FileNotFoundError(f"{source} is not a folder; give a local copy of the model repository or a Hugging "
                                    "Face repo id (owner/name)")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(source or REPO_ID), allow_patterns=FILES))


def eval_path(source: str | os.PathLike | None = None) -> Path:
    """The ``eval/`` folder of a model: in a local copy (``source`` or ``THALAMUS_DECISION_HOME``), else fetched from
    the Hub on first use; the server's snapshot (``FILES``) leaves it out."""
    local = source or os.environ.get("THALAMUS_DECISION_HOME")
    if local and Path(local).expanduser().is_dir():
        folder = Path(local).expanduser()
        if not (folder / "eval").is_dir():
            raise FileNotFoundError(f"{folder} has no eval/ folder; fetch it with "
                                    f"`hf download {REPO_ID} --include 'eval/*' --local-dir {folder}`")
        return folder / "eval"
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(source or REPO_ID), allow_patterns=EVAL_FILES)) / "eval"
