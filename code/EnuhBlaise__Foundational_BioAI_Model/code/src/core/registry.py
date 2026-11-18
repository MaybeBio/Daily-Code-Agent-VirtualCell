"""Component registry.

Every extensible piece of this repository -- stems, backbones, heads,
tokenizers, data sources, splitters, metrics, baselines -- is registered
under a string name and instantiated from a plain config mapping.

Design note: we deliberately do *not* use ``hydra.utils.instantiate`` with
``_target_`` import paths.  Hydra composes configs and parses the CLI; object
construction goes through this registry.  The reason is reviewability: a
config says ``name: binned_value`` rather than a dotted import path, so the
set of legal values is enumerable (``REGISTRY.keys()``), typo'd names fail
loudly with suggestions, and nothing in a YAML file can import arbitrary code.

Adding a component never requires editing a training loop.
"""
from __future__ import annotations

import difflib
import importlib
import inspect
import pkgutil
from typing import Any, Callable, Dict, Generic, Iterator, Mapping, Type, TypeVar

T = TypeVar("T")

__all__ = [
    "Registry",
    "RegistryError",
    "STEMS",
    "BACKBONES",
    "HEADS",
    "TOKENIZERS",
    "SOURCES",
    "SPLITTERS",
    "BASELINES",
    "METRICS",
    "register_stem",
    "register_backbone",
    "register_head",
    "register_tokenizer",
    "register_source",
    "register_splitter",
    "register_baseline",
    "register_metric",
    "build",
    "autodiscover",
]


class RegistryError(KeyError):
    """Raised when a name is missing from a registry or double-registered."""


def _snake(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


class Registry(Generic[T]):
    """A named collection of components with config-driven construction."""

    def __init__(self, kind: str, base: Type[T] | None = None) -> None:
        self.kind = kind
        self.base = base
        self._entries: Dict[str, Type[T]] = {}
        self._provenance: Dict[str, str] = {}

    # -- registration ----------------------------------------------------
    def register(
        self, name: str | None = None, *, aliases: tuple[str, ...] = ()
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator: ``@STEMS.register("expression")``."""

        def deco(obj: Type[T]) -> Type[T]:
            key = name or _snake(obj.__name__)
            self._add(key, obj)
            for alias in aliases:
                self._add(alias, obj)
            return obj

        return deco

    def _add(self, key: str, obj: Type[T]) -> None:
        if key in self._entries and self._entries[key] is not obj:
            raise RegistryError(
                f"{self.kind} '{key}' already registered by "
                f"{self._provenance.get(key, '?')}; refusing to shadow it."
            )
        if self.base is not None and inspect.isclass(obj):
            if not issubclass(obj, self.base):
                raise RegistryError(
                    f"{obj.__name__} must subclass {self.base.__name__} "
                    f"to register as a {self.kind}."
                )
        self._entries[key] = obj
        mod = getattr(obj, "__module__", "?")
        self._provenance[key] = f"{mod}.{getattr(obj, '__name__', obj)}"

    # -- lookup ----------------------------------------------------------
    def get(self, key: str) -> Type[T]:
        try:
            return self._entries[key]
        except KeyError:
            close = difflib.get_close_matches(key, self._entries, n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise RegistryError(
                f"unknown {self.kind} '{key}'. "
                f"Registered: {sorted(self._entries) or '<none>'}.{hint}"
            ) from None

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def provenance(self) -> Dict[str, str]:
        return dict(self._provenance)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<Registry {self.kind}: {len(self)} entries>"

    # -- construction ----------------------------------------------------
    def build(self, cfg: Mapping[str, Any] | str | None = None, **overrides: Any) -> T:
        """Instantiate from ``{'name': ..., **kwargs}`` or a bare name string.

        Keys beginning with ``_`` are treated as annotations and dropped, so a
        config may carry documentation (``_note:``) without reaching __init__.
        """
        if cfg is None:
            cfg = {}
        if isinstance(cfg, str):
            cfg = {"name": cfg}
        cfg = _to_plain(cfg)
        if "name" not in cfg:
            raise RegistryError(
                f"{self.kind} config needs a 'name' key; got keys {sorted(cfg)}"
            )
        kwargs = {k: v for k, v in cfg.items() if k != "name" and not k.startswith("_")}
        kwargs.update(overrides)
        cls = self.get(cfg["name"])
        try:
            return cls(**kwargs)  # type: ignore[call-arg]
        except TypeError as exc:
            sig = inspect.signature(cls)
            raise RegistryError(
                f"cannot build {self.kind} '{cfg['name']}' ({cls.__name__}) "
                f"with keys {sorted(kwargs)}; signature is {sig}. Original: {exc}"
            ) from exc


def _to_plain(cfg: Any) -> dict:
    """Accept dict / OmegaConf DictConfig / dataclass-ish mappings."""
    try:  # avoid a hard omegaconf import at module load
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - omegaconf optional at runtime
        pass
    return dict(cfg)


# --- the registries -----------------------------------------------------
# Base classes are attached in core.base to avoid a circular import; these
# start untyped and base classes are bound there via ``bind_base``.
STEMS: Registry = Registry("stem")
BACKBONES: Registry = Registry("backbone")
HEADS: Registry = Registry("head")
TOKENIZERS: Registry = Registry("tokenizer")
SOURCES: Registry = Registry("data source")
SPLITTERS: Registry = Registry("splitter")
BASELINES: Registry = Registry("baseline")
METRICS: Registry = Registry("metric")

ALL_REGISTRIES: Dict[str, Registry] = {
    "stem": STEMS,
    "backbone": BACKBONES,
    "head": HEADS,
    "tokenizer": TOKENIZERS,
    "source": SOURCES,
    "splitter": SPLITTERS,
    "baseline": BASELINES,
    "metric": METRICS,
}

register_stem = STEMS.register
register_backbone = BACKBONES.register
register_head = HEADS.register
register_tokenizer = TOKENIZERS.register
register_source = SOURCES.register
register_splitter = SPLITTERS.register
register_baseline = BASELINES.register
register_metric = METRICS.register


def build(kind: str, cfg: Mapping[str, Any] | str, **overrides: Any) -> Any:
    """Generic entry point: ``build('backbone', cfg.model.backbone)``."""
    if kind not in ALL_REGISTRIES:
        raise RegistryError(
            f"no registry '{kind}'; have {sorted(ALL_REGISTRIES)}"
        )
    return ALL_REGISTRIES[kind].build(cfg, **overrides)


_DISCOVERED = False


def autodiscover(package: str = "src") -> None:
    """Import every submodule so decorators run.

    Called once by the entry point.  Import errors in optional backends (e.g.
    a missing ``tiledbsoma``) are swallowed with a note rather than killing a
    run that does not need them.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    skipped: list[str] = []
    for sub in ("core", "data", "models", "eval"):
        try:
            pkg = importlib.import_module(f"{package}.{sub}")
        except Exception as exc:  # pragma: no cover
            skipped.append(f"{package}.{sub}: {exc}")
            continue
        for mod in pkgutil.walk_packages(pkg.__path__, f"{pkg.__name__}."):
            try:
                importlib.import_module(mod.name)
            except Exception as exc:
                skipped.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    if skipped:
        import logging

        logging.getLogger(__name__).debug(
            "registry autodiscover skipped %d module(s): %s",
            len(skipped),
            "; ".join(skipped),
        )
