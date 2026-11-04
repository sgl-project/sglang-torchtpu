"""Out-of-tree class substitution.

Core constructs some classes directly (``Fp8LinearMethod(config)`` inside
several quantization configs) where no factory lets a platform pick the class.
A class that mixes in :class:`OotReplaceable` lets a platform plugin register a
subclass once; every later construction of exactly that class then builds the
subclass. vLLM's ``CustomOp.register_oot`` has the same shape.

Lookup is by exact type: subclasses of a registered base, and the replacement
itself, construct unchanged. Pickling and deepcopy rebuild through
``type(obj).__new__`` and so keep the object's concrete class.
"""

from __future__ import annotations

__all__ = [
    "OotReplaceable",
    "clear_oot_replacements",
    "oot_base_class",
    "register_oot_replacement",
]

# base class -> the subclass constructed in its place, and the reverse.
_OOT_REPLACEMENTS: dict[type, type] = {}
_OOT_BASES: dict[type, type] = {}


class OotReplaceable:
    """Lets a platform plugin substitute a subclass wherever core builds this class."""

    def __new__(cls, *args, **kwargs):
        # Python then runs the substitute's __init__, as it is an instance of cls.
        return super().__new__(_OOT_REPLACEMENTS.get(cls, cls))


def register_oot_replacement(base_cls: type, oot_cls: type) -> None:
    """Construct ``oot_cls`` wherever core constructs exactly ``base_cls``.

    Registering the same pair again is a no-op; any other conflict raises.
    """
    if not issubclass(base_cls, OotReplaceable):
        raise TypeError(f"{base_cls.__name__} does not mix in OotReplaceable")
    if oot_cls is base_cls or not issubclass(oot_cls, base_cls):
        raise TypeError(
            f"{oot_cls.__name__} must be a proper subclass of {base_cls.__name__}"
        )
    existing = _OOT_REPLACEMENTS.get(base_cls)
    if existing is oot_cls:
        return
    if existing is not None:
        raise ValueError(
            f"{base_cls.__name__} is already replaced by {existing.__name__}"
        )
    if oot_cls in _OOT_BASES:
        raise ValueError(
            f"{oot_cls.__name__} already replaces {_OOT_BASES[oot_cls].__name__}"
        )
    _OOT_REPLACEMENTS[base_cls] = oot_cls
    _OOT_BASES[oot_cls] = base_cls


def oot_base_class(cls: type) -> type:
    """The core class ``cls`` stands in for: its registered base, or ``cls`` itself.

    Core code that dispatches on an exact class should compare this, so a
    substituted object takes the same branch as the class core constructed.
    """
    return _OOT_BASES.get(cls, cls)


def clear_oot_replacements() -> None:
    """Drop every registration; for tests."""
    _OOT_REPLACEMENTS.clear()
    _OOT_BASES.clear()
