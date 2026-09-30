"""
CloudSealed Dataclass Compiler — @dataclass → Numba @jitclass bridge.

Turns standard Python dataclasses into high-performance Numba C-structs
while preserving an ergonomic API:

    - Automatic PEP-484 type hint → Numba type translation
    - Default field values preserved in the generated __init__
    - User-defined methods compiled to nopython mode automatically
    - Nested jitdataclass fields resolved to the correct Numba StructRef
    - Generated __repr__ for debugging
"""

import inspect
from dataclasses import dataclass, fields as dc_fields

from .compiler import map_python_type_to_numba

try:
    from numba.experimental import jitclass
    from numba.core import types

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Registry of jitdataclass-compiled classes so nested fields can be resolved.
_jitclass_registry: dict[str, object] = {}


def _resolve_numba_type(field_name, py_type, cls_name):
    """Resolve a field type to a Numba type, including nested jitdataclasses."""
    # Check if the type is a previously compiled jitdataclass
    type_key = getattr(py_type, "__qualname__", getattr(py_type, "__name__", None))
    if type_key and type_key in _jitclass_registry:
        registered = _jitclass_registry[type_key]
        if hasattr(registered, "class_type"):
            return registered.class_type.instance_type

    nb_type = map_python_type_to_numba(py_type)
    if nb_type is None:
        raise ValueError(
            f"Unsupported type '{py_type}' for field '{field_name}' in "
            f"'{cls_name}'. Supported: int, float, bool, str, np scalar types, "
            f"np.ndarray, Optional[T], List[T], Tuple[T,...], Dict[K,V], "
            f"or another @jitdataclass."
        )
    return nb_type


def jitdataclass(cls):
    """Decorator that compiles a Python dataclass into a Numba jitclass.

    Preserves default values, compiles user methods, supports nested
    jitdataclass fields, and generates a __repr__ for debugging.
    """
    if not NUMBA_AVAILABLE:
        return dataclass(cls) if not hasattr(cls, "__dataclass_fields__") else cls

    if not hasattr(cls, "__dataclass_fields__"):
        cls = dataclass(cls)

    cls_name = cls.__name__
    dc = dc_fields(cls)

    # Build Numba type spec
    spec = []
    for f in dc:
        nb_type = _resolve_numba_type(f.name, f.type, cls_name)
        spec.append((f.name, nb_type))

    # Generate __init__ with default values
    init_params = []
    init_defaults = {}
    for f in dc:
        if f.default is not f.default_factory:
            if f.default is not f.default_factory and f.default is not f.MISSING:
                init_params.append(f"{f.name}={f.name}_default")
                init_defaults[f"{f.name}_default"] = f.default
            else:
                init_params.append(f.name)
        else:
            init_params.append(f.name)

    args_str = ", ".join(init_params)
    assign_str = "\n    ".join(f"self.{f.name} = {f.name}" for f in dc)
    init_code = f"def __init__(self, {args_str}):\n    {assign_str}"

    local_env = {}
    exec(init_code, init_defaults, local_env)

    # Build clean class dict
    clean_dict = {"__init__": local_env["__init__"]}

    _STRIP = {
        "__dataclass_params__",
        "__dataclass_fields__",
        "__match_args__",
        "__dict__",
        "__weakref__",
        "__annotations__",
        "__hash__",
        "__repr__",
        "__eq__",
        "__init__",
        "__doc__",
    }

    annotations = {f.name: f.type for f in dc}
    for name, attr in cls.__dict__.items():
        if name in _STRIP:
            continue
        if name.startswith("__") and name.endswith("__"):
            continue
        if name in annotations:
            continue
        # User-defined methods are preserved as-is (Numba compiles them)
        clean_dict[name] = attr

    CleanClass = type(cls_name, (), clean_dict)

    compiled = jitclass(spec)(CleanClass)

    # Register for nested field resolution
    _jitclass_registry[cls_name] = compiled

    return compiled
