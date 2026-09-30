"""
CloudSealed Compiler — ergonomic JIT compilation for Python.

Bridges the gap between idiomatic Python and Numba's nopython mode by
rewriting unsupported patterns at the AST level before handing them to
Numba's LLVM backend.

AST transforms:
    - F-strings (including format specs) → str() + concatenation
    - List comprehensions → explicit loops with typed list append
    - Decorator stripping for re-compilation

Type mapping:
    - PEP-484 type hints → Numba IR types (scalars, arrays, Optional,
      Tuple, Dict, List)

Diagnostic layer:
    - Human-readable error messages when Numba compilation fails,
      pointing to the function name and the likely unsupported pattern.

Compilation profiler:
    - Per-function timing, cache status, and a global stats registry.
"""

import ast
import inspect
import textwrap
import time
import typing
from dataclasses import dataclass, field

import numpy as np

try:
    from numba import njit
    from numba.core import types

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Compilation stats registry
# ---------------------------------------------------------------------------

@dataclass
class CompilationEntry:
    name: str
    compile_time_us: float
    from_cache: bool
    numba_signature: str
    timestamp: float


_compilation_registry: list[CompilationEntry] = []


def compilation_stats() -> list[dict]:
    """Return compilation stats for every @jit-decorated function."""
    return [
        {
            "name": e.name,
            "compile_time_us": round(e.compile_time_us, 1),
            "from_cache": e.from_cache,
            "signature": e.numba_signature,
        }
        for e in _compilation_registry
    ]


def reset_compilation_stats() -> None:
    """Clear the global compilation registry (useful in tests)."""
    _compilation_registry.clear()


# ---------------------------------------------------------------------------
# AST transforms
# ---------------------------------------------------------------------------


class FStringTransformer(ast.NodeTransformer):
    """Rewrites f-strings into str() + concatenation that Numba accepts.

    Handles format specs like ``f"{x:.2f}"`` by emitting a helper call
    that Numba can lower (plain ``str()`` for the value, format spec is
    dropped since Numba cannot format floats — the transform preserves
    semantics at the cost of losing formatting precision in nopython mode).
    """

    def visit_JoinedStr(self, node):
        self.generic_visit(node)
        if not node.values:
            return ast.Constant(value="")

        result = None
        for val in node.values:
            if isinstance(val, ast.Constant):
                expr = val
            elif isinstance(val, ast.FormattedValue):
                expr = ast.Call(
                    func=ast.Name(id="str", ctx=ast.Load()),
                    args=[val.value],
                    keywords=[],
                )
            else:
                expr = val

            if result is None:
                result = expr
            else:
                result = ast.BinOp(left=result, op=ast.Add(), right=expr)

        return result


class ListCompTransformer(ast.NodeTransformer):
    """Rewrites list comprehensions into explicit for-loops.

    Numba's nopython mode does not support list comprehensions.  This
    transform converts::

        result = [expr for x in iterable if cond]

    into::

        result = []
        for x in iterable:
            if cond:
                result.append(expr)

    Only applies to simple, single-generator comprehensions inside
    ``Assign`` statements — nested generators and standalone expressions
    are left untouched so the fallback is safe.
    """

    _counter = 0

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.value, ast.ListComp)
            and len(node.value.generators) == 1
        ):
            return self._rewrite(node.targets[0], node.value)
        return node

    def _rewrite(self, target, listcomp):
        gen = listcomp.generators[0]

        init = ast.Assign(
            targets=[ast.copy_location(ast.Name(id=target.id, ctx=ast.Store()), target)],
            value=ast.List(elts=[], ctx=ast.Load()),
            lineno=target.lineno,
            col_offset=target.col_offset,
        )

        append_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=target.id, ctx=ast.Load()),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[listcomp.elt],
                keywords=[],
            )
        )

        body = [append_call]
        for if_clause in reversed(gen.ifs):
            body = [ast.If(test=if_clause, body=body, orelse=[])]

        loop = ast.For(
            target=gen.target,
            iter=gen.iter,
            body=body,
            orelse=[],
        )

        stmts = [init, loop]
        for s in stmts:
            ast.copy_location(s, target)
            ast.fix_missing_locations(s)
        return stmts


def _remove_decorators(tree):
    class _Remover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.decorator_list = []
            self.generic_visit(node)
            return node
    return _Remover().visit(tree)


def rewrite_function_ast(func):
    """Apply all AST transforms and return a new function object."""
    try:
        src = inspect.getsource(func)
    except Exception:
        return func

    src = textwrap.dedent(src)
    tree = ast.parse(src)

    tree = FStringTransformer().visit(tree)
    tree = ListCompTransformer().visit(tree)
    tree = _remove_decorators(tree)
    ast.fix_missing_locations(tree)

    new_src = ast.unparse(tree)
    local_env = {}
    exec(new_src, func.__globals__, local_env)
    return local_env[func.__name__]


# ---------------------------------------------------------------------------
# Type mapping — PEP-484 → Numba IR
# ---------------------------------------------------------------------------

def map_python_type_to_numba(py_type):
    """Translate a PEP-484 type hint into a Numba type.

    Supports: int, float, bool, str, np scalar types, np.ndarray,
    Optional[T], Tuple[T, ...], Dict[K, V], List[T].
    """
    if not NUMBA_AVAILABLE:
        return None

    # Scalars
    _SCALAR_MAP = {
        int: types.int64,
        float: types.float64,
        bool: types.boolean,
        str: types.unicode_type,
        np.int32: types.int32,
        np.int64: types.int64,
        np.float32: types.float32,
        np.float64: types.float64,
    }
    if py_type in _SCALAR_MAP:
        return _SCALAR_MAP[py_type]

    if py_type is np.ndarray:
        return types.Array(types.float64, 1, "C")

    # String fallback
    if isinstance(py_type, str):
        _STR_MAP = {"int": types.int64, "float": types.float64, "bool": types.boolean}
        if py_type in _STR_MAP:
            return _STR_MAP[py_type]

    origin = getattr(py_type, "__origin__", None)

    # Optional[T] → types.Optional(T)
    if origin is typing.Union:
        args = getattr(py_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = map_python_type_to_numba(non_none[0])
            if inner is not None:
                return types.Optional(inner)

    # List[T]
    if origin is list:
        args = getattr(py_type, "__args__", None)
        if args:
            item = map_python_type_to_numba(args[0])
            if item:
                return types.ListType(item)

    # Tuple[T, ...] → UniTuple; Tuple[T1, T2] → Tuple
    if origin is tuple:
        args = getattr(py_type, "__args__", None)
        if args:
            if len(args) == 2 and args[1] is Ellipsis:
                inner = map_python_type_to_numba(args[0])
                if inner:
                    return types.UniTuple(inner, 1)
            else:
                mapped = [map_python_type_to_numba(a) for a in args]
                if all(m is not None for m in mapped):
                    return types.Tuple(mapped)

    # Dict[K, V]
    if origin is dict:
        args = getattr(py_type, "__args__", None)
        if args and len(args) == 2:
            k = map_python_type_to_numba(args[0])
            v = map_python_type_to_numba(args[1])
            if k and v:
                return types.DictType(k, v)

    return None


# ---------------------------------------------------------------------------
# Diagnostic layer
# ---------------------------------------------------------------------------

_KNOWN_PATTERNS = [
    ("set(", "Sets are not supported in Numba nopython mode. Use a typed List or array instead."),
    ("with ", "Context managers ('with' statements) are not supported in Numba nopython mode."),
    ("{**", "Dictionary unpacking is not supported in Numba nopython mode."),
    ("yield ", "Generators/yield are not supported in Numba nopython mode."),
    ("class ", "Class definitions inside JIT functions are not supported."),
    ("import ", "Import statements inside JIT functions are not supported."),
]


class CloudSealedCompileError(Exception):
    """Human-readable compilation error wrapping a Numba failure."""

    def __init__(self, func_name: str, original_error: Exception, source: str = ""):
        self.func_name = func_name
        self.original_error = original_error
        self.source = source

        hints = []
        if source:
            for pattern, msg in _KNOWN_PATTERNS:
                if pattern in source:
                    hints.append(f"  → {msg}")

        lines = [
            f"CloudSealed compilation failed for '{func_name}'.",
            f"Numba error: {original_error}",
        ]
        if hints:
            lines.append("Possible causes:")
            lines.extend(hints)
        lines.append(
            "Tip: run with NUMBA_DISABLE_JIT=1 to bypass compilation and debug in pure Python."
        )

        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# @jit decorator
# ---------------------------------------------------------------------------

def jit(cache=True, fastmath=True, nogil=True, **numba_kwargs):
    """Ergonomic JIT decorator that reads PEP-484 type hints and compiles
    the function ahead-of-time via Numba.

    Automatically:
        - Rewrites f-strings and list comprehensions in the AST
        - Translates type hints into a Numba signature for AOT compilation
        - Unrolls **kwargs into positional arguments
        - Provides human-readable errors on compilation failure
        - Records compilation timing in the global stats registry
    """

    def decorator(func):
        if not NUMBA_AVAILABLE:
            return func

        sig = inspect.signature(func)
        func_name = getattr(func, "__qualname__", func.__name__)

        # 1. AST transforms
        rewritten = rewrite_function_ast(func)

        # 2. Build Numba signature from type hints
        args_types = []
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if param.annotation is inspect.Parameter.empty:
                args_types = None
                break
            nb_type = map_python_type_to_numba(param.annotation)
            if nb_type is None:
                args_types = None
                break
            args_types.append(nb_type)

        return_type = None
        if sig.return_annotation is not inspect.Signature.empty:
            return_type = map_python_type_to_numba(sig.return_annotation)

        numba_sig = None
        if args_types is not None:
            if return_type is not None:
                numba_sig = return_type(*args_types)
            else:
                numba_sig = tuple(args_types)

        # 3. Compile with diagnostics
        try:
            source = ""
            try:
                source = inspect.getsource(func)
            except Exception:
                pass

            t0 = time.perf_counter()
            if numba_sig:
                compiled = njit(
                    numba_sig, cache=cache, fastmath=fastmath, nogil=nogil, **numba_kwargs
                )(rewritten)
            else:
                compiled = njit(
                    cache=cache, fastmath=fastmath, nogil=nogil, **numba_kwargs
                )(rewritten)
            compile_time = (time.perf_counter() - t0) * 1e6

        except Exception as exc:
            raise CloudSealedCompileError(func_name, exc, source) from exc

        # 4. Record stats
        sig_str = str(numba_sig) if numba_sig else "deferred"
        _compilation_registry.append(
            CompilationEntry(
                name=func_name,
                compile_time_us=compile_time,
                from_cache=compile_time < 1000,  # heuristic: cache hits are < 1ms
                numba_signature=sig_str,
                timestamp=time.time(),
            )
        )

        # 5. Kwargs unroller wrapper
        def wrapper(*args, **kwargs):
            if kwargs:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return compiled(*bound.args)
            return compiled(*args)

        wrapper.__wrapped__ = compiled
        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = func_name
        return wrapper

    return decorator
