# How we made Numba ergonomic: AST transforms, type hints, and a diagnostic layer

*Originally published at [dev.to/cloudsealed](https://dev.to/cloudsealed)*

---

Numba is one of the most powerful JIT compilers available for Python. It can
take a loop that runs in 50 ms and make it run in 50 µs — real LLVM-compiled
machine code, not CPython bytecode. But there's a problem: Numba's `nopython`
mode is notoriously hostile to idiomatic Python.

Here's what raw Numba refuses to compile:

```python
from numba import njit

@njit
def process(data, label):
    result = [x * 2 for x in data]   # TypingError: list comprehensions
    print(f"Processing: {label}")     # TypingError: f-strings
    return result[0]
```

Both of those lines — a list comprehension and an f-string — are completely
standard Python. But `nopython` mode has no support for either. The error
message Numba gives you is a 40-line LLVM traceback that, unless you already
know the internals, tells you almost nothing useful.

We built the **CloudSealed Compiler** to solve exactly this problem. It's
open-source, ships inside [`cloudsealed-jit`](https://github.com/cloudsealed/JIT-Optimization-Engine),
and this is how it works.

---

## The approach: rewrite at the AST level

The insight is simple: Numba's restriction is at the IR level, not the source
level. If we rewrite the Python source into an equivalent form that Numba
*does* accept — before Numba ever sees it — the problem goes away.

Python gives us everything we need: the `ast` module to parse and transform
source, `inspect.getsource()` to get the source of a function at runtime, and
`exec()` to create a new function object from the transformed source.

The compiler applies two transforms:

---

### Transform 1: F-strings → `str()` + concatenation

F-strings compile to a `JoinedStr` node in the AST. We replace every
`JoinedStr` with a chain of `str()` calls and `+` operators:

```python
class FStringTransformer(ast.NodeTransformer):
    def visit_JoinedStr(self, node):
        self.generic_visit(node)
        if not node.values:
            return ast.Constant(value="")

        result = None
        for val in node.values:
            if isinstance(val, ast.Constant):
                expr = val
            elif isinstance(val, ast.FormattedValue):
                # Drop format spec — Numba can't format floats anyway
                expr = ast.Call(
                    func=ast.Name(id="str", ctx=ast.Load()),
                    args=[val.value],
                    keywords=[],
                )
            else:
                expr = val

            result = expr if result is None else ast.BinOp(
                left=result, op=ast.Add(), right=expr
            )
        return result
```

Before → after:

```python
# Before
msg = f"Processing {label} at cost {amount}"

# After (what Numba sees)
msg = "Processing " + str(label) + " at cost " + str(amount)
```

---

### Transform 2: List comprehensions → explicit for-loops

`ListComp` nodes get rewritten into an `init + for loop + append` pattern.
We handle `if` filters correctly:

```python
class ListCompTransformer(ast.NodeTransformer):
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
            targets=[ast.Name(id=target.id, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
            lineno=target.lineno, col_offset=target.col_offset,
        )
        append_call = ast.Expr(value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=target.id, ctx=ast.Load()),
                attr="append", ctx=ast.Load()
            ),
            args=[listcomp.elt], keywords=[],
        ))

        body = [append_call]
        for if_clause in reversed(gen.ifs):
            body = [ast.If(test=if_clause, body=body, orelse=[])]

        loop = ast.For(target=gen.target, iter=gen.iter, body=body, orelse=[])
        return [init, loop]
```

Before → after:

```python
# Before
result = [x * 2 for x in data if x > 0]

# After (what Numba sees)
result = []
for x in data:
    if x > 0:
        result.append(x * 2)
```

---

## PEP-484 type hints → Numba signatures

Raw Numba requires you to spell out signatures as strings:

```python
@njit("float64[:](float64[:], float64)")
def scale(arr, factor):
    ...
```

This is error-prone, not composable with type checkers, and breaks when
you refactor. We built `map_python_type_to_numba()` to translate standard
Python type hints automatically:

```python
def map_python_type_to_numba(py_type):
    if not NUMBA_AVAILABLE:
        return None

    _SCALAR_MAP = {
        int: types.int64, float: types.float64,
        bool: types.boolean, str: types.unicode_type,
        np.float32: types.float32, np.float64: types.float64,
    }
    if py_type in _SCALAR_MAP:
        return _SCALAR_MAP[py_type]

    if py_type is np.ndarray:
        return types.Array(types.float64, 1, "C")

    origin = getattr(py_type, "__origin__", None)

    if origin is typing.Union:          # Optional[T]
        non_none = [a for a in py_type.__args__ if a is not type(None)]
        if len(non_none) == 1:
            inner = map_python_type_to_numba(non_none[0])
            if inner: return types.Optional(inner)

    if origin is list:                  # List[T]
        item = map_python_type_to_numba(py_type.__args__[0])
        if item: return types.ListType(item)

    if origin is dict:                  # Dict[K, V]
        k = map_python_type_to_numba(py_type.__args__[0])
        v = map_python_type_to_numba(py_type.__args__[1])
        if k and v: return types.DictType(k, v)

    if origin is tuple:                 # Tuple[T1, T2, ...]
        mapped = [map_python_type_to_numba(a) for a in py_type.__args__]
        if all(m is not None for m in mapped):
            return types.Tuple(mapped)

    return None
```

Combined with `inspect.signature()`, the `@jit` decorator now compiles
functions AOT from their type hints alone:

```python
@jit()
def scale(arr: np.ndarray, factor: float) -> np.ndarray:
    return arr * factor
```

No string signature. No manual Numba types. The function is compiled before
the first call.

---

## `@jitdataclass`: Python dataclasses as C-structs

Numba's `@jitclass` works, but it requires manually spelling out a spec list
and rewriting your `__init__`. The `@jitdataclass` decorator bridges the gap:

```python
@jitdataclass
class Particle:
    x: float
    y: float
    mass: float         # Numba field

@jit()
def kinetic_energy(p: Particle, v: float) -> float:
    return 0.5 * p.mass * v ** 2
```

Under the hood it:

1. Calls `dataclass(cls)` if not already done
2. Iterates `dataclasses.fields()` and maps each type to a Numba spec
3. Generates a clean `__init__` (stripping dunder methods that `@jitclass` rejects)
4. Calls `jitclass(spec)(CleanClass)` and registers the result for nested resolution

Default values work. User methods are preserved and compiled. Nested
`@jitdataclass` fields resolve to the correct `StructRef` type automatically.

---

## Human-readable errors

Numba's compilation errors are famously cryptic:

```
numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
  non-precise type pyobject
  [1] During: typing of argument value at ...
```

We wrap the Numba call and produce a `CloudSealedCompileError` instead:

```python
try:
    compiled = njit(numba_sig, cache=cache, fastmath=fastmath)(rewritten)
except Exception as exc:
    raise CloudSealedCompileError(func_name, exc, source) from exc
```

```
CloudSealed compilation failed for 'my_func'.
Numba error: Failed in nopython mode pipeline...
Possible causes:
  → Sets are not supported in Numba nopython mode. Use a typed List or array instead.
Tip: run with NUMBA_DISABLE_JIT=1 to bypass compilation and debug in pure Python.
```

---

## Compilation profiler

Every compiled function is recorded in a global registry:

```python
from cloudsealed_jit import compilation_stats

stats = compilation_stats()
# [
#   {'name': 'mymodule.scale', 'compile_time_us': 18.3,
#    'from_cache': True, 'signature': 'float64[:](float64[:], float64)'},
# ]
```

Useful for identifying cold-start costs and verifying cache hits.

---

## Putting it all together

Here's what the full `@jit` decorator does, in order:

```python
def jit(cache=True, fastmath=True, nogil=True, **numba_kwargs):
    def decorator(func):
        if not NUMBA_AVAILABLE:
            return func                          # silent pure-Python fallback

        sig = inspect.signature(func)

        # 1. AST transforms
        rewritten = rewrite_function_ast(func)   # f-strings + list comps

        # 2. Build Numba signature from type hints
        numba_sig = _build_signature(sig)        # PEP-484 → Numba IR types

        # 3. Compile (with diagnostics on failure)
        t0 = time.perf_counter()
        compiled = njit(numba_sig, ...)(rewritten)
        compile_time = (time.perf_counter() - t0) * 1e6

        # 4. Record stats
        _compilation_registry.append(CompilationEntry(...))

        # 5. Kwargs unroller
        def wrapper(*args, **kwargs):
            if kwargs:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return compiled(*bound.args)
            return compiled(*args)

        wrapper.__name__ = func.__name__
        return wrapper
    return decorator
```

---

## Benchmarks

Measured on x86-64 Linux, Python 3.11:

| Operation | Time |
|---|---|
| F-string AST rewrite | ~680 µs median |
| List comprehension AST rewrite | ~680 µs median |
| Kwargs dispatch overhead | ~0.05 µs/call |

The AST rewrite happens once at decoration time, not at each call. The kwargs
overhead is negligible for any real numerical workload.

---

## Why not Cython or mypyc?

| | `@jit` (this project) | raw `@njit` | Cython | mypyc |
|---|---|---|---|---|
| Type hints drive compilation | ✓ | ✗ | ✓ (pxd) | ✓ |
| f-strings | ✓ | ✗ | ✓ | ✓ |
| List comprehensions | ✓ | ✗ | ✓ | ✓ |
| Kwargs | ✓ | ✗ | ✓ | ✓ |
| Pure Python fallback | ✓ | ✗ | ✗ | ✗ |
| Build step required | ✗ | ✗ | ✓ | ✓ |
| Targets LLVM / SIMD | ✓ | ✓ | ✗ | ✗ |

Cython and mypyc require a compilation step separate from normal Python
execution — you can't just `pip install` and use them dynamically. Numba
(and this wrapper) compile at import time with no separate build step.

---

## Try it

```bash
pip install cloudsealed-jit
pip install "cloudsealed-jit[jit]"   # + Numba
```

```python
from cloudsealed_jit import jit, jitdataclass

@jitdataclass
class Vec2:
    x: float
    y: float

@jit()
def magnitude(v: Vec2) -> float:
    components = [v.x ** 2, v.y ** 2]   # list comp — rewritten automatically
    return (components[0] + components[1]) ** 0.5

print(magnitude(Vec2(3.0, 4.0)))   # 5.0
```

Source: [github.com/cloudsealed/JIT-Optimization-Engine](https://github.com/cloudsealed/JIT-Optimization-Engine)

---

*If this is useful for your numerical work, a GitHub star helps others find it.
Issues and PRs welcome.*
