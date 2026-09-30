"""Tests for the CloudSealed compiler and dataclass compiler."""

import ast
import math
import textwrap
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from cloudsealed_jit.compiler import (
    CloudSealedCompileError,
    FStringTransformer,
    ListCompTransformer,
    compilation_stats,
    jit,
    map_python_type_to_numba,
    reset_compilation_stats,
    rewrite_function_ast,
)
from cloudsealed_jit.dataclass_compiler import jitdataclass

# Try importing Numba to decide which tests can run.
try:
    from numba.core import types

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

needs_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")


# -----------------------------------------------------------------------
# AST Transform: FStringTransformer
# -----------------------------------------------------------------------


class TestFStringTransformer:
    def _transform(self, src: str) -> str:
        tree = ast.parse(textwrap.dedent(src))
        tree = FStringTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def test_simple_fstring(self):
        result = self._transform('x = f"hello {name}"')
        assert "str(name)" in result
        assert "+" in result

    def test_empty_fstring(self):
        result = self._transform("x = f''")
        assert "''" in result

    def test_multiple_expressions(self):
        result = self._transform('x = f"{a} and {b}"')
        assert "str(a)" in result
        assert "str(b)" in result

    def test_format_spec_is_handled(self):
        result = self._transform('x = f"{val:.2f}"')
        assert "str(val)" in result

    def test_nested_expression(self):
        result = self._transform('x = f"result: {a + b}"')
        assert "str(a + b)" in result


# -----------------------------------------------------------------------
# AST Transform: ListCompTransformer
# -----------------------------------------------------------------------


class TestListCompTransformer:
    def _transform(self, src: str) -> str:
        tree = ast.parse(textwrap.dedent(src))
        tree = ListCompTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def test_simple_listcomp(self):
        result = self._transform("result = [x * 2 for x in items]")
        assert "for x in items" in result
        assert "append" in result
        assert "[x * 2 for" not in result

    def test_listcomp_with_condition(self):
        result = self._transform("result = [x for x in items if x > 0]")
        assert "if x > 0" in result
        assert "append" in result

    def test_non_listcomp_assignment_untouched(self):
        src = "result = some_function()"
        result = self._transform(src)
        assert result.strip() == src.strip()

    def test_rewritten_code_executes_correctly(self):
        src = "result = [x ** 2 for x in range(5)]"
        tree = ast.parse(src)
        tree = ListCompTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        code = compile(tree, "<test>", "exec")
        ns = {}
        exec(code, ns)
        assert ns["result"] == [0, 1, 4, 9, 16]

    def test_rewritten_with_filter_executes(self):
        src = "result = [x for x in range(10) if x % 2 == 0]"
        tree = ast.parse(src)
        tree = ListCompTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        code = compile(tree, "<test>", "exec")
        ns = {}
        exec(code, ns)
        assert ns["result"] == [0, 2, 4, 6, 8]


# -----------------------------------------------------------------------
# Type Mapping
# -----------------------------------------------------------------------


class TestTypeMapping:
    @needs_numba
    def test_scalar_types(self):
        assert map_python_type_to_numba(int) == types.int64
        assert map_python_type_to_numba(float) == types.float64
        assert map_python_type_to_numba(bool) == types.boolean
        assert map_python_type_to_numba(str) == types.unicode_type

    @needs_numba
    def test_numpy_types(self):
        assert map_python_type_to_numba(np.int32) == types.int32
        assert map_python_type_to_numba(np.float64) == types.float64

    @needs_numba
    def test_ndarray(self):
        result = map_python_type_to_numba(np.ndarray)
        assert isinstance(result, types.Array)

    @needs_numba
    def test_optional(self):
        result = map_python_type_to_numba(Optional[float])
        assert isinstance(result, types.Optional)

    @needs_numba
    def test_list(self):
        result = map_python_type_to_numba(List[int])
        assert isinstance(result, types.ListType)

    @needs_numba
    def test_tuple(self):
        result = map_python_type_to_numba(Tuple[int, float])
        assert isinstance(result, types.BaseTuple)

    @needs_numba
    def test_dict(self):
        result = map_python_type_to_numba(Dict[str, float])
        assert isinstance(result, types.DictType)

    @needs_numba
    def test_string_hints(self):
        assert map_python_type_to_numba("int") == types.int64
        assert map_python_type_to_numba("float") == types.float64

    @needs_numba
    def test_unsupported_returns_none(self):
        assert map_python_type_to_numba(object) is None
        assert map_python_type_to_numba(bytes) is None

    def test_without_numba_returns_none(self):
        with patch("cloudsealed_jit.compiler.NUMBA_AVAILABLE", False):
            assert map_python_type_to_numba(int) is None


# -----------------------------------------------------------------------
# @jit decorator
# -----------------------------------------------------------------------


class TestJitDecorator:
    def setup_method(self):
        reset_compilation_stats()

    @needs_numba
    def test_basic_jit_function(self):
        @jit()
        def add(a: float, b: float) -> float:
            return a + b

        assert add(2.0, 3.0) == pytest.approx(5.0)

    @needs_numba
    def test_kwargs_support(self):
        @jit()
        def multiply(a: float, b: float) -> float:
            return a * b

        assert multiply(a=3.0, b=4.0) == pytest.approx(12.0)
        assert multiply(4.0, b=5.0) == pytest.approx(20.0)

    @needs_numba
    def test_array_input(self):
        @jit()
        def array_sum(arr: np.ndarray) -> float:
            total = 0.0
            for i in range(arr.shape[0]):
                total += arr[i]
            return total

        assert array_sum(np.array([1.0, 2.0, 3.0])) == pytest.approx(6.0)

    @needs_numba
    def test_compilation_stats_recorded(self):
        @jit()
        def trivial(x: float) -> float:
            return x

        trivial(1.0)
        stats = compilation_stats()
        assert len(stats) >= 1
        assert stats[-1]["name"].endswith("trivial")
        assert "compile_time_us" in stats[-1]

    def test_fallback_without_numba(self):
        with patch("cloudsealed_jit.compiler.NUMBA_AVAILABLE", False):
            @jit()
            def plain(x):
                return x * 2

            assert plain(5) == 10

    @needs_numba
    def test_no_type_hints_still_compiles(self):
        @jit()
        def untyped(a, b):
            return a + b

        assert untyped(2.0, 3.0) == pytest.approx(5.0)

    @needs_numba
    def test_wrapper_preserves_name(self):
        @jit()
        def my_func(x: float) -> float:
            return x

        assert my_func.__name__ == "my_func"


# -----------------------------------------------------------------------
# Diagnostic errors
# -----------------------------------------------------------------------


class TestDiagnostics:
    @needs_numba
    def test_compile_error_is_human_readable(self):
        with pytest.raises(CloudSealedCompileError, match="compilation failed"):
            @jit()
            def bad_func(x: float) -> float:
                return set([x])  # sets not supported in nopython

            bad_func(1.0)


# -----------------------------------------------------------------------
# @jitdataclass
# -----------------------------------------------------------------------


class TestJitDataclass:
    @needs_numba
    def test_basic_dataclass(self):
        @jitdataclass
        class Point:
            x: float
            y: float

        p = Point(1.0, 2.0)
        assert p.x == pytest.approx(1.0)
        assert p.y == pytest.approx(2.0)

    @needs_numba
    def test_default_values(self):
        @jitdataclass
        class Config:
            threshold: float
            enabled: int

        c = Config(3.5, 1)
        assert c.threshold == pytest.approx(3.5)

    @needs_numba
    def test_used_in_jit_function(self):
        @jitdataclass
        class Vec2:
            x: float
            y: float

        @jit()
        def magnitude(v: Vec2) -> float:
            return (v.x ** 2 + v.y ** 2) ** 0.5

        v = Vec2(3.0, 4.0)
        assert magnitude(v) == pytest.approx(5.0)

    def test_fallback_without_numba(self):
        with patch("cloudsealed_jit.dataclass_compiler.NUMBA_AVAILABLE", False):
            @jitdataclass
            class Simple:
                x: float
                y: float

            s = Simple(x=1.0, y=2.0)
            assert s.x == 1.0

    @needs_numba
    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported type"):
            @jitdataclass
            class Bad:
                data: bytes


# -----------------------------------------------------------------------
# Compilation stats
# -----------------------------------------------------------------------


class TestCompilationStats:
    def setup_method(self):
        reset_compilation_stats()

    def test_stats_empty_initially(self):
        assert compilation_stats() == []

    def test_reset_clears_stats(self):
        reset_compilation_stats()
        assert compilation_stats() == []

    @needs_numba
    def test_multiple_functions_tracked(self):
        @jit()
        def f1(x: float) -> float:
            return x

        @jit()
        def f2(x: float) -> float:
            return x * 2

        stats = compilation_stats()
        names = [s["name"] for s in stats]
        assert any("f1" in n for n in names)
        assert any("f2" in n for n in names)
