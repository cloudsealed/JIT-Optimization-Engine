import ast
import inspect
import typing
import textwrap
import numpy as np

try:
    from numba import njit
    from numba.core import types
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class FStringTransformer(ast.NodeTransformer):
    """
    Transforms F-Strings (JoinedStr) into standard string concatenation (Add),
    which Numba supports in nopython mode.
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
                    func=ast.Name(id='str', ctx=ast.Load()),
                    args=[val.value],
                    keywords=[]
                )
            else:
                expr = val
                
            if result is None:
                result = expr
            else:
                result = ast.BinOp(left=result, op=ast.Add(), right=expr)
                
        return result

def remove_decorators(tree):
    class DecoratorRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.decorator_list = []
            self.generic_visit(node)
            return node
    return DecoratorRemover().visit(tree)

def rewrite_function_for_fstrings(func):
    """
    Rewrites the AST of a function to replace f-strings with basic string concatenation.
    Returns the newly compiled function object.
    """
    try:
        src = inspect.getsource(func)
    except Exception:
        return func
    
    src = textwrap.dedent(src)
    tree = ast.parse(src)
    tree = FStringTransformer().visit(tree)
    tree = remove_decorators(tree)
    ast.fix_missing_locations(tree)
    
    new_src = ast.unparse(tree)
    local_env = {}
    
    # We must pass the original globals so it resolves external references
    exec(new_src, func.__globals__, local_env)
    return local_env[func.__name__]


def map_python_type_to_numba(py_type):
    """
    Translates standard Python/Numpy type hints into Numba internal types.
    """
    if not NUMBA_AVAILABLE:
        return None

    if py_type is int: return types.int64
    if py_type is float: return types.float64
    if py_type is bool: return types.boolean
    if py_type is str: return types.unicode_type

    if py_type is np.int32: return types.int32
    if py_type is np.int64: return types.int64
    if py_type is np.float32: return types.float32
    if py_type is np.float64: return types.float64

    if py_type is np.ndarray:
        return types.Array(types.float64, 1, "C")

    if isinstance(py_type, str):
        if py_type == "int": return types.int64
        if py_type == "float": return types.float64
        if py_type == "bool": return types.boolean
    
    origin = getattr(py_type, "__origin__", None)
    if origin is list or origin is typing.List:
        args = getattr(py_type, "__args__", None)
        if args:
            item_type = map_python_type_to_numba(args[0])
            if item_type:
                return types.ListType(item_type)
            
    return None


def jit(cache=True, fastmath=True, nogil=True, **numba_kwargs):
    """
    A smart JIT decorator that reads Python PEP 484 Type Hints and compiles the 
    function Ahead-of-Time using Numba. It automatically translates F-Strings 
    and handles dynamic **kwargs and out-of-order arguments.
    """
    def decorator(func):
        if not NUMBA_AVAILABLE:
            return func

        sig = inspect.signature(func)
        
        # 1. Rewrite F-strings in AST
        func_no_fstrings = rewrite_function_for_fstrings(func)
        
        # 2. Extract Types for AOT Compilation
        args_types = []
        for name, param in sig.parameters.items():
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

        # 3. Apply Numba Compilation
        if numba_sig:
            compiled_func = njit(numba_sig, cache=cache, fastmath=fastmath, nogil=nogil, **numba_kwargs)(func_no_fstrings)
        else:
            compiled_func = njit(cache=cache, fastmath=fastmath, nogil=nogil, **numba_kwargs)(func_no_fstrings)
            
        # 4. Kwargs Unroller Wrapper
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            # Numba gets exactly the positional arguments it expects
            return compiled_func(*bound.args)
            
        return wrapper
    return decorator
