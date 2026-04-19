# Copyright 2026 Eresh Gorantla
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Output validation — JSON schema checking, assertions, and retry prompt building.

Used by the runtime dispatcher to validate LLM outputs against user-defined
schemas and assertions, with optional retry-on-invalid support.
"""

from __future__ import annotations

import ast
import operator
from typing import Any


# ---------------------------------------------------------------------------
# JSON Schema validation
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def validate_schema(output: Any, schema: dict[str, Any]) -> list[str]:
    """Validate output against a JSON Schema. Returns a list of error strings.

    Uses the ``jsonschema`` library if installed (full JSON Schema support,
    collects all errors). Falls back to a lightweight built-in validator.
    """
    try:
        import jsonschema
        validator_cls = jsonschema.Draft7Validator
        validator = validator_cls(schema)
        errors = sorted(validator.iter_errors(output), key=lambda e: list(e.absolute_path))
        return [
            f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}" if e.absolute_path
            else e.message
            for e in errors
        ]
    except ImportError:
        return _validate_lightweight(output, schema, path="")


def _validate_lightweight(output: Any, schema: dict[str, Any], path: str) -> list[str]:
    """Lightweight JSON Schema validation without external dependencies.

    Covers the most common cases: type, required, properties, enum, items.
    """
    errors: list[str] = []
    prefix = f"'{path}': " if path else ""

    # Type check
    expected_type = schema.get("type")
    if expected_type:
        expected = _TYPE_MAP.get(expected_type)
        if expected and not isinstance(output, expected):
            errors.append(f"{prefix}expected type '{expected_type}', got '{type(output).__name__}'")
            return errors  # can't check further if type is wrong

    # Enum check
    if "enum" in schema and output not in schema["enum"]:
        errors.append(f"{prefix}must be one of {schema['enum']}, got {output!r}")

    # Object checks
    if isinstance(output, dict):
        # Required keys
        for key in schema.get("required", []):
            if key not in output:
                errors.append(f"{prefix}missing required key: '{key}'")

        # Properties
        for key, prop_schema in schema.get("properties", {}).items():
            if key in output:
                child_path = f"{path}.{key}" if path else key
                errors.extend(_validate_lightweight(output[key], prop_schema, child_path))

    # Array checks
    if isinstance(output, list) and "items" in schema:
        for i, item in enumerate(output):
            child_path = f"{path}[{i}]" if path else f"[{i}]"
            errors.extend(_validate_lightweight(item, schema["items"], child_path))

    return errors


# ---------------------------------------------------------------------------
# Assertion evaluation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Safe AST-based expression evaluator (replaces eval)
# ---------------------------------------------------------------------------

_SAFE_FUNCS: dict[str, Any] = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "type": type,
    "isinstance": isinstance, "min": min, "max": max, "sum": sum,
    "abs": abs, "round": round, "sorted": sorted, "any": any, "all": all,
}

_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}
_BOOL_OPS = {ast.And: all, ast.Or: any}
_UNARY_OPS = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _safe_eval(expr: str, ns: dict[str, Any]) -> Any:
    """Evaluate a simple Python expression via AST walking — no code execution."""
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body, ns)


def _eval_node(node: ast.AST, ns: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "True": return True
        if node.id == "False": return False
        if node.id == "None": return None
        if node.id in ns:
            return ns[node.id]
        raise NameError(f"name '{node.id}' is not defined")
    if isinstance(node, ast.List):
        return [_eval_node(e, ns) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, ns) for e in node.elts)
    if isinstance(node, ast.Set):
        return {_eval_node(e, ns) for e in node.elts}
    if isinstance(node, ast.Dict):
        return {_eval_node(k, ns): _eval_node(v, ns) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Subscript):
        val = _eval_node(node.value, ns)
        sl = _eval_node(node.slice, ns)
        return val[sl]
    if isinstance(node, ast.Attribute):
        val = _eval_node(node.value, ns)
        attr = node.attr
        # Only allow safe attributes — no dunder access
        if attr.startswith("_"):
            raise AttributeError(f"access to '{attr}' is not allowed")
        return getattr(val, attr)
    if isinstance(node, ast.Index):
        return _eval_node(node.value, ns)
    if isinstance(node, ast.Call):
        func = _eval_node(node.func, ns)
        if func not in _SAFE_FUNCS.values():
            raise TypeError(f"calling {func!r} is not allowed")
        args = [_eval_node(a, ns) for a in node.args]
        return func(*args)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, ns)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, ns)
            fn = _CMP_OPS.get(type(op))
            if fn is None:
                raise TypeError(f"unsupported comparison: {type(op).__name__}")
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, ns) for v in node.values]
        return _BOOL_OPS[type(node.op)](values)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, ns)
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise TypeError(f"unsupported unary op: {type(node.op).__name__}")
        return fn(operand)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, ns)
        right = _eval_node(node.right, ns)
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise TypeError(f"unsupported binary op: {type(node.op).__name__}")
        return fn(left, right)
    if isinstance(node, ast.IfExp):
        return _eval_node(node.body, ns) if _eval_node(node.test, ns) else _eval_node(node.orelse, ns)
    raise TypeError(f"unsupported expression: {type(node).__name__}")


def evaluate_assertions(
    output: Any, assertions: list[str], context: dict[str, Any]
) -> list[str]:
    """Evaluate assertion expressions against the step output.

    Returns a list of failure messages (empty = all passed).
    Uses AST-based evaluation — no exec/eval.

    The assertion has access to:
      - ``output`` — the step's output value
      - All keys from ``output`` if it is a dict (for convenience)
      - Safe builtins (len, str, int, etc.)
    """
    failures: list[str] = []

    eval_ns: dict[str, Any] = {}
    eval_ns.update(_SAFE_FUNCS)
    # Only inject simple string-keyed context values, skip internal objects
    for k, v in context.items():
        if isinstance(k, str) and not k.startswith("_"):
            eval_ns[k] = v
    eval_ns["output"] = output
    if isinstance(output, dict):
        eval_ns.update(output)

    for assertion in assertions:
        try:
            result = _safe_eval(assertion, eval_ns)
            if not result:
                failures.append(f"assertion failed: {assertion}")
        except Exception as e:
            failures.append(f"assertion error in '{assertion}': {e}")

    return failures


# ---------------------------------------------------------------------------
# Combined validation
# ---------------------------------------------------------------------------

def validate_output(
    output: Any,
    schema: dict[str, Any] | None,
    assertions: list[str],
    context: dict[str, Any],
) -> list[str]:
    """Run all validations on a step output. Returns combined error list."""
    errors: list[str] = []

    if schema:
        errors.extend(validate_schema(output, schema))

    if assertions:
        errors.extend(evaluate_assertions(output, assertions, context))

    return errors


# ---------------------------------------------------------------------------
# Retry prompt construction
# ---------------------------------------------------------------------------

def build_retry_prompt(original_prompt: str, raw_output: str, errors: list[str]) -> str:
    """Build a retry prompt that includes the original request and validation errors.

    The LLM sees what went wrong and is asked to fix it.
    """
    error_text = "\n".join(f"- {e}" for e in errors)
    return (
        f"{original_prompt}\n\n"
        f"Your previous response was invalid. Errors:\n{error_text}\n\n"
        f"Previous response:\n{raw_output[:2000]}\n\n"
        f"Please fix the errors and respond again with valid output."
    )
