#!/usr/bin/env python3
"""Translate the supported textual DxCGC subset to HIP dialect MLIR."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OP_MAP = {
    "add": "add", "mul": "mul", "sub": "sub", "div": "div",
    "min": "min", "max": "max", "equal": "equal", "less": "less",
    "and": "and", "or": "or", "not": "not", "abs": "abs",
    "cos": "cos", "sin": "sin", "ceil": "ceil", "exp": "exp",
    "log": "log", "sign": "sign", "reshape": "reshape",
    "transpose": "transpose", "cast": "cast", "matmul": "matmul",
    "gemm": "gemm", "convolution": "conv", "max_pooling": "pool",
    "average_pooling": "pool", "global_avg_pool": "global_pool",
}

TYPE_RE = re.compile(r"!cgc\.tensor<([^<>]*)>|!dxgml\.tensor<([^<>]*)>")
ELEMENT_RE = re.compile(r"!cgc\.|!dxgml\.")
ARG_RE = re.compile(r"%(\w+)\s*:\s*([^,\)]+)")
OP_RE = re.compile(
    r"%(?P<result>\w+)\s*=\s*(?P<namespace>cgc_op|cgc|dxgml_op)\.(?P<op>\w+)"
    r"\s*\((?P<operands>[^)]*)\)\s*(?P<attrs>\{[^}]*\})?\s*"
    r":\s*\((?P<input_types>[^)]*)\)\s*->\s*(?P<result_type>[^\s\n]+)"
)
RETURN_RE = re.compile(r"(?:cgc|dxgml|func)\.return\s+([^:]+)\s*:\s*(.+)")
FUNC_RE = re.compile(
    r"func\.func\s+@(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*"
    r"(?:->\s*(?P<return_type>[^\s{]+))?\s*\{(?P<body>.*?)\n\s*\}",
    re.DOTALL,
)


class ConversionError(ValueError):
    pass


def normalize_type(type_text: str) -> str:
    normalized = TYPE_RE.sub(
        lambda match: "tensor<" + (match.group(1) or match.group(2)) + ">",
        type_text.strip(),
    )
    normalized = ELEMENT_RE.sub("", normalized)
    scalar_types = {
        "float16": "f16", "float32": "f32", "float64": "f64",
        "bfloat16": "bf16", "bool": "i1",
        "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
        "uint8": "ui8", "uint16": "ui16", "uint32": "ui32", "uint64": "ui64",
    }
    for source, target in scalar_types.items():
        normalized = normalized.replace(source, target)
    return normalized


def split_values(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def convert_function(match: re.Match[str]) -> str:
    name = match.group("name")
    args = [item for item in ARG_RE.finditer(match.group("args"))]
    body = match.group("body")
    return_match = RETURN_RE.search(body)
    if return_match is None:
        raise ConversionError(f"function @{name} has no return")
    return_value = return_match.group(1).strip()
    return_type = normalize_type(return_match.group(2))
    output = [f"  func.func @{name}(%ctx: !hip.context"]
    output.extend(f", %{arg.group(1)}: {normalize_type(arg.group(2))}" for arg in args)
    output.append(f") -> {return_type} {{")
    for op_match in OP_RE.finditer(body):
        source_op = op_match.group("op")
        target_op = OP_MAP.get(source_op)
        if target_op is None:
            raise ConversionError(f"unsupported CGC operation '{source_op}' in function @{name}")
        if op_match.group("attrs"):
            raise ConversionError(f"attributes on '{source_op}' are not yet supported")
        operands = split_values(op_match.group("operands"))
        input_types = split_values(op_match.group("input_types"))
        if len(operands) != len(input_types):
            raise ConversionError(f"operand/type count mismatch for '{source_op}'")
        result_type = normalize_type(op_match.group("result_type"))
        operand_names = ", ".join(operands)
        operand_types = ", ".join(normalize_type(type_text) for type_text in input_types)
        result = op_match.group("result")
        output.append(f"    %{result}_init = tensor.empty() : {result_type}")
        output.append(
            f"    %{result} = hip.{target_op}(%ctx) ins({operand_names} : {operand_types}) "
            f"outs(%{result}_init : {result_type})"
        )
    output.append(f"    return {return_value} : {return_type}")
    output.append("  }")
    return "\n".join(output)


def convert(text: str) -> str:
    if "cgc.module" not in text and "dxgml.module" not in text:
        raise ConversionError("input does not contain a cgc.module or dxgml.module")
    functions = list(FUNC_RE.finditer(text))
    if not functions:
        raise ConversionError("input contains no func.func entry point")
    return "module {\n" + "\n\n".join(convert_function(f) for f in functions) + "\n}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    try:
        result = convert(args.input.read_text(encoding="utf-8"))
    except (OSError, ConversionError) as error:
        print(f"dxcgc-to-hip: error: {error}", file=sys.stderr)
        return 1
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



