#!/usr/bin/env python3
"""Summarize Conv cost in the exported stage2/refiner ONNX graph."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper


DEFAULT_MODEL = "deploy/webgpu/public/models/frame_interpolation_stage2_fp32.onnx"


def get_shape(value_info):
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            return None
    return dims


def shape_map(model):
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for value in [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]:
        shape = get_shape(value)
        if shape:
            shapes[value.name] = shape
    return shapes


def initializer_map(model):
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def conv_attrs(node):
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    kernel = tuple(attrs.get("kernel_shape", ()))
    strides = tuple(attrs.get("strides", (1, 1)))
    pads = tuple(attrs.get("pads", (0, 0, 0, 0)))
    groups = int(attrs.get("group", 1))
    return kernel, strides, pads, groups


def first_consumer_by_input(nodes):
    consumers = defaultdict(list)
    for node in nodes:
        for name in node.input:
            consumers[name].append(node)
    return consumers


def fusable_activation(conv_node, consumers, initializers):
    output = conv_node.output[0]
    users = consumers.get(output, [])
    if len(users) != 1 or users[0].op_type != "PRelu":
        return "-"
    prelu = users[0]
    if len(prelu.input) < 2:
        return "PRelu"
    slope = initializers.get(prelu.input[1])
    if slope is None:
        return "PRelu"
    return f"PRelu{slope.shape}"


def conv_rows(model):
    shapes = shape_map(model)
    initializers = initializer_map(model)
    consumers = first_consumer_by_input(model.graph.node)
    rows = []

    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        weight = initializers.get(node.input[1] if len(node.input) > 1 else "")
        if weight is None or weight.ndim != 4:
            continue
        output_shape = shapes.get(node.output[0])
        input_shape = shapes.get(node.input[0])
        if not output_shape or len(output_shape) != 4:
            continue

        kernel, strides, pads, groups = conv_attrs(node)
        batch, cout, height, width = output_shape
        cin_per_group = weight.shape[1]
        macs = batch * cout * height * width * cin_per_group * weight.shape[2] * weight.shape[3]
        rows.append(
            {
                "index": index,
                "name": node.name or f"Conv_{index}",
                "input": input_shape,
                "output": output_shape,
                "weight": tuple(weight.shape),
                "kernel": kernel or tuple(weight.shape[2:]),
                "stride": strides,
                "pads": pads,
                "groups": groups,
                "bias": "yes" if len(node.input) >= 3 and node.input[2] in initializers else "no",
                "activation": fusable_activation(node, consumers, initializers),
                "gmac": macs / 1e9,
            }
        )
    return rows


def print_table(rows, limit):
    selected = sorted(rows, key=lambda row: row["gmac"], reverse=True)
    if limit:
        selected = selected[:limit]

    header = (
        f"{'#':>3}  {'GMAC':>8}  {'kernel':>6}  {'stride':>6}  "
        f"{'in -> out':>28}  {'bias':>4}  {'act':>14}  name"
    )
    print(header)
    print("-" * len(header))
    for row in selected:
        input_shape = row["input"] or ["?"]
        output_shape = row["output"]
        in_ch = input_shape[1] if len(input_shape) > 1 else "?"
        out_ch = output_shape[1]
        hw = f"{output_shape[2]}x{output_shape[3]}"
        print(
            f"{row['index']:>3}  {row['gmac']:8.3f}  "
            f"{format_pair(row['kernel']):>6}  {format_pair(row['stride']):>6}  "
            f"{in_ch}->{out_ch} @ {hw:>14}  "
            f"{row['bias']:>4}  {row['activation']:>14}  {row['name']}"
        )


def format_pair(values):
    if not values:
        return "?"
    if len(values) == 2 and values[0] == values[1]:
        return f"{values[0]}x{values[1]}"
    return "x".join(str(value) for value in values)


def print_summary(rows):
    total = sum(row["gmac"] for row in rows)
    by_kernel = defaultdict(float)
    by_fusable = defaultdict(float)
    for row in rows:
        by_kernel[row["kernel"]] += row["gmac"]
        key = "Conv+PRelu" if row["activation"].startswith("PRelu") else "Conv only/other"
        by_fusable[key] += row["gmac"]

    print()
    print(f"Conv nodes: {len(rows)}")
    print(f"Total Conv cost: {total:.3f} GMAC")
    for kernel, gmac in sorted(by_kernel.items(), key=lambda item: item[1], reverse=True):
        pct = 100.0 * gmac / total if total else 0.0
        print(f"  {format_pair(kernel):>6}: {gmac:.3f} GMAC ({pct:.1f}%)")
    for key, gmac in sorted(by_fusable.items(), key=lambda item: item[1], reverse=True):
        pct = 100.0 * gmac / total if total else 0.0
        print(f"  {key}: {gmac:.3f} GMAC ({pct:.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Conv GMAC and Conv+PReLU fusion candidates in stage2 ONNX."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Path to stage2 ONNX model. Defaults to {DEFAULT_MODEL}",
    )
    parser.add_argument("--top", type=int, default=12, help="Number of heaviest Conv nodes to print")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    model = onnx.load(str(model_path))
    rows = conv_rows(model)
    print(f"Model: {model_path}")
    print_table(rows, args.top)
    print_summary(rows)


if __name__ == "__main__":
    main()
