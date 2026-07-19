# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "onnx"]
# ///
"""Generate a reduction over a large, fully dynamic tensor -- a benchmark that takes real time.

Run with uv (no manual venv needed):

    uv run examples/large_reduce/make_model.py

The toy add_two model next door finishes in about three microseconds, which is useful for
checking that inputs are routed correctly and useless for checking anything about performance:
at that scale the numbers are dominated by call overhead, memory deltas round to nothing, and
the tail is indistinguishable from noise.

This model reduces a float32 tensor to a scalar mean. The work is memory-bandwidth bound and
proportional to the input, so sizing the input decides how long a run takes.

Every dimension is symbolic -- "batch", "height", "width" -- and none has a size baked in. That
is the point: the same file covers a 4 KiB run and a 256 MiB one, and it exercises the whole
dimension-resolution path (`--dim`, `--default-dim`, and the warning for a `--dim` naming an axis
the model does not have).

    # 16 MiB input, lands in the millisecond range
    ort_runner --model large_reduce.onnx --dim batch=1 --dim height=2048 --dim width=2048

    # every dynamic axis takes the same fallback
    ort_runner --model large_reduce.onnx --default-dim 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).parent
BYTES_PER_FLOAT32 = 4

# Named so `--dim <name>=<size>` can address each one individually. An anonymous dynamic axis is
# only reachable through --default-dim, which makes for a poorer fixture.
AXES = ("batch", "height", "width")


def build() -> onnx.ModelProto:
    """A model computing the mean of a fully dynamic float32 tensor."""
    tensor_input = helper.make_tensor_value_info("data", TensorProto.FLOAT, list(AXES))
    scalar_output = helper.make_tensor_value_info("mean", TensorProto.FLOAT, [])

    # No axes given, so every axis is reduced; keepdims=0 makes the output a true scalar.
    node = helper.make_node("ReduceMean", inputs=["data"], outputs=["mean"], keepdims=0)

    graph = helper.make_graph(
        [node],
        "large_reduce",
        [tensor_input],
        [scalar_output],
        doc_string="Mean over a fully dynamic [batch, height, width] float32 tensor.",
    )

    model = helper.make_model(
        graph,
        # Pinned, and deliberately well below the newest. The onnx package stamps whatever opset
        # it was built against -- 27 at time of writing -- and ONNX Runtime refuses anything past
        # the last officially released one, so an unpinned fixture stops loading whenever the
        # generating environment updates. 17 is old enough to be accepted everywhere and new
        # enough for a plain reduction.
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="ort_runner-examples",
        producer_version="1",
        doc_string=(
            "Benchmark fixture: a memory-bandwidth-bound reduction over a fully dynamic input, "
            "so the run can be sized from the command line."
        ),
    )
    # The same metadata_props a real exporter would use for a training run id or a git commit;
    # here it lets a benchmark report identify which fixture produced a number.
    model.metadata_props.append(
        onnx.StringStringEntryProto(key="fixture", value="large_reduce")
    )

    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suggest-side",
        type=int,
        default=2048,
        help="size used only for the printed example command (default: 2048)",
    )
    args = parser.parse_args()

    destination = HERE / "large_reduce.onnx"
    onnx.save(build(), destination)

    side = args.suggest_side
    megabytes = (side * side * BYTES_PER_FLOAT32) / (1024 * 1024)

    print(f"wrote {destination}")
    print(f"  input  data: float32[{', '.join(AXES)}]  (every axis dynamic)")
    print("  output mean: float32 scalar")
    print()
    print(f"  suggested run ({megabytes:.0f} MiB input):")
    print(
        f"    --model {destination.name} "
        f"--dim batch=1 --dim height={side} --dim width={side}"
    )


if __name__ == "__main__":
    main()
