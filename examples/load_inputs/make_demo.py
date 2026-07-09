# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "onnx"]
# ///
"""Generate a 2-input toy ONNX model and a matching sample.npz for the --inputs demo.

Run with uv (no manual venv needed):

    uv run examples/load_inputs/make_demo.py

It writes two files next to itself:
  - add_two.onnx : a model with two float32[1, 4] inputs, "input_a" and "input_b"
  - sample.npz   : one array per input, keyed by the exact input names

The npz keys are the model's input names -- that name match is the whole contract ort_runner
relies on to route each array to the right input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).parent
SHAPE = [1, 4]


def build_model() -> onnx.ModelProto:
    """A minimal two-input graph: sum = input_a + input_b."""
    input_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, SHAPE)
    input_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, SHAPE)
    output = helper.make_tensor_value_info("sum", TensorProto.FLOAT, SHAPE)
    add = helper.make_node("Add", ["input_a", "input_b"], ["sum"])
    graph = helper.make_graph([add], "add_two", [input_a, input_b], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


def build_sample() -> dict[str, np.ndarray]:
    """One deterministic array per model input, dtype/shape matching the model exactly."""
    rng = np.random.default_rng(0)
    return {
        "input_a": rng.standard_normal(SHAPE).astype(np.float32),
        "input_b": rng.standard_normal(SHAPE).astype(np.float32),
    }


def main() -> None:
    model_path = HERE / "add_two.onnx"
    onnx.save(build_model(), model_path)

    sample = build_sample()
    npz_path = HERE / "sample.npz"
    np.savez(npz_path, **sample)  # keyword names become the archive member names

    print(f"wrote {model_path}")
    print(f"wrote {npz_path} with arrays: {', '.join(sample)}")


if __name__ == "__main__":
    main()
