# Loading real inputs from a `.npz`

By default `ort_runner` **synthesizes** its input tensors (random/ones/zeros) from the model's
declared shapes and dtypes. Pass `--inputs sample.npz` to feed **real** data instead.

The format is a NumPy `.npz` archive written by [`numpy.savez`][savez]: one named array per
model input, where **each array's name matches a model input name**. That name match is the
whole contract — it is how each array is routed to the right input, so a model with two inputs
`input_a` and `input_b` needs a `.npz` with arrays under exactly those two keys.

One `.npz` = one **sample** (the complete set of inputs for one inference call). The same sample
is run repeatedly for the benchmark, exactly as synthesized inputs are today.

## The rules

- **One file, many inputs.** `np.savez("sample.npz", input_a=a, input_b=b)` — the keyword names
  become the array names, which must equal the model's input names.
- **Partial is fine.** Any input *not* present in the archive is synthesized as usual. So you can
  pin one input to real data and let the rest be generated.
- **dtype and static dims must match the model.** A mismatched dtype, rank, or statically-declared
  dimension is a hard error naming the offending input. Dynamic/symbolic dimensions instead take
  the array's own size (so `--dim`/`--default-dim` don't apply to a loaded input).
- **Uncompressed only.** Use `numpy.savez` (uncompressed), **not** `numpy.savez_compressed` —
  the reader is dependency-free and reads only STORED zip members. A compressed archive errors
  with a message telling you to re-save.
- **C-contiguous, little-endian.** What `numpy.save` writes by default. Big-endian or
  Fortran-ordered arrays are rejected. Supported dtypes:
  `float32, float64, int64, int32, int16, int8, uint8, bool`.

## Try it

Generate the toy 2-input model and a matching sample (needs [uv]):

```bash
uv run examples/load_inputs/make_demo.py
```

Confirm the input names/shapes/dtypes you need to match:

```bash
just run-linux examples/load_inputs/add_two.onnx --list-io
```

Run the benchmark on the real sample:

```bash
just run-linux examples/load_inputs/add_two.onnx --inputs examples/load_inputs/sample.npz
```

In the preamble each input line ends with `source=file:...` (loaded) or `source=synth`
(generated), so you can see at a glance which inputs came from the archive.

## Producing a `.npz` for your own model

```python
import numpy as np

# One array per model input; keys must equal the model's input names, dtypes must match.
np.savez(
    "sample.npz",
    input_ids=ids.astype("int64"),
    attention_mask=mask.astype("int64"),
)
```

Then: `ort_runner --model your_model.onnx --inputs sample.npz`.

[savez]: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
[uv]: https://docs.astral.sh/uv/
