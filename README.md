# lcm

lcm is a fork of [seqax by MatX](https://github.com/MatX-inc/seqax) modified to run on Google's TPU v4-32s and uses the framework outlined by Meta in [Large Concept Models: Language Modeling in a Sentence Representation Space](https://arxiv.org/pdf/2412.08821) as a technique to compress the KV Cache. More details on the implementation and experiments can be read [here](./docs/lcm.ipynb).

The installation procedure is identical to that described in [seqax](https://github.com/MatX-inc/seqax).

## Getting started

### Installation

1. Install `graphviz` from your system package manager: e.g. `brew install graphviz` or `apt install graphviz`.
2. Install Python dependencies, typically inside a virtualenv: `python -m pip install -r requirements-cpu.txt`.

   NOTE: the `requirements-cpu.txt` is configured for CPU-based installation. For GPU or TPU installation, you may need a different install of JAX and jaxlib. Consult the [JAX install documentation](https://jax.readthedocs.io/en/latest/installation.html). If your GPU environment has a Torch-GPU installation, you may need to switch it to a Torch-CPU installation to avoid conflicts with JAX-GPU.

### Run on CPU for local development

For development and testing you can run on CPU. Typically you'd use our synthetic dataset (which is [checked into this repository](/synthetic_dataset.zarr)) or the [Huggingface data loader](./input_loader.py#L355) and you'd set XLA flags to simulate multiple devices so as to test that parallelism is working as intended:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 python -m train --config-name=local_test_synthetic +paths.model_name=synthetic_000 training.steps=10 model.layers=1 +model.n_e_layers=1 +model.n_t_layers=1 +model.concept_size=2 +model.reduction_strategy="attn"
```

The `paths.model_name` flag specifies which subdirectory on disk (inside `/tmp`) to write model checkpoints to. You'll typically want to change this when starting a new model run.

## Acknowledgements

Thanks to the [MatX team](https://matx.com/) for their implementation of GPT in seqax which I used to implement LCM.

Thanks to the [Google TPU Research Cloud](https://sites.research.google/trc/about/), which has supported my investigations.
