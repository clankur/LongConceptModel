"""Main training loop, including the model, loss function, and optimizer."""

import dataclasses
from jax.tree_util import tree_leaves
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from clearml import Task
import training_io
from jax_extra import fold_in_str, explicit_activation_checkpointing, save_for_backward
import jax_extra
import einops
import shardlib.shardops as shardops
from shardlib.shardtypes import bf16, bool_, f32, pytree_dataclass, u32, make_shardings
from input_loader import (
    FlatTokensParams,
    HuggingFaceDataParams,
    SyntheticDataParams,
    TokenBatch,
    TokenBatchParams,
    get_loader,
)
import math
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax import Array, lax
import jax
from dataclasses import dataclass
from typeguard import typechecked
import hydra
from typing import Any, Optional, Tuple, Union
from functools import cached_property, partial
from collections import defaultdict
import datetime
import gcsfs  # Needed for clearml setup
import shardlib.shardtypes as shardtypes
import operator
import os
import time
import subprocess
import signal
from collections import namedtuple
import env

env.set_variables()

shardtypes.register_with_typeguard()


P = PartitionSpec

PRNGKey = Any

# TODO:
# 3 Stages
# depending on the stage mask is different
#   1. mask is all 1s for everything within chunk and previous chunks, shape (L, L)
#   2. mask is standard causal mask, shape (block_size, block_size)
#   3. mask is standard causal mask, shape (L, L)


# convert model to Transformer so we can have different implementations of it across the stages


@dataclass(frozen=True)
class BaseWidths:
    d_model: int
    n_q_per_kv: int
    n_kv: int
    d_head: int
    d_ff: int
    block_size: int


@dataclass(frozen=True)
class Hparams:
    d_model: int
    n_q_per_kv: int
    n_kv: int
    d_head: int
    vocab: int
    d_ff: int
    block_size: int
    layers: int
    n_e_layers: int  # Number of encoder layers
    n_t_layers: int  # Number of token decoder layers
    base: BaseWidths

    # fields for position embeddings
    rope_max_timescale: int

    # parameters for mup
    a_attn: float
    a_output: float
    zero_queries: bool
    zero_unembed: bool

    # parameters for exp scaling
    parameterization: str
    fully_aligned: bool
    gamma_embed: float
    gamma_hidden: float
    gamma_unembed: float
    reduction_strategy: str


def get_parameterization(style: str, fully_aligned: bool = True):
    Parameterization = namedtuple(
        "Parameterization",
        [
            "embed_init_var",
            "embed_param_mult",
            "embed_lr",
            "embed_grad",
            "hidden_init_var",
            "hidden_param_mult",
            "hidden_lr",
            "hidden_grad",
            "unembed_init_var",
            "unembed_param_mult",
            "unembed_lr",
            "unembed_grad",
        ],
    )

    base_params = {
        "sp": Parameterization(
            embed_init_var=0.0,
            embed_param_mult=0.0,
            embed_lr=0.0,
            embed_grad=0.5,
            hidden_init_var=1.0,
            hidden_param_mult=0.0,
            hidden_lr=1.0,
            hidden_grad=0.5,
            unembed_init_var=1.0,
            unembed_param_mult=0.0,
            unembed_lr=1.0,
            unembed_grad=0.0,
        ),
        "mup": Parameterization(
            embed_init_var=1.0,
            embed_param_mult=0.5,
            embed_lr=0.5,
            embed_grad=0.5,
            hidden_init_var=1.0,
            hidden_param_mult=0.0,
            hidden_lr=1.0,
            hidden_grad=1.0,
            unembed_init_var=1.0,
            unembed_param_mult=0.5,
            unembed_lr=0.5,
            unembed_grad=0.5,
        ),
        "ntk": Parameterization(
            embed_init_var=0.0,
            embed_param_mult=0.0,
            embed_lr=0.0,
            embed_grad=0.5,
            hidden_init_var=0.0,
            hidden_param_mult=0.5,
            hidden_lr=0.5,
            hidden_grad=1.0,
            unembed_init_var=0.0,
            unembed_param_mult=0.5,
            unembed_lr=0.5,
            unembed_grad=0.5,
        ),
        "mean-field": Parameterization(
            embed_init_var=0.0,
            embed_param_mult=0.0,
            embed_lr=0.0,
            embed_grad=1.0,
            hidden_init_var=0.0,
            hidden_param_mult=0.5,
            hidden_lr=0.5,
            hidden_grad=1.5,
            unembed_init_var=0.0,
            unembed_param_mult=1.0,
            unembed_lr=0.0,
            unembed_grad=1.0,
        ),
    }

    style = style.lower()
    if style not in base_params:
        raise ValueError(f"Unknown parameterization style: {style}")

    params = base_params[style]._asdict()

    if not fully_aligned:
        if style == "sp":
            params.update(
                {
                    "embed_lr": 0.0,
                    "hidden_lr": 0.5,
                    "unembed_lr": 0.5,
                }
            )
        elif style == "mup":
            params.update(
                {
                    "embed_lr": 0.5,
                    "hidden_lr": 0.5,
                    "unembed_lr": 0.0,
                }
            )
        elif style == "ntk":
            params.update(
                {
                    "embed_lr": 0.0,
                    "hidden_lr": 0.0,
                    "unembed_lr": 0.0,
                }
            )
        elif style == "mean-field":
            params.update(
                {
                    "embed_lr": 0.0,
                    "hidden_lr": 0.0,
                    "unembed_lr": -0.5,
                }
            )

    return Parameterization(**params)


@pytree_dataclass
class SyntheticMetrics:
    avg_confidence: f32[b""]
    avg_char_confidence: f32[b""]
    max_char_confidence: f32[b""]
    avg_start_char_confidence: f32[b""]
    avg_final_char_confidence: f32[b""]


@pytree_dataclass
class Model:
    embed: f32["vocab/t d_model/d"]
    unembed: f32["vocab/t d_model/d"]
    ln1: f32["layers d_model/t/d"]
    ln2: f32["layers d_model/t/d"]
    w_q: f32["layers d_model/d n_q_per_kv n_kv/t d_head"]
    w_kv: f32["layers 2 d_model/d n_kv/t d_head"]
    w_o: f32["layers d_model/d n_q_per_kv n_kv/t d_head"]
    w_gate: f32["layers d_model/d d_ff/t"]
    w_up: f32["layers d_model/d d_ff/t"]
    w_down: f32["layers d_model/d d_ff/t"]
    final_layer_norm: f32["d_model/d/t"]

    # New encoder weights
    e_ln1: f32["n_e_layers d_model/t/d"]
    e_ln2: f32["n_e_layers d_model/t/d"]
    e_w_q: f32["n_e_layers d_model/d n_q_per_kv n_kv/t d_head"]
    e_w_kv: f32["n_e_layers 2 d_model/d n_kv/t d_head"]
    e_w_o: f32["n_e_layers d_model/d n_q_per_kv n_kv/t d_head"]
    e_w_gate: f32["n_e_layers d_model/d d_ff/t"]
    e_w_up: f32["n_e_layers d_model/d d_ff/t"]
    e_w_down: f32["n_e_layers d_model/d d_ff/t"]

    # New token decoder weights
    t_ln1: f32["n_t_layers d_model/t/d"]
    t_ln2: f32["n_t_layers d_model/t/d"]
    t_w_q: f32["n_t_layers d_model/d n_q_per_kv n_kv/t d_head"]
    t_w_kv: f32["n_t_layers 2 d_model/d n_kv/t d_head"]
    t_w_o: f32["n_t_layers d_model/d n_q_per_kv n_kv/t d_head"]
    t_w_gate: f32["n_t_layers d_model/d d_ff/t"]
    t_w_up: f32["n_t_layers d_model/d d_ff/t"]
    t_w_down: f32["n_t_layers d_model/d d_ff/t"]

    # Cross attention weights for token decoder
    x_w_q: f32[
        "n_t_layers d_model/d n_q_per_kv n_kv/t d_head"
    ]  # Query weights for cross attention
    x_w_kv: f32[
        "n_t_layers 2 d_model/d n_kv/t d_head"
    ]  # Key/value weights for cross attention
    x_w_o: f32[
        "n_t_layers d_model/d n_q_per_kv n_kv/t d_head"
    ]  # Output weights for cross attention
    x_lnx: f32["n_t_layers d_model/t/d"]
    x_lnz: f32["n_t_layers d_model/t/d"]

    w_mix: f32["block_size 1"]  # Separate tensor for weighted sum reduction
    w_reduce_q: f32["1 d_model/t/d"]  # Modified shape for direct query tensor
    w_reduce_kv: f32["2 block_size block_size"]

    @staticmethod
    @typechecked
    def init(h: Hparams, rng: PRNGKey) -> "Model":
        # https://github.com/google/jax/issues/20390 for ones_like with sharding.
        ln1 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln2 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        final_layer_norm = jnp.ones((h.d_model,), dtype=jnp.float32)

        # All of wi/wq/wo/wo/w_kv use truncated_normal initializers with 'fan_in' scaling,
        # i.e. variance set to 1.0/fan_in.
        # The constant is stddev of standard normal truncated to (-2, 2)
        truncated_normal_stddev = 0.87962566103423978
        p = get_parameterization(h.parameterization)
        base = h.base

        embed_scale = (
            math.sqrt(base.d_model) / (h.d_model * truncated_normal_stddev)
        ) ** (p.embed_init_var)
        # scale for tensors with d_model fan_in and truncated normal truncated to (-2, 2)
        d_model_scale = (
            math.sqrt(base.d_model) / (h.d_model * truncated_normal_stddev)
        ) ** (p.hidden_init_var)
        w_kv_scale = d_model_scale
        target_head_dim = h.n_q_per_kv * h.n_kv * h.d_head
        base_head_dim = base.n_q_per_kv * base.n_kv * base.d_head
        w_o_scale = (
            math.sqrt(base_head_dim) / (target_head_dim * truncated_normal_stddev)
        ) ** (p.hidden_init_var)
        w_up_scale = d_model_scale
        w_down_scale = (math.sqrt(base.d_ff) / (h.d_ff * truncated_normal_stddev)) ** (
            p.hidden_init_var
        )

        w_kv_shape = (h.layers, 2, h.d_model, h.n_kv, h.d_head)
        w_kv = w_kv_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_kv"), -2, 2, w_kv_shape, dtype=jnp.float32
        )

        ff_shape = (h.layers, h.d_model, h.d_ff)
        embed = embed_scale * jax.random.normal(
            jax_extra.fold_in_str(rng, "embed"), (h.vocab, h.d_model), dtype=jnp.float32
        )
        w_gate = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_gate"), -2, 2, ff_shape, dtype=jnp.float32
        )
        w_up = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_up"), -2, 2, ff_shape, dtype=jnp.float32
        )
        w_down = w_down_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_down"), -2, 2, ff_shape, dtype=jnp.float32
        )

        w_q_scale = d_model_scale
        w_q_shape = (h.layers, h.d_model, h.n_q_per_kv, h.n_kv, h.d_head)
        w_o_shape = w_q_shape
        unembed_scale = (
            math.sqrt(base.d_model) / (h.d_model * truncated_normal_stddev)
        ) ** (p.unembed_init_var)
        w_o = w_o_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_o"), -2, 2, w_o_shape, dtype=jnp.float32
        )

        if h.zero_queries:
            w_q = jnp.zeros(w_q_shape, dtype=jnp.float32)
        else:
            w_q = w_q_scale * jax.random.truncated_normal(
                fold_in_str(rng, "w_q"), -2, 2, w_q_shape, dtype=jnp.float32
            )

        if h.zero_unembed:
            unembed = jnp.zeros((h.vocab, h.d_model), dtype=jnp.float32)
        else:
            unembed = unembed_scale * jax.random.truncated_normal(
                fold_in_str(rng, "unembed"),
                -2,
                2,
                (h.vocab, h.d_model),
                dtype=jnp.float32,
            )

        # Initialize encoder weights with layers dimension
        e_ln1 = jnp.ones((h.n_e_layers, h.d_model), dtype=jnp.float32)
        e_ln2 = jnp.ones((h.n_e_layers, h.d_model), dtype=jnp.float32)

        e_w_q_shape = (h.n_e_layers, h.d_model, h.n_q_per_kv, h.n_kv, h.d_head)
        e_w_kv_shape = (h.n_e_layers, 2, h.d_model, h.n_kv, h.d_head)
        e_ff_shape = (h.n_e_layers, h.d_model, h.d_ff)

        e_w_q = w_q_scale * jax.random.truncated_normal(
            fold_in_str(rng, "e_w_q"), -2, 2, e_w_q_shape, dtype=jnp.float32
        )
        e_w_kv = w_kv_scale * jax.random.truncated_normal(
            fold_in_str(rng, "e_w_kv"), -2, 2, e_w_kv_shape, dtype=jnp.float32
        )
        e_w_o = w_o_scale * jax.random.truncated_normal(
            fold_in_str(rng, "e_w_o"), -2, 2, e_w_q_shape, dtype=jnp.float32
        )
        e_w_gate = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "e_w_gate"), -2, 2, e_ff_shape, dtype=jnp.float32
        )
        e_w_up = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "e_w_up"), -2, 2, e_ff_shape, dtype=jnp.float32
        )
        e_w_down = w_down_scale * jax.random.truncated_normal(
            fold_in_str(rng, "e_w_down"), -2, 2, e_ff_shape, dtype=jnp.float32
        )

        # Initialize token decoder weights with n_t_layers dimension
        t_ln1 = jnp.ones((h.n_t_layers, h.d_model), dtype=jnp.float32)
        t_ln2 = jnp.ones((h.n_t_layers, h.d_model), dtype=jnp.float32)

        t_w_q_shape = (h.n_t_layers, h.d_model, h.n_q_per_kv, h.n_kv, h.d_head)
        t_w_kv_shape = (h.n_t_layers, 2, h.d_model, h.n_kv, h.d_head)
        t_ff_shape = (h.n_t_layers, h.d_model, h.d_ff)

        t_w_q = w_q_scale * jax.random.truncated_normal(
            fold_in_str(rng, "t_w_q"), -2, 2, t_w_q_shape, dtype=jnp.float32
        )
        t_w_kv = w_kv_scale * jax.random.truncated_normal(
            fold_in_str(rng, "t_w_kv"), -2, 2, t_w_kv_shape, dtype=jnp.float32
        )
        t_w_o = w_o_scale * jax.random.truncated_normal(
            fold_in_str(rng, "t_w_o"), -2, 2, t_w_q_shape, dtype=jnp.float32
        )
        t_w_gate = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "t_w_gate"), -2, 2, t_ff_shape, dtype=jnp.float32
        )
        t_w_up = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "t_w_up"), -2, 2, t_ff_shape, dtype=jnp.float32
        )
        t_w_down = w_down_scale * jax.random.truncated_normal(
            fold_in_str(rng, "t_w_down"), -2, 2, t_ff_shape, dtype=jnp.float32
        )

        # Initialize cross attention weights for token decoder
        x_w_q = w_q_scale * jax.random.truncated_normal(
            fold_in_str(rng, "x_w_q"), -2, 2, t_w_q_shape, dtype=jnp.float32
        )
        x_w_kv = w_kv_scale * jax.random.truncated_normal(
            fold_in_str(rng, "x_w_kv"), -2, 2, t_w_kv_shape, dtype=jnp.float32
        )
        x_w_o = w_o_scale * jax.random.truncated_normal(
            fold_in_str(rng, "x_w_o"), -2, 2, t_w_q_shape, dtype=jnp.float32
        )

        # Initialize cross attention layer norms with n_t_layers dimension
        x_lnx = jnp.ones((h.n_t_layers, h.d_model), dtype=jnp.float32)
        x_lnz = jnp.ones((h.n_t_layers, h.d_model), dtype=jnp.float32)

        # Initialize w_mix for weighted sum reduction
        block_size_scale = (h.block_size / h.base.block_size) ** -p.hidden_param_mult
        w_mix = block_size_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_mix"), -2, 2, (h.block_size, 1), dtype=jnp.float32
        )

        # Initialize w_reduce_q as a learned query vector
        w_reduce_q = jax.random.truncated_normal(
            fold_in_str(rng, "w_reduce_q"), -2, 2, (1, h.d_model), dtype=jnp.float32
        )

        # Initialize reduction weights for k/v
        w_reduce_kv = block_size_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_reduce_kv"),
            -2,
            2,
            (2, h.block_size, h.block_size),
            dtype=jnp.float32,
        )

        arrays = Model(
            embed=embed,
            unembed=unembed,
            ln1=ln1,
            ln2=ln2,
            w_q=w_q,
            w_kv=w_kv,
            w_o=w_o,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            final_layer_norm=final_layer_norm,
            e_ln1=e_ln1,
            e_ln2=e_ln2,
            e_w_q=e_w_q,
            e_w_kv=e_w_kv,
            e_w_o=e_w_o,
            e_w_gate=e_w_gate,
            e_w_up=e_w_up,
            e_w_down=e_w_down,
            t_ln1=t_ln1,
            t_ln2=t_ln2,
            t_w_q=t_w_q,
            t_w_kv=t_w_kv,
            t_w_o=t_w_o,
            t_w_gate=t_w_gate,
            t_w_up=t_w_up,
            t_w_down=t_w_down,
            x_w_q=x_w_q,
            x_w_kv=x_w_kv,
            x_w_o=x_w_o,
            x_lnx=x_lnx,
            x_lnz=x_lnz,
            w_mix=w_mix,
            w_reduce_q=w_reduce_q,
            w_reduce_kv=w_reduce_kv,
        )
        shardings = make_shardings(Model)
        return jax.tree.map(lax.with_sharding_constraint, arrays, shardings)

    @typechecked
    def forward_pass(
        self, h: Hparams, ids: u32[b"B/d L"], is_seq_start: bool_[b"B/d L"]
    ) -> f32[b"B/d L V/t"]:
        p = get_parameterization(h.parameterization)
        embed_mult = (h.d_model / h.base.d_model) ** -p.embed_param_mult
        hidden_mult = (h.d_model / h.base.d_model) ** -p.hidden_param_mult
        unembed_mult = (h.d_model / h.base.d_model) ** -p.unembed_param_mult

        # Initial embedding lookup.
        embed = embed_mult * shardops.all_gather(
            "V/t M/d -> V/t M", jnp.bfloat16(self.embed)
        )
        one_hot_ids = jax.nn.one_hot(ids, self.embed.shape[0])
        x = shardops.einsum_unreduced("B/d L V/t, V/t M -> B/d L M", one_hot_ids, embed)

        embed_x = x = shardops.psum_scatter("B/d L M -> B/d L M/t", x)

        L = ids.shape[1]
        n_blocks = L // h.block_size
        segment_ids = jnp.cumsum(is_seq_start, axis=1)
        segment_mask: bool_[b"B/d L L"] = (
            segment_ids[:, :, jnp.newaxis] == segment_ids[:, jnp.newaxis, :]
        )
        segment_mask: bool_[b"B/d L L 1 1"] = segment_mask[
            ..., jnp.newaxis, jnp.newaxis
        ]  # add axes for q_per_k, num_kv_heads dimensions
        causal_mask: bool_[b"1 L L 1 1"] = jnp.tril(
            jnp.ones((L, L), dtype=jnp.bool_), 0
        )[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]
        causal_mask: bool_[b"B/d L L 1 1"] = jnp.logical_and(segment_mask, causal_mask)
        chunk_indices = jnp.arange(L) // h.block_size
        encoder_mask = chunk_indices[:, None] > chunk_indices[None, :]
        encoder_mask = encoder_mask[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]
        concept_causal_mask = jnp.tril(jnp.ones((n_blocks, n_blocks), dtype=jnp.bool_))[
            jnp.newaxis, ..., jnp.newaxis, jnp.newaxis
        ]
        q_pos = jnp.arange(L)
        k_pos = jnp.arange(L // h.block_size)
        x_causal_mask = q_pos[:, None] // h.block_size >= k_pos[None, :]
        x_causal_mask = x_causal_mask[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]

        rope_table = RopeTable.create(L, h, pos_mult=h.block_size)
        concept_rope_table = RopeTable.create(n_blocks, h)

        # Encoder block
        #  need weights for e_w_q, e_w_kv, e_w_o, e_w_gate, e_w_up, e_w_down, e_ln1, e_ln2
        #  Encodes chunks of the input into a concept embedding
        #     - Input: chunks of the input, Mask enabling attending to tokens within the same chunk AND to previous chunks
        #     - perform standard attention on the chunks with this mask
        #     - reduce for each chunk BLOCK_SIZE to a single embedding

        @explicit_activation_checkpointing
        @typechecked
        def encoder_block(
            x: bf16[b"B/d L M/t"], layer_weights: Any
        ) -> Tuple[bf16[b"B/d L M/t"], Tuple[()]]:
            w_q, w_kv, w_o, w_gate, w_up, w_down, ln1, ln2 = layer_weights

            # Pre-attention RMSNorm
            ln1 = shardops.all_gather("M/t/d -> M", jnp.float32(ln1))
            gx = shardops.all_gather("B/d L M/t -> B/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln1)

            # Attention, using Grouped Query Attention and RoPE position embeddings.
            w_q = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(w_q))
            q = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced(
                    "B/d L M, M Q K/t D -> B/d L Q K/t D", nx, w_q
                )
            )
            q = rope_table.apply("L D -> 1 L 1 1 D", q)
            w_kv = shardops.all_gather("2 M/d K/t D -> 2 M K/t D", jnp.bfloat16(w_kv))
            k, v = hidden_mult * shardops.einsum_unreduced(
                "B/d L M, k_v M K/t D -> k_v B/d L K/t D", nx, w_kv
            )
            k = save_for_backward(k)
            v = save_for_backward(v)
            k = rope_table.apply("L d -> 1 L 1 d", k)

            logit_scale = jax.lax.select(
                h.parameterization.lower() == "mup",
                h.a_attn * math.sqrt(h.base.d_head) / h.d_head,
                1.0 / math.sqrt(h.d_head),
            )
            logits = logit_scale * shardops.einsum_unreduced(
                "B/d Qlen Q K/t D, B/d Klen K/t D -> B/d Qlen Klen Q K/t",
                q,
                k,
                preferred_element_type=jnp.float32,
            )

            logits = jnp.where(encoder_mask, logits, -1e10)
            probs = jnp.bfloat16(jax.nn.softmax(logits, axis=2))
            attn_out = shardops.einsum_unreduced(
                "B/d Qlen Klen Q K/t, B/d Klen K/t D -> B/d Qlen Q K/t D",
                probs,
                v,
            )
            w_o = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(w_o))
            attn_out = hidden_mult * shardops.einsum_unreduced(
                "B/d Qlen Q K/t D, M Q K/t D -> B/d Qlen M", attn_out, w_o
            )
            attn_out = shardops.psum_scatter("B/d Qlen M -> B/d Qlen M/t", attn_out)
            x = save_for_backward(x + attn_out)

            # Pre-FFN RMSNorm
            ln2 = save_for_backward(shardops.all_gather("M/t/d -> M", jnp.float32(ln2)))
            gx = shardops.all_gather("B/d L M/t -> B/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln2)

            # FFN, using SwiGLU
            w_gate = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(w_gate))
            gate_proj = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced("B/d L M, M F/t -> B/d L F/t", nx, w_gate)
            )
            w_up = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(w_up))
            up_proj = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced("B/d L M, M F/t -> B/d L F/t", nx, w_up)
            )
            y = jax.nn.swish(gate_proj) * up_proj
            w_down = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(w_down))

            ffn_out_mult = (h.d_ff / h.base.d_ff) ** -p.hidden_param_mult
            ffn_out = ffn_out_mult * shardops.einsum_unreduced(
                "B/d L F/t, M F/t -> B/d L M", y, w_down
            )
            ffn_out = shardops.psum_scatter("B/d L M -> B/d L M/t", ffn_out)
            x = x + ffn_out

            return jnp.bfloat16(x), ()

        # Concept decoder block
        @explicit_activation_checkpointing
        @typechecked
        def concept_decoder_block(
            x: bf16[b"B/d n_blocks M/t"], layer_weights: Any
        ) -> Tuple[bf16[b"B/d n_blocks M/t"], Tuple[()]]:
            w_q, w_kv, w_o, w_gate, w_up, w_down, ln1, ln2 = layer_weights

            # Pre-attention RMSNorm
            ln1 = shardops.all_gather("M/t/d -> M", jnp.float32(ln1))
            gx = shardops.all_gather("B/d n_blocks M/t -> B/d n_blocks M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln1)

            # Standard decoder attention with causal mask for concept embeddings
            w_q = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(w_q))
            q = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced(
                    "B/d n_blocks M, M Q K/t D -> B/d n_blocks Q K/t D", nx, w_q
                )
            )
            q = concept_rope_table.apply("n_blocks D -> 1 n_blocks 1 1 D", q)
            w_kv = shardops.all_gather("2 M/d K/t D -> 2 M K/t D", jnp.bfloat16(w_kv))
            k, v = hidden_mult * shardops.einsum_unreduced(
                "B/d n_blocks M, k_v M K/t D -> k_v B/d n_blocks K/t D", nx, w_kv
            )
            k = save_for_backward(k)
            v = save_for_backward(v)
            k = concept_rope_table.apply("n_blocks d -> 1 n_blocks 1 d", k)

            logit_scale = jax.lax.select(
                h.parameterization.lower() == "mup",
                h.a_attn * math.sqrt(h.base.d_head) / h.d_head,
                1.0 / math.sqrt(h.d_head),
            )
            logits = logit_scale * shardops.einsum_unreduced(
                "B/d Qblocks Q K/t D, B/d Kblocks K/t D -> B/d Qblocks Kblocks Q K/t",
                q,
                k,
                preferred_element_type=jnp.float32,
            )

            logits = jnp.where(concept_causal_mask, logits, -1e10)
            probs = jnp.bfloat16(jax.nn.softmax(logits, axis=2))
            attn_out = shardops.einsum_unreduced(
                "B/d Qblocks Kblocks Q K/t, B/d Kblocks K/t D -> B/d Qblocks Q K/t D",
                probs,
                v,
            )
            w_o = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(w_o))
            attn_out = hidden_mult * shardops.einsum_unreduced(
                "B/d Qblocks Q K/t D, M Q K/t D -> B/d Qblocks M", attn_out, w_o
            )
            attn_out = shardops.psum_scatter(
                "B/d Qblocks M -> B/d Qblocks M/t", attn_out
            )
            x = save_for_backward(x + attn_out)

            # Pre-FFN RMSNorm
            ln2 = save_for_backward(shardops.all_gather("M/t/d -> M", jnp.float32(ln2)))
            gx = shardops.all_gather("B/d n_blocks M/t -> B/d n_blocks M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln2)

            # FFN, using SwiGLU
            w_gate = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(w_gate))
            gate_proj = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced(
                    "B/d n_blocks M, M F/t -> B/d n_blocks F/t", nx, w_gate
                )
            )
            w_up = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(w_up))
            up_proj = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced(
                    "B/d n_blocks M, M F/t -> B/d n_blocks F/t", nx, w_up
                )
            )
            y = jax.nn.swish(gate_proj) * up_proj
            w_down = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(w_down))

            ffn_out_mult = (h.d_ff / h.base.d_ff) ** -p.hidden_param_mult
            ffn_out = ffn_out_mult * shardops.einsum_unreduced(
                "B/d n_blocks F/t, M F/t -> B/d n_blocks M", y, w_down
            )
            ffn_out = shardops.psum_scatter(
                "B/d n_blocks M -> B/d n_blocks M/t", ffn_out
            )

            return jnp.bfloat16(x + ffn_out), ()

        # Token decoder
        #  3. Token Decoder that decodes concept embedding to get the output tokens
        #     - Input: output concept embedding (z) from CausalEmbedding, Mask enabling attending to previous tokens
        #     - get tokenized embeddings, apply standard attention on them, get out, use x = x + out
        #     - apply cross attention
        #           - apply x_wq to x to get queries, apply x_wkv to z get keys and values
        #           - standard decoder after this with MLP
        #     - each embedding gets mapped back to a BLOCK_SIZE and we join them to form the sequence?

        # Token decoder block
        @explicit_activation_checkpointing
        @typechecked
        def token_decoder_block(
            carry: Tuple[bf16[b"B/d L M/t"], bf16[b"B/d n_blocks M/t"]],
            layer_weights: Any,
        ) -> Tuple[Tuple[bf16[b"B/d L M/t"], bf16[b"B/d n_blocks M/t"]], Tuple[()]]:
            x, z = carry
            (
                w_q,
                w_kv,
                w_o,
                w_gate,
                w_up,
                w_down,
                ln1,
                ln2,
                x_w_q,
                x_w_kv,
                x_w_o,
                x_lnx,
                x_lnz,
            ) = layer_weights

            # Pre-attention RMSNorm
            ln1 = shardops.all_gather("M/t/d -> M", jnp.float32(ln1))
            gx = shardops.all_gather("B/d L M/t -> B/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln1)

            # Self attention with causal mask
            w_q = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(w_q))
            q = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced(
                    "B/d L M, M Q K/t D -> B/d L Q K/t D", nx, w_q
                )
            )
            q = rope_table.apply("L D -> 1 L 1 1 D", q)
            w_kv = shardops.all_gather("2 M/d K/t D -> 2 M K/t D", jnp.bfloat16(w_kv))
            k, v = hidden_mult * shardops.einsum_unreduced(
                "B/d L M, k_v M K/t D -> k_v B/d L K/t D", nx, w_kv
            )
            k = save_for_backward(k)
            v = save_for_backward(v)
            k = rope_table.apply("L d -> 1 L 1 d", k)

            logit_scale = jax.lax.select(
                h.parameterization.lower() == "mup",
                h.a_attn * math.sqrt(h.base.d_head) / h.d_head,
                1.0 / math.sqrt(h.d_head),
            )
            logits = logit_scale * shardops.einsum_unreduced(
                "B/d Qlen Q K/t D, B/d Klen K/t D -> B/d Qlen Klen Q K/t",
                q,
                k,
                preferred_element_type=jnp.float32,
            )

            logits = jnp.where(causal_mask, logits, -1e10)
            probs = jnp.bfloat16(jax.nn.softmax(logits, axis=2))
            attn_out = shardops.einsum_unreduced(
                "B/d Qlen Klen Q K/t, B/d Klen K/t D -> B/d Qlen Q K/t D", probs, v
            )
            w_o = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(w_o))
            attn_out = hidden_mult * shardops.einsum_unreduced(
                "B/d L Q K/t D, M Q K/t D -> B/d L M", attn_out, w_o
            )
            attn_out = shardops.psum_scatter("B/d L M -> B/d L M/t", attn_out)
            x = save_for_backward(x + attn_out)

            # Cross attention with concept embeddings z
            # Apply layer norms before cross attention
            x_lnx = shardops.all_gather("M/t/d -> M", jnp.float32(x_lnx))
            x_lnz = shardops.all_gather("M/t/d -> M", jnp.float32(x_lnz))

            nx = jnp.bfloat16(rms_norm(gx) * x_lnx)  # Normalize token decoder input
            gz = shardops.all_gather("B/d n_blocks M/t -> B/d n_blocks M", z)

            nz = jnp.bfloat16(rms_norm(gz) * x_lnz)  # Normalize concept embeddings

            x_w_q = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(x_w_q))
            q = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced(
                    "B/d L M, M Q K/t D -> B/d L Q K/t D", nx, x_w_q
                )
            )
            q = rope_table.apply("L D -> 1 L 1 1 D", q)

            x_w_kv = shardops.all_gather(
                "2 M/d K/t D -> 2 M K/t D", jnp.bfloat16(x_w_kv)
            )
            k, v = hidden_mult * shardops.einsum_unreduced(
                "B/d n_blocks M, k_v M K/t D -> k_v B/d n_blocks K/t D", nz, x_w_kv
            )
            k = concept_rope_table.apply("n_blocks d -> 1 n_blocks 1 d", k)
            k = save_for_backward(k)
            v = save_for_backward(v)

            logits = logit_scale * shardops.einsum_unreduced(
                "B/d L Q K/t D, B/d n_blocks K/t D -> B/d L n_blocks Q K/t",
                q,
                k,
                preferred_element_type=jnp.float32,
            )
            logits = jnp.where(x_causal_mask, logits, -1e10)

            # need a mask of size L x n_blocks
            probs = jnp.bfloat16(jax.nn.softmax(logits, axis=2))
            attn_out = shardops.einsum_unreduced(
                "B/d L n_blocks Q K/t, B/d n_blocks K/t D -> B/d L Q K/t D",
                probs,
                v,
            )
            x_w_o = shardops.all_gather("M/d Q K/t D -> M Q K/t D", jnp.bfloat16(x_w_o))
            attn_out = hidden_mult * shardops.einsum_unreduced(
                "B/d L Q K/t D, M Q K/t D -> B/d L M", attn_out, x_w_o
            )
            attn_out = shardops.psum_scatter("B/d L M -> B/d L M/t", attn_out)
            x = save_for_backward(x + attn_out)

            # Pre-FFN RMSNorm
            ln2 = save_for_backward(shardops.all_gather("M/t/d -> M", jnp.float32(ln2)))
            gx = shardops.all_gather("B/d L M/t -> B/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln2)

            # FFN, using SwiGLU
            w_gate = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(w_gate))
            gate_proj = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced("B/d L M, M F/t -> B/d L F/t", nx, w_gate)
            )
            w_up = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(w_up))
            up_proj = save_for_backward(
                hidden_mult
                * shardops.einsum_unreduced("B/d L M, M F/t -> B/d L F/t", nx, w_up)
            )
            y = jax.nn.swish(gate_proj) * up_proj
            w_down = shardops.all_gather("M/d F/t -> M F/t", jnp.bfloat16(w_down))

            ffn_out_mult = (h.d_ff / h.base.d_ff) ** -p.hidden_param_mult
            ffn_out = ffn_out_mult * shardops.einsum_unreduced(
                "B/d L F/t, M F/t -> B/d L M", y, w_down
            )
            ffn_out = shardops.psum_scatter("B/d L M -> B/d L M/t", ffn_out)

            return (jnp.bfloat16(x + ffn_out), z), ()

        # Process input through encoder blocks
        x, () = jax.lax.scan(
            encoder_block,
            jnp.bfloat16(x),
            (
                self.e_w_q,
                self.e_w_kv,
                self.e_w_o,
                self.e_w_gate,
                self.e_w_up,
                self.e_w_down,
                self.e_ln1,
                self.e_ln2,
            ),
        )

        # reduce for each chunk of block_size to a single embedding
        x = einops.rearrange(
            x,
            "B (n_blocks block_size) M -> B n_blocks block_size M",
            n_blocks=n_blocks,
        )
        if h.reduction_strategy == "sum":
            x = einops.reduce(x, "B n_blocks block_size M -> B n_blocks M", "sum")
        elif h.reduction_strategy == "max":
            x = einops.reduce(x, "B n_blocks block_size M -> B n_blocks M", "max")
        elif h.reduction_strategy == "wei_sum":
            x = shardops.einsum_unreduced(
                "B/d n_blocks block_size M/t, block_size 1 -> B/d n_blocks M/t",
                x,
                self.w_mix,
            )
        elif h.reduction_strategy == "attn":
            w_reduce_q = shardops.all_gather("1 M/t/d -> 1 M/t", self.w_reduce_q)

            reduce_k, reduce_v = hidden_mult * shardops.einsum_unreduced(
                "B/d n_blocks block_size M/t, k_v block_size b_size -> k_v B/d n_blocks b_size M/t",
                x,
                self.w_reduce_kv,
            )
            print(reduce_k.shape)
            logits = shardops.einsum_unreduced(
                "1 M/t, B/d n_blocks block_size M/t -> B/d n_blocks 1 block_size M/t",
                w_reduce_q,
                reduce_k,
                preferred_element_type=jnp.float32,
            )

            attn_weights = jnp.bfloat16(jax.nn.softmax(logits, axis=-1))

            x = shardops.einsum_unreduced(
                "B/d n_blocks 1 block_size M/t, B/d n_blocks block_size M/t -> B/d n_blocks M/t",
                attn_weights,
                reduce_v,
            )
        elif h.reduction_strategy == "cnn":
            pass

        # Process through concept decoder blocks
        x, () = jax.lax.scan(
            concept_decoder_block,
            jnp.bfloat16(x),
            (
                self.w_q,
                self.w_kv,
                self.w_o,
                self.w_gate,
                self.w_up,
                self.w_down,
                self.ln1,
                self.ln2,
            ),
        )

        # Process through token decoder blocks
        (x, _), () = jax.lax.scan(
            token_decoder_block,
            (
                jnp.bfloat16(embed_x),
                jnp.bfloat16(x),
            ),  # Pass both token embeddings and concept embeddings
            (
                self.t_w_q,
                self.t_w_kv,
                self.t_w_o,
                self.t_w_gate,
                self.t_w_up,
                self.t_w_down,
                self.t_ln1,
                self.t_ln2,
                self.x_w_q,
                self.x_w_kv,
                self.x_w_o,
                self.x_lnx,
                self.x_lnz,
            ),
        )

        # Final layernorm and output projection.
        x = shardops.all_gather("B/d L M/t -> B/d L M", x)
        ln = shardops.all_gather("M/t/d -> M", jnp.float32(self.final_layer_norm))
        x = jnp.bfloat16(rms_norm(x) * ln)
        unembed = unembed_mult * shardops.all_gather(
            "V/t M/d -> V/t M", jnp.bfloat16(self.unembed)
        )
        logits = shardops.einsum_unreduced(
            "B/d L M, V/t M -> B/d L V/t",
            x,
            unembed,
            preferred_element_type=jnp.float32,
        )

        return logits

    @typechecked
    def loss(self, h: Hparams, batch: TokenBatch) -> Tuple[f32[b""], SyntheticMetrics]:
        # Given sequence-packed targets:
        #   [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        # we want inputs:
        #   [[0, 1], [0, 3, 4], [0, 6, 7, 8]]
        # which we get by shifting the targets right by 1 and
        # masking sequence-start tokens to 0.
        inputs = jnp.pad(batch.targets[:, :-1], pad_width=((0, 0), (1, 0)))
        is_seq_start: bool_[b"batch/d len"] = batch.is_seq_start
        inputs: u32[b"batch/d len"] = jnp.where(is_seq_start, 0, inputs)

        logits: f32[b"batch/d len V/t"] = self.forward_pass(h, inputs, is_seq_start)
        max_logits: f32[b"batch/d len 1"] = lax.pmax(
            jnp.max(lax.stop_gradient(logits), axis=-1, keepdims=True), "t"
        )
        logits = logits - max_logits
        sum_logits = lax.psum(jnp.sum(jnp.exp(logits), axis=-1, keepdims=True), "t")
        logsumexp = jnp.log(sum_logits)
        logprobs: f32[b"batch/d len V/t"] = logits - logsumexp
        logprobs_at_targets = shardops.index_unreduced(
            "batch/d len [V/t], batch/d len -> batch/d len", logprobs, batch.targets
        )
        logprobs_at_targets = shardops.psum_scatter(
            "batch/d len -> batch/d len/t", logprobs_at_targets
        )
        if batch.loss_masks is not None:
            logprobs_at_targets = jnp.where(batch.loss_masks, logprobs_at_targets, 0)
        tokens_in_global_batch = logprobs_at_targets.size * jax.lax.psum(1, ("d", "t"))

        probs_at_targets = jnp.exp(logprobs_at_targets)

        batch_size, length = probs_at_targets.shape

        if batch.comment_starts is not None and batch.comment_ends is not None:
            comment_starts: u32[b"batch/d n_print"] = batch.comment_starts
            comment_ends: u32[b"batch/d n_print"] = batch.comment_ends

            batch_indices = jnp.arange(batch_size)[:, jnp.newaxis]  # (batch, 1)
            start_char_probs = probs_at_targets[batch_indices, comment_starts]
            avg_start_char_probs: f32[b""] = jnp.mean(start_char_probs)
            last_char_probs = probs_at_targets[batch_indices, comment_ends - 1]
            avg_last_char_probs: f32[b""] = jnp.mean(last_char_probs)

            comment_mask = jax.vmap(
                lambda starts_row, ends_row: jax.vmap(
                    lambda start, end: (jnp.arange(length) >= start)
                    & (jnp.arange(length) < end)
                )(starts_row, ends_row)
            )(comment_starts, comment_ends)

            probs_at_targets = probs_at_targets[:, jnp.newaxis, :]

            p_answer = jnp.prod(jnp.where(comment_mask, probs_at_targets, 1), axis=-1)

            # average confidence for each prints in sequence
            avg_p_answer: f32[b""] = jnp.mean(p_answer)

            total_tokens = jnp.sum(comment_ends - comment_starts + 1)
            comment_probs = jnp.where(comment_mask, probs_at_targets, 0)
            average_char_confidence = jnp.sum(comment_probs) / total_tokens
            max_char_confidence = jnp.max(comment_probs)

            synth_metrics = SyntheticMetrics(
                avg_confidence=avg_p_answer,
                max_char_confidence=max_char_confidence,
                avg_char_confidence=average_char_confidence,
                avg_start_char_confidence=avg_start_char_probs,
                avg_final_char_confidence=avg_last_char_probs,
            )
        else:
            synth_metrics = SyntheticMetrics(
                avg_confidence=jnp.float32(0.0),
                max_char_confidence=jnp.float32(0.0),
                avg_char_confidence=jnp.float32(0.0),
                avg_start_char_confidence=jnp.float32(0.0),
                avg_final_char_confidence=jnp.float32(0.0),
            )

        return (
            -jnp.sum(logprobs_at_targets) / jnp.float32(tokens_in_global_batch),
            synth_metrics,
        )


@pytree_dataclass
class RopeTable:
    sin: f32["len d_head2"]
    cos: f32["len d_head2"]

    @staticmethod
    def create(max_len: int, hparams: Hparams, pos_scale: int = 1) -> "RopeTable":
        rope_max_timescale = hparams.rope_max_timescale
        d_head = hparams.d_head
        d = d_head // 2
        # endpoint=False is equivalent to what MaxText does. endpoint=True would be more natural, though.
        timescale = jnp.logspace(
            0, jnp.log10(jnp.float32(rope_max_timescale)), d, endpoint=False
        )
        position = jnp.arange(max_len, dtype=jnp.int32) // pos_scale
        sinusoid_inp = jnp.float32(position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        return RopeTable(sin=sin, cos=cos)

    def apply(self, rearrange_spec, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = einops.rearrange(self.sin, rearrange_spec)
        cos = einops.rearrange(self.cos, rearrange_spec)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.append(r1, r2, axis=-1)


@typechecked
def rms_norm(
    x: Union[bf16[b"batch/d len M"], bf16[b"batch/d n_block M"]]
) -> Union[bf16[b"batch/d len M"], bf16[b"batch/d n_block M"]]:
    mean2 = save_for_backward(
        jnp.mean(jax.lax.square(jnp.float32(x)), axis=-1, keepdims=True)
    )
    return jnp.bfloat16(x * jax.lax.rsqrt(mean2 + 1e-6))


@pytree_dataclass
class Metrics:
    loss: f32[b""]
    learning_rate: f32[b""]
    grad_norm: f32[b""]
    raw_grad_norm: f32[b""]


@dataclass(frozen=True)
class TrainingHparams:
    adam_b1: float
    adam_b2: float
    adam_eps: float
    adam_eps_root: float
    weight_decay: float
    warmup_steps: int
    steps: int
    steps_for_lr: int
    cosine_learning_rate_final_fraction: float
    learning_rate: float
    tokens: TokenBatchParams
    seed: int
    queue: Optional[str] = None
    use_grad_clip: bool = True
    use_gpu: bool = False
    use_single_pod: bool = False


@pytree_dataclass
class State:
    weights: Model
    adam_mu: Model
    adam_nu: Model

    @staticmethod
    def init(hparams: Hparams, rng: PRNGKey) -> "State":
        weights = Model.init(hparams, rng)
        adam_mu = jax.tree.map(lambda p: p * 0.0, weights)
        adam_nu = jax.tree.map(lambda p: p * 0.0, weights)
        return State(weights=weights, adam_mu=adam_mu, adam_nu=adam_nu)


@partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0,))
@shardtypes.scope
def training_step(
    state: State,
    step: u32[b""],
    h: Hparams,
    hparams: TrainingHparams,
    batch: TokenBatch,
) -> Tuple[Any, Metrics, SyntheticMetrics]:
    @partial(
        shardtypes.typed_shard_map, check_rep=False
    )  # check_rep=False for https://github.com/google/jax/issues/20335
    def sharded_step(
        state: State, step: u32[b""], batch: TokenBatch
    ) -> Tuple[State, Metrics, SyntheticMetrics]:
        (loss, synth_metrics), grad = jax.value_and_grad(
            lambda weights: weights.loss(h, batch), has_aux=True
        )(state.weights)
        # Gradients have already been reduced across chips because the gradient of the weight `all_gather`
        # is weight-gradient `psum_scatter`. Loss, on the other hand, hasn't been reduced across chips: if we
        # did that inside the autodiff, we'd be double-reducing the loss, effectively multiplying it by the
        # amount of data parallelism.
        #
        # So we reduce the loss across chips _outside_ the autodiff.
        loss = jax.lax.psum(loss, ("d", "t"))

        # Other than global-norm of gradients, no other communication is needed during the weight update,
        # because weights and grads are already fully sharded, as checked below.

        # Calculate learning rate from step number.
        # We use linear warmup then cosine decay. See https://arxiv.org/pdf/2307.09288.pdf section 2.2
        warmup_lr = (
            jnp.float32(step) / jnp.float32(hparams.warmup_steps)
        ) * hparams.learning_rate
        cosine = jnp.cos(
            jnp.pi
            * (
                jnp.float32(step - hparams.warmup_steps)
                / jnp.float32(hparams.steps_for_lr - hparams.warmup_steps)
            )
        )
        cosine_lr = hparams.learning_rate * (
            hparams.cosine_learning_rate_final_fraction
            + (1 - hparams.cosine_learning_rate_final_fraction) * (cosine * 0.5 + 0.5)
        )
        lr = jnp.where(step < hparams.warmup_steps, warmup_lr, cosine_lr)

        # AdamW optimizer with global gradient clipping.
        grad_leaves, grad_treedef = jax.tree_util.tree_flatten(grad)
        grad_leaves = [
            shardops.pmean_across_replicas(pspec, g)
            for g, pspec in zip(
                grad_leaves, tree_leaves(shardtypes.make_partition_specs(State))
            )
        ]
        global_norm_square = jnp.float32(0.0)
        for g in grad_leaves:
            assert g.dtype == jnp.float32
            global_norm_square += jnp.sum(jax.lax.square(g))
        global_norm_square = jax.lax.psum(global_norm_square, ("d", "t"))
        global_norm = jnp.sqrt(global_norm_square)

        base = h.base

        p = get_parameterization(h.parameterization)
        target_head_dim = h.n_q_per_kv * h.n_kv * h.d_head
        base_head_dim = base.n_q_per_kv * base.n_kv * base.d_head

        embed_lr_scale = h.gamma_embed * (h.d_model / base.d_model) ** -p.embed_lr
        unembed_lr_scale = h.gamma_unembed * (h.d_model / base.d_model) ** -p.unembed_lr

        lr_scales = Model(
            embed=embed_lr_scale,
            unembed=unembed_lr_scale,
            ln1=1.0,
            ln2=1.0,
            w_q=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
            w_kv=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
            w_o=h.gamma_hidden * (target_head_dim / base_head_dim) ** -p.hidden_lr,
            w_gate=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
            w_up=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
            w_down=h.gamma_hidden * (h.d_ff / base.d_ff) ** -p.hidden_lr,
            final_layer_norm=1.0,
            e_ln1=1.0,
            e_ln2=1.0,
            e_w_q=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
            e_w_kv=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
            e_w_o=h.gamma_hidden * (target_head_dim / base_head_dim) ** -p.hidden_lr,
            e_w_gate=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
            e_w_up=h.gamma_hidden * (h.d_model / base.d_model) ** -p.hidden_lr,
            e_w_down=h.gamma_hidden * (h.d_ff / base.d_ff) ** -p.hidden_lr,
            # Token decoder lr scales
            t_ln1=1.0,
            t_ln2=1.0,
            t_w_q=1.0,
            t_w_kv=1.0,
            t_w_o=1.0,
            t_w_gate=1.0,
            t_w_up=1.0,
            t_w_down=1.0,
            # Cross attention lr scales
            x_w_q=1.0,
            x_w_kv=1.0,
            x_w_o=1.0,
            x_lnx=1.0,
            x_lnz=1.0,
            w_mix=1.0,
            w_reduce_q=1.0,
            w_reduce_kv=1.0,
        )

        if hparams.use_grad_clip:
            clip_value = 1.0
            rescale = jnp.minimum(1.0, clip_value / global_norm)
        else:
            rescale = 1.0

        new_ps = []
        new_mus = []
        new_nus = []
        for p, g, mu, nu, spec, lr_scale in zip(
            tree_leaves(state.weights),
            grad_leaves,
            tree_leaves(state.adam_mu),
            tree_leaves(state.adam_nu),
            tree_leaves(shardtypes.make_partition_specs(State)),
            tree_leaves(lr_scales),
        ):
            # Gradient clipping
            g = g * rescale
            # Adam scaling
            mu = (1 - hparams.adam_b1) * g + hparams.adam_b1 * mu
            nu = (1 - hparams.adam_b2) * jax.lax.square(g) + hparams.adam_b2 * nu
            # We need step numbers to start at 1, not 0. Otherwise the bias correction produces NaN.
            completed_steps = step + 1
            mu_hat = mu / (1 - jnp.float32(hparams.adam_b1) ** completed_steps)
            nu_hat = nu / (1 - jnp.float32(hparams.adam_b2) ** completed_steps)
            # as per C.5. in https://arxiv.org/pdf2407.05872
            # they mention introducing hp a, b to below function,
            # TODO: test and see if a = b = something besides 1
            g = jnp.arctan2(mu_hat, jnp.sqrt(nu_hat))

            # Weight decay
            g += hparams.weight_decay * p
            # Learning rate
            g *= lr * lr_scale

            # Apply update
            new_ps.append(p - g)
            new_mus.append(mu)
            new_nus.append(nu)

        new_state = State(
            weights=jax.tree_util.tree_unflatten(grad_treedef, new_ps),
            adam_mu=jax.tree_util.tree_unflatten(grad_treedef, new_mus),
            adam_nu=jax.tree_util.tree_unflatten(grad_treedef, new_nus),
        )
        metrics = Metrics(
            loss=loss,
            learning_rate=lr,
            grad_norm=global_norm * rescale,
            raw_grad_norm=global_norm,
        )
        return new_state, metrics, synth_metrics

    return sharded_step(state, step, batch)


@dataclass(frozen=True)
class Paths:
    root_working_dir: str
    model_name: Optional[str]


@dataclass(frozen=True)
class MeshConfig:
    d: int
    t: int


@dataclass(frozen=True)
class Config:
    model: Hparams
    training: TrainingHparams
    paths: Paths
    num_hosts: int
    checkpoint_interval: int
    mesh: MeshConfig
    io: training_io.IOConfig
    flat_tokens: Optional[FlatTokensParams] = None
    hf_dataset: Optional[HuggingFaceDataParams] = None
    synthetic_dataset: Optional[SyntheticDataParams] = None

    def __post_init__(self):
        assert (
            self.flat_tokens is not None
            or self.hf_dataset is not None
            or self.synthetic_dataset is not None
        ), "Must provide either flat_tokens or hf_dataset or synthetic_dataset."
        assert not (
            self.flat_tokens is not None
            and self.hf_dataset is not None
            and self.synthetic_dataset is not None
        ), "Should not specify both flat_tokens and hf_dataset and synthetic_dataset."

    @cached_property
    def training_data(
        self,
    ) -> Union[FlatTokensParams, HuggingFaceDataParams, SyntheticDataParams]:
        return self.flat_tokens or self.hf_dataset or self.synthetic_dataset


def main_contained(config, logger):
    """Main program, which does not access external services except as specified by config.paths or logger."""
    # Use partitionable (and hopefully fusable!) RNG.
    #
    # This is slower in compute time than 'unsafe_rbg' with flag '--xla_tpu_spmd_rng_bit_generator_unsafe=true',
    # but hopefully faster in memory time because it's fusable.
    # TODO: check this is true and if not, provide our own that actually is fusable.

    # 4x 1 chip (2 cores) per process:
    if config.training.use_single_pod:
        os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
        os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
    jax.config.update("jax_threefry_partitionable", True)
    with Mesh(
        mesh_utils.create_device_mesh([config.mesh.d, config.mesh.t], jax.devices()),
        ("d", "t"),
    ):
        root_rng = jax.random.PRNGKey(config.training.seed)

        loader = get_loader("train", config.training_data, config.training.tokens)
        assert (
            config.model.vocab > loader.max_token_id
        ), f"{config.model.vocab} vs {loader.max_token_id}"
        config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
        model_name = (
            config.paths.model_name
            if config.paths.model_name
            else get_model_name(config_name)
        )
        model_dir = os.path.join(config.paths.root_working_dir, model_name)
        print(model_name)

        state = jax.jit(partial(State.init, config.model))(
            fold_in_str(root_rng, "init")
        )
        training_io.mkdir(model_dir)

        state, start_step = training_io.load_checkpoint_if_it_exists(
            model_dir, state, config.io
        )

        # Explicitly compile training step, to record XLA HLO graph.
        # See https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev
        c_training_step = training_step.lower(
            state, jnp.uint32(0), config.model, config.training, loader.load(0)
        ).compile()
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # training_io.save_hlo_svg(os.path.join(model_dir, f'training_step_optimized_hlo_{date}.svg'), c_training_step)

        log_interval = math.ceil(config.training.steps / 5000)
        print(f"{log_interval=}")

        cum_metrics = None

        def update_metrics(metrics: Metrics):
            nonlocal cum_metrics
            cum_metrics.loss += metrics.loss
            cum_metrics.grad_norm += metrics.grad_norm
            cum_metrics.raw_grad_norm += metrics.raw_grad_norm
            cum_metrics.learning_rate += metrics.learning_rate

        start_time = time.time()

        for step in range(start_step, config.training.steps):
            if step % config.checkpoint_interval == 0 and step > start_step:
                training_io.save_checkpoint(model_dir, step, state, config.io)

            # We profile on the second step, because the first step has a long pause for XLA
            # compilation and initial shuffle buffer loading.
            if training_io.is_device_0() and step == start_step + 1:
                jax.block_until_ready(state)
                training_io.start_profile()
                profile_start = time.time()

            # if half way point, double seq length and halve batch size
            if step == config.training.steps // 2:
                print("updating seq length and batch size")
                tokens = dataclasses.replace(
                    config.training.tokens,
                    len=config.training.tokens.len * 2,
                    batch=max(config.mesh.d, config.training.tokens.batch // 2),
                )
                config = dataclasses.replace(
                    config, training=dataclasses.replace(config.training, tokens=tokens)
                )
                loader = get_loader(
                    "train", config.training_data, config.training.tokens
                )
                c_training_step = training_step.lower(
                    state,
                    jnp.uint32(0),
                    config.model,
                    config.training,
                    loader.load(step),
                ).compile()

            batch = loader.load(step)
            state, output, synth_metrics = c_training_step(
                state, jnp.uint32(step), batch
            )

            # Run profile for two steps, to include data loading time in between them.
            if training_io.is_device_0() and step == start_step + 2:
                jax.block_until_ready(state)
                profile_duration = time.time() - profile_start
                training_io.stop_profile(model_dir)

                # Print MFU, including (one step of) data loading time.
                print(f"Profile time: {profile_duration}s for 2 steps.")
                model_params = jax.tree.reduce(
                    operator.add, jax.tree.map(lambda w: w.size, state.weights)
                )
                tokens = config.training.tokens.batch * config.training.tokens.len
                print(f"Model params: {model_params:_}")
                print(f"Tokens: {tokens:_}")
                device_flops = training_io.get_flops_per_device()
                num_devices = jax.device_count()
                print(
                    f"MFU (projections only): {100 * (2 * 6 * model_params * tokens / (num_devices * profile_duration)) / device_flops:.2f}% MFU"
                )

            if step % log_interval == 0:
                if cum_metrics:
                    cum_metrics = Metrics(
                        loss=cum_metrics.loss / log_interval,
                        learning_rate=cum_metrics.learning_rate / log_interval,
                        grad_norm=cum_metrics.grad_norm / log_interval,
                        raw_grad_norm=cum_metrics.raw_grad_norm / log_interval,
                    )
                else:
                    cum_metrics = output
                if batch.loss_masks is not None:
                    training_io.log(step, logger, synth_metrics)
                training_io.log(step, logger, cum_metrics)
                cum_metrics = output
            else:
                update_metrics(output)

        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds")


def clear_tpu_locks():
    try:
        raw_pids = subprocess.run(
            ["lsof", "-w", "/dev/accel0"], capture_output=True, text=True
        ).stdout
        pids = set()
        for line in raw_pids.splitlines()[1:]:
            parts = line.split()
            if len(parts) > 1:
                pids.add(parts[1])
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
        if pids:
            os.remove("/tmp/libtpu_lockfile")
    except Exception as e:
        print(f"Error clearing TPU locks: {e}")
        pass


def get_model_name(config_name: str):
    overrides = hydra.core.hydra_config.HydraConfig.get()["job"]["override_dirname"]
    ignore_overrides = [
        "training.queue",
    ]
    overrides = [
        override.lstrip("+")
        for override in overrides.split(",")
        if override.lstrip("+").split("=")[0] not in ignore_overrides
    ]

    overrides = "_".join(overrides)
    return f"{config_name}_{overrides}" if overrides else config_name


@hydra.main(config_path="configs", version_base=None)
def main(config):
    config = jax_extra.make_dataclass_from_dict(Config, config)
    if config.training.queue:
        config_name = hydra.core.hydra_config.HydraConfig.get()["job"]["config_name"]
        task_name = (
            config.paths.model_name
            if config.paths.model_name
            else get_model_name(config_name)
        )
        git_branch_name = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        task = Task.init(
            project_name=f"{config_name}/{git_branch_name}", task_name=task_name
        )

        if config.training.use_gpu:
            task.set_packages("requirements-gpu.txt")
        else:
            task.set_packages("requirements-tpu.txt")

        result = subprocess.run(
            ["datasets-cli", "env"], capture_output=True, text=True, check=True
        )

        print("Datasets CLI Environment:")
        print(result.stdout)

        task.add_tags([git_branch_name])
        logger = task.get_logger()
        task.execute_remotely(queue_name=config.training.queue)
        task.launch_multi_node(
            config.num_hosts, wait=True, queue=config.training.queue + "-workers"
        )
        clear_tpu_locks()
        jax.distributed.initialize(
            os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"],
            num_processes=int(os.environ["WORLD_SIZE"]),
            process_id=int(os.environ["RANK"]),
        )
    else:
        logger = None
    main_contained(config, logger)

    if not training_io.is_device_0():
        task.set_system_tags((task.get_system_tags() or []) + ["hidden"])


if __name__ == "__main__":
    main()
