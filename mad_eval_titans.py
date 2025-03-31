from torch import nn
import sys
import os
import pandas as pd 

from benchmark import benchmark
from mad.model.layers.titans.memory_models import (
    # MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention
)
from mad.model.layers.titans.titans_block import TitansLayer
from mad.model import AutoEncoder
from mad.model.language_model import LanguageModel

# neural memory related

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)                   # layers 2, 4, 6 have neural memory, can add more
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4                      # set smaller for more granularity for learning rate / momentum etc
NEURAL_MEM_BATCH_SIZE = 128                     # set smaller to update the neural memory weights more often as it traverses the sequence
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS

# perf related

USE_ACCELERATED_SCAN = True
USE_FLEX_ATTN = True
USE_FAST_INFERENCE = False


if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(
        dim = 64
    )
else:
    neural_memory_model = MemoryMLP(
        dim = 64,
        depth = NEURAL_MEMORY_DEPTH
    )



# def make_model_fn(
#         task: str,
#         vocab_size: int,
#         max_length: int,
# ) -> nn.Module:
#     dim = 384 
#     dim = 128 
#     return  MemoryAsContextTransformer(
#             num_tokens = vocab_size,
#             # dim = 384,
#             dim=dim,
#             depth = 8,
#             segment_len = WINDOW_SIZE,
#             num_persist_mem_tokens = NUM_PERSIST_MEM,
#             num_longterm_mem_tokens = NUM_LONGTERM_MEM,
#             neural_memory_layers = NEURAL_MEM_LAYERS,
#             neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
#             neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,
#             neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
#             neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
#             neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
#             use_flex_attn = USE_FLEX_ATTN,
#             sliding_window_attn = SLIDING_WINDOWS,
#             neural_memory_model = neural_memory_model,
#             neural_memory_kwargs = dict(
#                 dim_head = 64,
#                 heads = 4,
#                 attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
#                 qk_rmsnorm = NEURAL_MEM_QK_NORM,
#                 momentum = NEURAL_MEM_MOMENTUM,
#                 momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
#                 default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
#                 use_accelerated_scan = USE_ACCELERATED_SCAN,
#                 per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR
#             )
#         )


def make_model_fn(
        task: str,
        vocab_size: int,
        max_length: int,
) -> nn.Module:
    layers = [TitansLayer]

    dim = 128
    layer_config = {
        "dim": dim,
        "layer_index" : 0,
        "segment_len": WINDOW_SIZE,
        "dim_head": 64,
        "heads": 8,
        "ff_mult": 4,
        "num_residual_streams": 4,
        "neural_memory_segment_len": NEURAL_MEM_SEGMENT_LEN,
        "neural_mem_gate_attn_output": NEURAL_MEM_GATE_ATTN_OUTPUT,
        "num_longterm_mem_tokens": NUM_LONGTERM_MEM,
        "num_persist_mem_tokens": NUM_PERSIST_MEM,
        "neural_memory_batch_size": NEURAL_MEM_BATCH_SIZE,
        "neural_memory_qkv_receives_diff_views": NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
        "neural_memory_model": neural_memory_model,
        "neural_memory_kwargs": dict(
            dim_head=64,
            heads=4,
            attn_pool_chunks=STORE_ATTN_POOL_CHUNKS,
            qk_rmsnorm=NEURAL_MEM_QK_NORM,
            momentum=NEURAL_MEM_MOMENTUM,
            momentum_order=NEURAL_MEM_MOMENTUM_ORDER,
            default_step_transform_max_lr=NEURAL_MEM_MAX_LR,
            use_accelerated_scan=USE_ACCELERATED_SCAN,
            per_parameter_lr_modulation=MEMORY_MODEL_PER_LAYER_LEARNED_LR,
        ),
        "use_flex_attn": USE_FLEX_ATTN,
        "sliding_window_attn": SLIDING_WINDOWS,
        "neural_mem_weight_residual": NEURAL_MEM_WEIGHT_RESIDUAL,
        "is_neural_memory_layer": True,
        "max_length": max_length
    }
    layer_configs = [layer_config]
    backbone = LanguageModel if task not in {"compression"} else AutoEncoder

    return backbone(
        dim=dim,
        vocab_size=vocab_size,
        max_length=max_length,
        layers=layers,
        layer_cfgs=layer_configs,
        titans=True
    )


mad_scores = benchmark(make_model_fn=make_model_fn, model_id="Titans_MAC_1L_NM")
mad_scores.to_csv("titans_mac_scores.csv", index=False)