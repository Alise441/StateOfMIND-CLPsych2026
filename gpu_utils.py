"""
GPU memory helper for CLPsych 2026 Task 1 notebooks.

Loading two ~25-27B models with vLLM in the same process can leave VRAM
fragmented even after the previous model is deleted; call free_gpu() between
models to clear the RAG embedding-model cache and force a CUDA cache release.
"""

import gc

import torch

import rag_index


def free_gpu():
    """Clear the cached embedding model(s) and empty the CUDA cache."""
    rag_index._model_cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU VRAM freed! Ready for vLLM.")
