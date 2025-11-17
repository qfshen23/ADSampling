#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import struct

import numpy as np
from huggingface_hub import snapshot_download
from datasets import load_dataset


def write_fvecs_stream(filename, dataset, max_num=None, emb_key="emb", log_every=10000):
    """
    从 HuggingFace iterable dataset (streaming) 流式写入 fvecs。
    不在内存里攒 list。
    """
    print(f"[write_fvecs_stream] Writing fvecs to {filename} ...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    dim0 = None
    count = 0

    with open(filename, "wb") as fp:
        for i, item in enumerate(dataset):
            if max_num is not None and i >= max_num:
                break

            if i % log_every == 0:
                print(f"  Processed {i} vectors...")

            # 取 embedding
            emb = np.asarray(item[emb_key], dtype=np.float32)

            # 维度检查
            if emb.ndim != 1:
                raise ValueError(f"Embedding at index {i} is not 1D: shape={emb.shape}")

            d = emb.shape[0]
            if dim0 is None:
                dim0 = d
                print(f"  Detected dimension: {dim0}")
            elif d != dim0:
                raise ValueError(f"Dimension mismatch at index {i}: got {d}, expected {dim0}")

            # 写入 fvecs: [dim(int32)] + [dim * float32]
            fp.write(struct.pack("I", d))
            fp.write(emb.tobytes())

            count += 1

    print(f"[write_fvecs_stream] Finished writing {count} vectors to {filename} (dim={dim0})")


def write_fvecs_from_npy_base(
    dataset_name="msmarco",
    hf_repo_id="Cohere/msmarco-v2.1-embed-english-v3",
    npy_subdir="passages_npy",
    max_base=None,
    log_every=1_000_000,
):
    """
    使用 HuggingFace Hub 上的 .npy 分片来写 base.fvecs，避免 Parquet + PyArrow 的 list index overflow 问题。

    目录结构假定类似：
        <repo_dir>/
          passages_npy/
            cohere-msmarco-passage-embeddings-00000-of-00108.npy
            cohere-msmarco-passage-embeddings-00001-of-00108.npy
            ...
    """
    # 1. 把整个 dataset repo 同步到本地缓存
    print(f"[write_fvecs_from_npy_base] Downloading snapshot of {hf_repo_id} ...")
    repo_dir = snapshot_download(repo_id=hf_repo_id, repo_type="dataset", cache_dir="/data/hf_cache", max_workers=16)
    print(f"[write_fvecs_from_npy_base] Local repo dir: {repo_dir}")

    # 2. 准备输出路径
    data_dir = f"/data/vector_datasets/{dataset_name}/"
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{dataset_name}_base.fvecs")
    print(f"[write_fvecs_from_npy_base] Output fvecs: {out_path}")

    # 3. 找到所有 npy 分片
    npy_root = os.path.join(repo_dir, npy_subdir)
    npy_paths = sorted(glob.glob(os.path.join(npy_root, "*.npy")))
    if not npy_paths:
        raise RuntimeError(f"No .npy files found under {npy_root}. Check repo layout or npy_subdir.")

    print(f"[write_fvecs_from_npy_base] Found {len(npy_paths)} npy shard(s)")
    for p in npy_paths:
        print(f"  - {p}")

    dim0 = None
    total_written = 0

    with open(out_path, "wb") as fp:
        for shard_idx, npy_path in enumerate(npy_paths):
            print(f"[write_fvecs_from_npy_base] Loading shard {shard_idx}: {npy_path}")
            # 使用 mmap_mode="r" 避免一次性加载整个 shard 到内存
            mat = np.load(npy_path, mmap_mode="r")  # 预期 shape: (num_rows, dim)

            if mat.ndim != 2:
                raise ValueError(f"Shard {npy_path} has invalid shape {mat.shape}, expected 2D")

            num_rows, dim = mat.shape

            if dim0 is None:
                dim0 = dim
                print(f"[write_fvecs_from_npy_base] Detected dimension: {dim0}")
            else:
                if dim != dim0:
                    raise ValueError(
                        f"Dimension mismatch in shard {npy_path}: got {dim}, expected {dim0}"
                    )

            for i in range(num_rows):
                if max_base is not None and total_written >= max_base:
                    break

                emb = np.asarray(mat[i], dtype=np.float32)  # shape: (dim0,)
                d = emb.shape[0]

                # 写入 fvecs: [dim(int32)] + [dim * float32]
                fp.write(struct.pack("I", d))
                fp.write(emb.tobytes())
                total_written += 1

                if total_written % log_every == 0:
                    print(f"  Written {total_written} base vectors...")

            if max_base is not None and total_written >= max_base:
                print("[write_fvecs_from_npy_base] Reached max_base limit, stopping early.")
                break

    print(
        f"[write_fvecs_from_npy_base] Finished writing {total_written} base vectors "
        f"to {out_path} (dim={dim0})"
    )
    return out_path


def download_and_save(dataset_name="msmarco", max_base=None, max_query=None):
    """
    顶层封装：
      - base：通过 .npy 分片写 /data/vector_datasets/<dataset_name>/<dataset_name>_base.fvecs
      - query：通过 datasets.load_dataset(..., streaming=True) 写 query.fvecs
    """
    data_dir = f"/data/vector_datasets/{dataset_name}/"
    os.makedirs(data_dir, exist_ok=True)
    print(f"[download_and_save] Data directory: {data_dir}")

    # 1. base vectors（passages）—— 使用 npy 分片，避免 Parquet 问题
    print("\n" + "=" * 60)
    print("Converting base vectors (passages) from .npy ...")
    print("=" * 60)

    base_file = write_fvecs_from_npy_base(
        dataset_name=dataset_name,
        hf_repo_id="Cohere/msmarco-v2.1-embed-english-v3",
        npy_subdir="passages_npy",
        max_base=max_base,
        log_every=1_000_000,
    )

    # 2. query vectors —— 数量较小，继续用 streaming parquet 即可
    print("\n" + "=" * 60)
    print("Downloading query vectors ...")
    print("=" * 60)

    queries_dataset = load_dataset(
        "Cohere/msmarco-v2.1-embed-english-v3",
        "queries",
        split="test",
        streaming=True,  # queries 规模不大，不会触发 list index overflow
    )
    query_file = os.path.join(data_dir, f"{dataset_name}_query.fvecs")
    write_fvecs_stream(
        query_file,
        queries_dataset,
        max_num=max_query,
        emb_key="emb",
        log_every=1000,
    )

    print("\n" + "=" * 60)
    print("[download_and_save] Download completed!")
    print(f"Base vectors:  {base_file}")
    print(f"Query vectors: {query_file}")
    print("=" * 60)


if __name__ == "__main__":
    # 你可以根据需要修改 max_base / max_query
    download_and_save(
        dataset_name="msmarco",
        max_base=None,   # None 表示尽可能全量（受磁盘空间限制）
        max_query=None,  # queries 很小，直接全部写
    )
