import argparse
import numpy as np
import faiss
import struct
import os

source = '/data/vector_datasets'

def read_fvecs(fname):
    """读取 *.fvecs -> numpy.float32 (n, d)"""
    with open(fname, "rb") as f:
        buf = f.read()
    # 每个向量: 第 4 字节是维度 d，后面 d 个 float32
    dim = struct.unpack('i', buf[:4])[0]
    vec_size = 4 + 4 * dim
    n = len(buf) // vec_size
    data = np.frombuffer(buf, dtype=np.float32).reshape(n, dim + 1)
    return data[:, 1:]  # 去掉头部 dim

def write_ivecs(fname, arr):
    """保存 *.ivecs (int32)"""
    n, k = arr.shape
    with open(fname, "wb") as f:
        for row in arr:
            f.write(struct.pack('i', k))       # 写入 k
            f.write(row.astype(np.int32).tobytes())

def build_gt(base, queries, k=100):
    """使用 Faiss 暴力 L2 搜索得到 top-k indices"""
    d = base.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(base)                       # n_base x d
    D, I = index.search(queries, k)       # n_query x k
    return I

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="数据前缀，如 'sift1m'（将读取 sift1m_base.fvecs 等）")
    parser.add_argument("-k", "--topk", type=int, default=100,
                        help="返回前 k 个近邻 (默认 100)")
    args = parser.parse_args()

    base_file   = os.path.join(source, args.data, f"{args.data}_base.fvecs")
    query_file  = os.path.join(source, args.data, f"{args.data}_query.fvecs")
    out_file    = os.path.join(source, args.data, f"{args.data}_groundtruth.ivecs")

    if not os.path.exists(base_file) or not os.path.exists(query_file):
        raise FileNotFoundError("找不到 base/query fvecs 文件")

    print("Loading base vectors...")
    xb = read_fvecs(base_file)
    print(f"  base: {xb.shape}")

    print("Loading query vectors...")
    xq = read_fvecs(query_file)
    print(f"  query: {xq.shape}")

    print(f"Computing {args.topk}-NN groundtruth (L2)...")
    idx = build_gt(xb, xq, k=args.topk)

    print(f"Writing groundtruth to {out_file}")
    write_ivecs(out_file, idx)
    print("Done.")

if __name__ == "__main__":
    main()