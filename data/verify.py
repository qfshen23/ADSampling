import numpy as np
import struct
import sys

def verify_fvecs(file_path, max_vectors=5):
    try:
        with open(file_path, 'rb') as f:
            index = 0
            ref_dim = None
            while True:
                # 读取维度头部
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break  # 正常 EOF
                if len(dim_bytes) < 4:
                    raise ValueError(f"Unexpected EOF when reading dimension of vector {index}")

                dim = struct.unpack('i', dim_bytes)[0]

                if dim <= 0 or dim > 10000:
                    raise ValueError(f"Vector {index} has invalid dimension: {dim}")

                if ref_dim is None:
                    ref_dim = dim
                    print(f"Reference dimension set to {ref_dim}")
                elif dim != ref_dim:
                    raise ValueError(f"Vector {index} has inconsistent dimension: {dim} (expected {ref_dim})")

                # 读取向量内容
                vec_bytes = f.read(4 * dim)
                if len(vec_bytes) < 4 * dim:
                    raise ValueError(f"Unexpected EOF when reading vector {index} values")

                vec = struct.unpack('f' * dim, vec_bytes)

                if index < max_vectors:
                    print(f"Vector {index} (dim={dim}): {vec[:5]}{'...' if dim > 5 else ''}")
                index += 1

            print(f"\n✅ File '{file_path}' passed verification with {index} vectors of dimension {ref_dim}.")

    except Exception as e:
        print(f"\n❌ Error verifying file '{file_path}': {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify the format of a .fvecs file.")
    parser.add_argument("file", type=str, help="Path to .fvecs file")
    args = parser.parse_args()

    verify_fvecs(args.file)
