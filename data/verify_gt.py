import numpy as np

def read_ivecs(filename):
    """读取 ivecs 格式文件（整数向量）"""
    iv = np.fromfile(filename, dtype=np.int32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv[0]
    assert dim > 0
    iv = iv.reshape(-1, 1 + dim)
    if not all(iv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    iv = iv[:, 1:]
    return iv

if __name__ == '__main__':
    # 读取 deep10m 的 groundtruth 文件
    gnd_path = '/data/vector_datasets/deep10m/deep10m_groundtruth.ivecs'
    
    print(f"正在读取 groundtruth 文件: {gnd_path}")
    groundtruth = read_ivecs(gnd_path)
    
    print(f"Groundtruth 形状: {groundtruth.shape}")
    print(f"\n第一个查询的 10-NN id:")
    print(groundtruth[3, :10])

