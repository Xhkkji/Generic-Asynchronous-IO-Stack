import argparse
import json
import os
import os.path as osp
import numpy as np
import torch

# 根目录和默认存储路径
data_root = osp.join(osp.dirname(__file__), '..', 'data')
user_root = "/home/lzl/nfs.d/dataset/graph_embedding/LinkPrediction/train_data/"
rmat_root = "/home/lzl/nfs.d/dataset/graph_embedding/graph_data/"
os.makedirs(data_root, exist_ok=True)


# 用户自定义数据集配置
USER_DATASET_CONFIG = {
    'com': {
        'edgelist_path': osp.join(user_root, 'com_srt_weg_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'LJ': {
        'edgelist_path': osp.join(user_root, 'LJ_srt_wei_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 16,
    },
    'soc': {
        'edgelist_path': osp.join(user_root, 'soc_srt_wei_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'wv': {
        'edgelist_path': osp.join(user_root, 'wv_srt_weg_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'ytb': {
        'edgelist_path': osp.join(user_root, 'ytb_srt_weg_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 100,
    },
    'uk': {
        'edgelist_path': osp.join(rmat_root, 'uk2007_srt_weg.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'pa': {
        'edgelist_path': osp.join(rmat_root, 'pa_srt_weg_commneg.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'twt': {
        'edgelist_path': osp.join(user_root, 'twt.edge'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
}


def load_edge_list(edgelist_path: str) -> np.ndarray:
    """读取原始边表，兼容两列或三列（带权重）格式。"""
    print(f"Loading edge list from: {edgelist_path}")
    if not osp.exists(edgelist_path):
        raise FileNotFoundError(f"Edgelist file not found at: {edgelist_path}")

    edge_data = np.loadtxt(edgelist_path, dtype=int)
    if edge_data.ndim == 1:
        edge_data = edge_data.reshape(1, -1)

    if edge_data.shape[1] == 3:
        print("Detected weighted edge list (3 columns). Using only source and target nodes.")
        edges_list = edge_data[:, :2]
    elif edge_data.shape[1] == 2:
        edges_list = edge_data
    else:
        raise ValueError("Edge list file should have 2 or 3 columns.")

    if edges_list.size == 0:
        raise ValueError("Edge list is empty.")

    return edges_list.astype(np.int64)


def generate_node_features(num_nodes: int, feature_dim: int, edges: np.ndarray, seed: int) -> torch.Tensor:
    """生成节点特征，首列包含归一化度信息，其余为随机噪声。"""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_nodes, feature_dim)).astype(np.float32)
    degrees = np.bincount(edges.reshape(-1), minlength=num_nodes)
    norm_deg = (degrees / (degrees.max() + 1e-6)).astype(np.float32)
    features[:, 0] = norm_deg if feature_dim > 0 else 0.0
    return torch.from_numpy(features)


def generate_labels(num_nodes: int, num_classes: int, edges: np.ndarray) -> torch.Tensor:
    """根据度分布分桶生成伪标签，保证每个节点有一个标签。"""
    degrees = np.bincount(edges.reshape(-1), minlength=num_nodes)
    if num_classes <= 1:
        labels = np.zeros(num_nodes, dtype=np.int64)
    else:
        quantiles = np.linspace(0, 100, num_classes + 1)[1:-1]
        bins = np.unique(np.percentile(degrees, quantiles))
        if bins.size == 0:
            bins = np.arange(1, num_classes)
        labels = np.digitize(degrees, bins, right=True).astype(np.int64)
    return torch.from_numpy(labels)


def split_masks(num_nodes: int, train_ratio: float, val_ratio: float, seed: int):
    """按照比例划分 train/val/test mask。"""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=gen)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


def save_numpy_artifacts(out_dir: str, edges: np.ndarray, features: torch.Tensor,
                         labels: torch.Tensor, train_mask: torch.Tensor,
                         val_mask: torch.Tensor, test_mask: torch.Tensor):
    np.save(osp.join(out_dir, "edge_index.npy"), edges)
    np.save(osp.join(out_dir, "node_feat.npy"), features.numpy())
    np.save(osp.join(out_dir, "node_label.npy"), labels.numpy())
    np.save(osp.join(out_dir, "train_mask.npy"), train_mask.numpy())
    np.save(osp.join(out_dir, "val_mask.npy"), val_mask.numpy())
    np.save(osp.join(out_dir, "test_mask.npy"), test_mask.numpy())

    np.savetxt(osp.join(out_dir, "train.csv"), np.nonzero(train_mask.numpy())[0], fmt="%d")
    np.savetxt(osp.join(out_dir, "valid.csv"), np.nonzero(val_mask.numpy())[0], fmt="%d")
    np.savetxt(osp.join(out_dir, "test.csv"), np.nonzero(test_mask.numpy())[0], fmt="%d")

    meta = {
        "num_edges": int(edges.shape[0]),
        "num_nodes": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_classes": int(labels.max().item() + 1),
    }
    with open(osp.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate a toy homogeneous dataset from an edge list.")
    parser.add_argument("--dataset", choices=USER_DATASET_CONFIG.keys(), default="com",
                        help="Key in USER_DATASET_CONFIG.")
    parser.add_argument("--edgelist_path", type=str, default=None, help="Override edge list path.")
    parser.add_argument("--feature_dim", type=int, default=None, help="Override feature dimension.")
    parser.add_argument("--num_classes", type=int, default=None, help="Override number of classes.")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = USER_DATASET_CONFIG[args.dataset]
    edgelist_path = args.edgelist_path or cfg["edgelist_path"]
    feature_dim = args.feature_dim or cfg["feature_dim"]
    num_classes = args.num_classes or cfg["num_classes"]

    edges = load_edge_list(edgelist_path)
    num_nodes = int(edges.max()) + 1

    features = generate_node_features(num_nodes, feature_dim, edges, args.seed)
    labels = generate_labels(num_nodes, num_classes, edges)
    train_mask, val_mask, test_mask = split_masks(num_nodes, args.train_ratio, args.val_ratio, args.seed)

    out_dir = osp.join(data_root, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    save_numpy_artifacts(out_dir, edges, features, labels, train_mask, val_mask, test_mask)

    print(f"Dataset `{args.dataset}` generated at: {out_dir}")
    print(f"Nodes: {num_nodes}, Edges: {edges.shape[0]}, Feature dim: {feature_dim}, Classes: {num_classes}")


if __name__ == "__main__":
    main()
