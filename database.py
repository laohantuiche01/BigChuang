import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import pickle
import os

class PolygonDatabase:
    """多边形检索后端数据库：单独存储所有核心数据，提供数据读写接口"""
    def __init__(self, db_path: str = "polygon_db.pkl"):
        self.db_path = db_path
        # 核心存储字段
        self.raw_polygons: List[np.ndarray] = []  # 原始顶点坐标
        self.normalized_polygons: List[np.ndarray] = []  # 标准化（平移+缩放）后多边形
        self.turning_functions: List[Tuple[np.ndarray, np.ndarray]] = []  # 转向函数（垂直移位后）
        self.mean_reduced_tfs: List[Tuple[np.ndarray, np.ndarray]] = []  # 均值归约后的转向函数
        self.polygon_vertex_counts: List[int] = []  # 顶点数（快速过滤）
        # LSH哈希表（双重哈希，支持L₁和L₂距离）
        self.l1_lsh_table: Dict[str, List[int]] = defaultdict(list)
        self.l2_lsh_table: Dict[str, List[List[int]]] = defaultdict(list)  # 存储分段哈希列表
        # 加载已有数据库
        self._load_db()

    def _load_db(self) -> None:
        """从文件加载数据库"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.raw_polygons = data.get("raw_polygons", [])
                self.normalized_polygons = data.get("normalized_polygons", [])
                self.turning_functions = data.get("turning_functions", [])
                self.mean_reduced_tfs = data.get("mean_reduced_tfs", [])
                self.polygon_vertex_counts = data.get("polygon_vertex_counts", [])
                self.l1_lsh_table = defaultdict(list, data.get("l1_lsh_table", {}))
                self.l2_lsh_table = defaultdict(list, data.get("l2_lsh_table", {}))

    def _save_db(self) -> None:
        """将数据库保存到文件"""
        data = {
            "raw_polygons": self.raw_polygons,
            "normalized_polygons": self.normalized_polygons,
            "turning_functions": self.turning_functions,
            "mean_reduced_tfs": self.mean_reduced_tfs,
            "polygon_vertex_counts": self.polygon_vertex_counts,
            "l1_lsh_table": dict(self.l1_lsh_table),
            "l2_lsh_table": dict(self.l2_lsh_table)
        }
        with open(self.db_path, 'wb') as f:
            pickle.dump(data, f)

    def add_polygon(self,
                   raw_poly: np.ndarray,
                   normalized_poly: np.ndarray,
                   tf_shifted: Tuple[np.ndarray, np.ndarray],
                   tf_reduced: Tuple[np.ndarray, np.ndarray],
                   l1_hash: str,
                   l2_hashes: List[str]) -> int:  # 改为接收哈希列表
        """添加多边形数据到数据库"""
        idx = len(self.raw_polygons)
        self.raw_polygons.append(raw_poly)
        self.normalized_polygons.append(normalized_poly)
        self.turning_functions.append(tf_shifted)
        self.mean_reduced_tfs.append(tf_reduced)
        self.polygon_vertex_counts.append(len(raw_poly))
        self.l1_lsh_table[l1_hash].append(idx)
        # 存储所有分段哈希
        for hash_str in l2_hashes:
            self.l2_lsh_table[hash_str].append(idx)
        return idx

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """计算两个哈希字符串的汉明距离"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def get_approximate_l1_candidates(self, query_hash: str, max_diff: int = 5) -> List[int]:
        """获取L1哈希的近似匹配候选集"""
        candidates = []
        for hash_str, indices in self.l1_lsh_table.items():
            if self.hamming_distance(query_hash, hash_str) <= max_diff:
                candidates.extend(indices)
        return candidates

    def get_approximate_l2_candidates(self, query_hashes: List[str]) -> List[int]:
        """获取L2分段哈希的匹配候选集（任一子段匹配）"""
        candidates = []
        for hash_str in query_hashes:
            candidates.extend(self.l2_lsh_table.get(hash_str, []))
        return candidates

    def get_candidates(self, query_l1_hash: str, query_l2_hashes: List[str],
                      distance_type: str) -> Set[int]:
        """根据查询哈希和距离类型获取候选多边形索引"""
        candidates = set()
        if distance_type in ['L1', 'D1']:
            # L1类型侧重L1哈希的近似匹配
            candidates.update(self.get_approximate_l1_candidates(query_l1_hash))
            candidates.update(self.get_approximate_l2_candidates(query_l2_hashes))
        else:
            # L2类型侧重L2分段哈希
            candidates.update(self.get_approximate_l2_candidates(query_l2_hashes))
            candidates.update(self.get_approximate_l1_candidates(query_l1_hash, max_diff=8))  # 放宽L1差异
        return candidates

    def get_polygon_data(self, idx: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int]:
        """获取指定索引的多边形数据"""
        return (
            self.raw_polygons[idx],
            self.turning_functions[idx],
            self.mean_reduced_tfs[idx],
            self.polygon_vertex_counts[idx]
        )

    def get_total_count(self) -> int:
        """获取数据库中多边形总数"""
        return len(self.raw_polygons)