import numpy as np
import re
import hashlib
from scipy.integrate import quad
from typing import List, Tuple, Dict, Set
from database import PolygonDatabase
from progress_bar import ProgressBar  # 导入进度条类

"""GDS文件解析"""


class GDSParser:
    @staticmethod
    def parse(gds_content: str) -> List[np.ndarray]:
        boundary_pattern = re.compile(r'BOUNDARY.*?ENDEL', re.DOTALL)
        boundaries = boundary_pattern.findall(gds_content)
        polygons = []
        # 添加进度条：解析BOUNDARY
        progress = ProgressBar(total=len(boundaries), desc="解析GDS边界")
        for boundary in boundaries:
            xy_pattern = re.compile(r'(\d+):\s*(\d+)')
            xy_matches = xy_pattern.findall(boundary)
            if len(xy_matches) < 3:
                progress.update()
                continue
            polygon = np.array([[int(x), int(y)] for x, y in xy_matches], dtype=np.float64)
            polygons.append(polygon)
            progress.update()
        progress.finish()
        return polygons


"""转向函数计算与距离度量（融合两篇论文核心逻辑）"""


class TurningFunction:
    @staticmethod
    def compute(polygon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 计算边向量与角度
        edges = np.diff(polygon, axis=0)
        edges = np.vstack([edges, polygon[0] - polygon[-1]])  # 闭合多边形
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        edge_lengths = np.linalg.norm(edges, axis=1)
        total_length = edge_lengths.sum()
        # 归一化周长断点
        x = np.cumsum(edge_lengths / total_length)[:-1]
        x = np.insert(x, 0, 0.0)
        # 累积角度（左拐增加，右拐减少）
        cumulative_angles = np.cumsum(angles)
        y = np.insert(cumulative_angles, 0, 0.0)
        return x, y

    @staticmethod
    def mean_reduce(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """均值归约：消除旋转影响（论文1 3.2节）"""
        y = y[1:]
        y_mean = np.trapz(y, x)
        y_reduced = y - y_mean
        return x, y_reduced

    @staticmethod
    def l1_distance(tf1: Tuple[np.ndarray, np.ndarray], tf2: Tuple[np.ndarray, np.ndarray]) -> float:
        """L₁距离（论文1公式）"""
        f_x, f_y = tf1
        g_x, g_y = tf2

        def f(x: float) -> float:
            idx = np.searchsorted(f_x, x) - 1
            return f_y[idx] if idx >= 0 else f_y[0]

        def g(x: float) -> float:
            idx = np.searchsorted(g_x, x) - 1
            return g_y[idx] if idx >= 0 else g_y[0]

        integrand = lambda x: abs(f(x) - g(x))
        distance, _ = quad(integrand, 0, 1, limit=1000)
        return distance

    @staticmethod
    def l2_distance(tf1: Tuple[np.ndarray, np.ndarray], tf2: Tuple[np.ndarray, np.ndarray]) -> float:
        """L₂距离（论文2核心metric）"""
        f_x, f_y = tf1
        g_x, g_y = tf2

        def f(x: float) -> float:
            idx = np.searchsorted(f_x, x) - 1
            return f_y[idx] if idx >= 0 else f_y[0]

        def g(x: float) -> float:
            idx = np.searchsorted(g_x, x) - 1
            return g_y[idx] if idx >= 0 else g_y[0]

        integrand = lambda x: (f(x) - g(x)) ** 2
        distance_sq, _ = quad(integrand, 0, 1, limit=1000)
        return np.sqrt(distance_sq)

    @staticmethod
    def d1_distance(tf_q: Tuple[np.ndarray, np.ndarray], tf_c: Tuple[np.ndarray, np.ndarray],
                    shift_steps: int = 5) -> float:
        """D₁距离：支持水平平移（适配遮挡，论文1 3.3节优化）"""
        min_dist = float('inf')
        # 扩大水平平移搜索范围，适配遮挡导致的断点偏移
        for s in range(shift_steps):
            shifted_x = np.roll(tf_c[0], s)
            shifted_y = np.roll(tf_c[1], s)
            dist = TurningFunction.l1_distance(tf_q, (shifted_x, shifted_y))
            min_dist = min(min_dist, dist)
        return min_dist

    @staticmethod
    def d2_distance(tf_q: Tuple[np.ndarray, np.ndarray], tf_c: Tuple[np.ndarray, np.ndarray],
                    shift_steps: int = 5) -> float:
        """D₂距离：支持水平平移（适配遮挡，论文2+论文1 B.3节融合）"""
        min_dist = float('inf')
        for s in range(shift_steps):
            shifted_x = np.roll(tf_c[0], s)
            shifted_y = np.roll(tf_c[1], s)
            dist = TurningFunction.l2_distance(tf_q, (shifted_x, shifted_y))
            min_dist = min(min_dist, dist)
        return min_dist


"""LSH哈希家族（论文1核心结构）"""


class LSHFamily:
    def __init__(self, m_max: int = 1000):
        self.m_max = m_max
        self.a_m = -((m_max // 2) - 1) * np.pi
        self.b_m = ((m_max // 2) + 3) * np.pi
        self.span_m = (self.b_m - self.a_m) / 2  # 范围优化

    def random_point_lsh(self, tf: Tuple[np.ndarray, np.ndarray], num_hashes: int = 100) -> str:
        """random-point-LSH（L₁适配）"""
        f_x, f_y = tf
        hash_bits = []
        for _ in range(num_hashes):
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, self.span_m)
            idx = np.searchsorted(f_x, x) - 1
            f_val = f_y[idx] if idx >= 0 else f_y[0]
            if f_val > y + 1e-8:
                hash_bits.append('1')
            elif f_val < y - 1e-8:
                hash_bits.append('-1')
            else:
                hash_bits.append('0')
        return ''.join(hash_bits)

    def discrete_sample_lsh(self, tf: Tuple[np.ndarray, np.ndarray], n_samples: int = 200) -> str:
        """discrete-sample-LSH（L₂适配）"""
        f_x, f_y = tf
        sample_x = np.linspace(0, 1, n_samples)
        sample_y = []
        for x in sample_x:
            idx = np.searchsorted(f_x, x) - 1
            sample_y.append(f_y[idx] if idx >= 0 else f_y[0])
        sample_y = np.array(sample_y)
        vec = sample_y / np.sqrt(n_samples)
        vec_str = ','.join([f"{v:.6f}" for v in vec])
        return hashlib.sha256(vec_str.encode()).hexdigest()


"""多边形标准化"""


class PolygonNormalizer:
    @staticmethod
    def normalize(polygon: np.ndarray) -> np.ndarray:
        # 平移：中心归零
        centroid = np.mean(polygon, axis=0)
        polygon_translated = polygon - centroid
        # 缩放：单位周长
        edges = np.diff(polygon_translated, axis=0)
        edges = np.vstack([edges, polygon_translated[0] - polygon_translated[-1]])
        edge_lengths = np.linalg.norm(edges, axis=1)
        total_length = edge_lengths.sum()
        if total_length == 0:
            return polygon_translated
        return polygon_translated / total_length

    @staticmethod
    def vertical_shift(tf: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """垂直移位：最小值为0（论文1 C.1节优化）"""
        x, y = tf
        y_shifted = y - np.min(y)
        return x, y_shifted


"""检索主类"""


class PolygonSimilarRetrieval:
    def __init__(self, m_max: int = 1000, db_path: str = "polygon_db.pkl"):
        self.db = PolygonDatabase(db_path)
        self.parser = GDSParser()
        self.tf_computer = TurningFunction()
        self.normalizer = PolygonNormalizer()
        self.lsh = LSHFamily(m_max=m_max)

    def add_polygon(self, polygon: np.ndarray) -> int:
        """添加单个多边形到数据库"""
        normalized_poly = self.normalizer.normalize(polygon)
        tf = self.tf_computer.compute(normalized_poly)
        tf_shifted = self.normalizer.vertical_shift(tf)
        tf_reduced = self.tf_computer.mean_reduce(*tf_shifted)
        l1_hash = self.lsh.random_point_lsh(tf_shifted, num_hashes=100)
        l2_hash = self.lsh.discrete_sample_lsh(tf_reduced, n_samples=200)
        return self.db.add_polygon(polygon, normalized_poly, tf_shifted, tf_reduced, l1_hash, l2_hash)

    def add_gds_file(self, gds_path: str) -> int:
        """从GDS文件批量添加多边形（带进度条）"""
        print(f"正在读取GDS文件：{gds_path}")
        with open(gds_path, 'r', encoding='utf-8') as f:
            gds_content = f.read()
        # 解析GDS得到多边形列表（内部已带进度条）
        polygons = self.parser.parse(gds_content)
        if not polygons:
            print("未解析到有效多边形")
            return 0
        # 添加多边形到数据库（带进度条）
        progress = ProgressBar(total=len(polygons), desc="添加多边形到数据库")
        for poly in polygons:
            self.add_polygon(poly)
            progress.update()
        progress.finish()
        return len(polygons)

    def retrieve_similar(self, query_polygon: np.ndarray,
                         distance_type: str = 'D2',
                         r: float = 0.3,  # 放松阈值适配遮挡
                         c: float = 2.5,  # 扩大近似因子，提升遮挡召回
                         shift_steps: int = 8) -> List[Tuple[int, float]]:
        """检索相似、旋转、部分遮挡的多边形（带进度条）"""
        # 查询预处理
        query_normalized = self.normalizer.normalize(query_polygon)
        query_tf = self.tf_computer.compute(query_normalized)
        query_tf_shifted = self.normalizer.vertical_shift(query_tf)
        query_tf_reduced = self.tf_computer.mean_reduce(*query_tf_shifted)

        # 获取候选集
        query_l1_hash = self.lsh.random_point_lsh(query_tf_shifted, num_hashes=100)
        query_l2_hash = self.lsh.discrete_sample_lsh(query_tf_reduced, n_samples=200)
        candidates = self.db.get_candidates(query_l1_hash, query_l2_hash)
        if not candidates:
            print("未找到候选多边形")
            return []
        candidates = list(candidates)

        # 计算真实距离（带进度条）
        progress = ProgressBar(total=len(candidates), desc="计算候选多边形距离")
        similar_results = []
        query_vertex_cnt = len(query_polygon)
        for idx in candidates:
            _, candidate_tf, candidate_tf_reduced, candidate_vertex_cnt = self.db.get_polygon_data(idx)
            # 筛选顶点数差异过大的多边形
            if abs(query_vertex_cnt - candidate_vertex_cnt) > max(5, query_vertex_cnt * 0.5):
                progress.update()
                continue

            # 根据距离类型计算距离
            if distance_type == 'L1':
                dist = self.tf_computer.l1_distance(query_tf_shifted, candidate_tf)
            elif distance_type == 'L2':
                dist = self.tf_computer.l2_distance(query_tf_reduced, candidate_tf_reduced)
            elif distance_type == 'D1':
                dist = self.tf_computer.d1_distance(query_tf_shifted, candidate_tf, shift_steps=shift_steps)
            elif distance_type == 'D2':
                dist = self.tf_computer.d2_distance(query_tf_reduced, candidate_tf_reduced, shift_steps=shift_steps)
            else:
                raise ValueError(f"不支持的距离类型：{distance_type}")

            # 筛选符合条件的结果
            if dist <= c * r:
                similar_results.append((idx, dist))
            progress.update()
        progress.finish()

        # 按距离排序
        similar_results.sort(key=lambda x: x[1])
        return similar_results


if __name__ == "__main__":
    retrieval = PolygonSimilarRetrieval(m_max=1000)
    # 添加GDS文件中的多边形
    gds_path = "test0.txt"
    added_cnt = retrieval.add_gds_file(gds_path)
    print(f"\n成功添加{added_cnt}个多边形到数据库")

    while True:
        if added_cnt > 0:
            a = int(input("\n请输入查询多边形索引（0~{}）：".format(added_cnt - 1)))
            if a < 0 or a >= added_cnt:
                print("索引超出范围")
                continue
            query_poly = retrieval.db.get_polygon_data(a)[0]
            print(f"\n查询第{a}个多边形（顶点数：{len(query_poly)}）")

            similar = retrieval.retrieve_similar(
                query_poly,
                distance_type='L1',
                r=0.8,
                c=2.5,
                shift_steps=10
            )

            print(f"\n找到{len(similar)}个相似多边形：")
            for idx, dist in similar:
                _, _, _, vertex_cnt = retrieval.db.get_polygon_data(idx)
                print(f"  - 索引{idx}：距离={dist:.4f}，顶点数={vertex_cnt}")