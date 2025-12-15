import numpy as np
import hashlib
from scipy.integrate import quad
from collections import defaultdict
import re
import math
from typing import List, Tuple, Dict, Set

"""解析GDSII文件，提取BOUNDARY图形的顶点坐标"""
class GDSParser:
    @staticmethod

    #返回的是一个顶点的列表
    def parse(gds_content: str) -> List[np.ndarray]:
        boundary_pattern = re.compile(r'BOUNDARY.*?ENDEL', re.DOTALL)
        boundaries = boundary_pattern.findall(gds_content)

        polygons = []
        for boundary in boundaries:
            xy_pattern = re.compile(r'(\d+):\s*(\d+)')
            xy_matches = xy_pattern.findall(boundary)
            if len(xy_matches) < 3:
                continue
            polygon = np.array([[int(x), int(y)] for x, y in xy_matches], dtype=np.float64)
            polygons.append(polygon)
        return polygons

"""计算多边形的转向函数（累积角度步长函数）"""
class TurningFunction:
    @staticmethod
    def compute(polygon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        #计算边向量
        edges = np.diff(polygon, axis=0)
        edges = np.vstack([edges, polygon[0] - polygon[-1]])  #闭合多边形
        #计算每条边的角度和长度
        angles = np.arctan2(edges[:, 1], edges[:, 0])  #与x轴夹角（弧度）
        edge_lengths = np.linalg.norm(edges, axis=1)
        total_length = edge_lengths.sum()
        #归一化周长，得到断点x
        x = np.cumsum(edge_lengths / total_length)[:-1]  #除最后一个点（与0重合）
        x = np.insert(x, 0, 0.0)  #插入起点0
        #计算累积角度y（左拐增加，右拐减少）
        cumulative_angles = np.cumsum(angles)
        y = np.insert(cumulative_angles, 0, 0.0)  #插入起点角度

        return x, y

    """均值归约（论文3.2节）：消除垂直平移影响，用于D₁^⊥和D₂^⊥距离"""
    @staticmethod
    def mean_reduce(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y=y[1:]
        y_mean = np.trapz(y, x)
        y_reduced = y - y_mean
        return x, y_reduced

    """计算两个转向函数的L₁距离（论文公式）"""
    @staticmethod
    def l1_distance(tf1: Tuple[np.ndarray, np.ndarray], tf2: Tuple[np.ndarray, np.ndarray]) -> float:
        f_x, f_y = tf1
        g_x, g_y = tf2
        def f(x: float) -> float:
            idx = np.searchsorted(f_x, x) - 1
            return f_y[idx] if idx >= 0 else f_y[0]
        def g(x: float) -> float:
            idx = np.searchsorted(g_x, x) - 1
            return g_y[idx] if idx >= 0 else g_y[0]
        # 数值积分计算L距离
        integrand = lambda x: abs(f(x) - g(x))
        distance, _ = quad(integrand, 0, 1)
        return distance

    """计算两个转向函数的L₂距离（论文公式）"""
    @staticmethod
    def l2_distance(tf1: Tuple[np.ndarray, np.ndarray], tf2: Tuple[np.ndarray, np.ndarray]) -> float:
        f_x, f_y = tf1
        g_x, g_y = tf2
        def f(x: float) -> float:
            idx = np.searchsorted(f_x, x) - 1
            return f_y[idx] if idx >= 0 else f_y[0]
        def g(x: float) -> float:
            idx = np.searchsorted(g_x, x) - 1
            return g_y[idx] if idx >= 0 else g_y[0]
        integrand = lambda x: (f(x) - g(x)) ** 2
        distance_sq, _ = quad(integrand, 0, 1)
        return np.sqrt(distance_sq)


"""实现论文中的LSH哈希家族：random-point-LSH（L₁）和discrete-sample-LSH（L₂）"""
class LSHFamily:
    def __init__(self, m_max: int = 10):
        self.m_max = m_max
        self.a_m = -((m_max // 2) - 1) * np.pi  # 下界
        self.b_m = ((m_max // 2) + 3) * np.pi  # 上界
        self.span_m = (self.b_m - self.a_m) / 2  # 论文定理12/24：span上限（减半优化）

    """论文3.1节：random-point-LSH（L₁距离适配）生成哈希值：随机采样(num_hashes)个(x,y)点，判断函数位置"""
    def random_point_lsh(self, tf: Tuple[np.ndarray, np.ndarray], num_hashes: int = 100) -> str:
        f_x, f_y = tf
        hash_bits = []
        for _ in range(num_hashes):
            # 随机采样x∈[0,1]，y∈[0, span_m]（垂直平移后范围）
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, self.span_m)

            idx = np.searchsorted(f_x, x) - 1
            f_val = f_y[idx] if idx >= 0 else f_y[0]

            # 记录位置关系（1=上方，0=重合，-1=下方）
            if f_val > y + 1e-8:
                hash_bits.append('1')
            elif f_val < y - 1e-8:
                hash_bits.append('-1')
            else:
                hash_bits.append('0')

        return ''.join(hash_bits)

        """论文B.1节：discrete-sample-LSH（L₂距离适配）"""
    def discrete_sample_lsh(self, tf: Tuple[np.ndarray, np.ndarray], n_samples: int = 200) -> str:

        f_x, f_y = tf
        # 均匀采样x坐标
        sample_x = np.linspace(0, 1, n_samples)
        # 插值得到对应y值
        sample_y = []
        for x in sample_x:
            idx = np.searchsorted(f_x, x) - 1
            sample_y.append(f_y[idx] if idx >= 0 else f_y[0])
        sample_y = np.array(sample_y)

        # 标准化向量（论文B.1节）
        vec = sample_y / np.sqrt(n_samples)
        # 转换为字符串计算哈希
        vec_str = ','.join([f"{v:.6f}" for v in vec])
        return hashlib.sha256(vec_str.encode()).hexdigest()

"""多边形标准化：平移、缩放、垂直移位（论文5节优化）"""
class PolygonNormalizer:
    @staticmethod
    def normalize(polygon: np.ndarray) -> np.ndarray:
        #平移：中心坐标归零
        centroid = np.mean(polygon, axis=0)
        polygon_translated = polygon - centroid
        #缩放至单位周长
        edges = np.diff(polygon_translated, axis=0)
        edges = np.vstack([edges, polygon_translated[0] - polygon_translated[-1]])
        edge_lengths = np.linalg.norm(edges, axis=1)
        total_length = edge_lengths.sum()
        if total_length == 0:
            return polygon_translated
        polygon_normalized = polygon_translated / total_length

        return polygon_normalized

    @staticmethod
    def vertical_shift(tf: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """垂直移位：使转向函数最小值为0"""
        x, y = tf
        y_shifted = y - np.min(y)
        return x, y_shifted

"""基于论文LSH理论的多边形相似检索系统"""
class PolygonSimilarRetrieval:
    """
        m_max: 支持的最大顶点数
        存储核心数据结构：哈希表+原始数据数组
    """
    def __init__(self, m_max: int = 1000):
        # 基础组件初始化
        self.m_max = m_max
        self.parser = GDSParser()
        self.tf_computer = TurningFunction()
        self.normalizer = PolygonNormalizer()
        self.lsh = LSHFamily(m_max=m_max)
        #核心存储数据结构（论文LSH存储核心）
        self.raw_polygons: List[np.ndarray] = []  #存储原始多边形顶点
        self.normalized_polygons: List[np.ndarray] = []  #存储标准化后的多边形
        self.turning_functions: List[Tuple[np.ndarray, np.ndarray]] = []  #存储转向函数（移位后）
        self.mean_reduced_tfs: List[Tuple[np.ndarray, np.ndarray]] = []  #存储均值归约后的转向函数

        #LSH哈希表（双重哈希存储，论文3.2+4.2节）
        self.l1_lsh_table: Dict[str, List[int]] = defaultdict(list)  # random-point-LSH哈希表（L₁）
        self.l2_lsh_table: Dict[str, List[int]] = defaultdict(list)  # discrete-sample-LSH哈希表（L₂）

        #多边形顶点数（快速过滤）
        self.polygon_vertex_counts: List[int] = []

    """
        添加单个多边形到检索库
        返回：多边形在库中的索引
    """
    def add_polygon(self, polygon: np.ndarray) -> int:

        idx = len(self.raw_polygons)
        #存储原始多边形
        self.raw_polygons.append(polygon)
        self.polygon_vertex_counts.append(len(polygon))
        #标准化多边形
        normalized_poly = self.normalizer.normalize(polygon)
        self.normalized_polygons.append(normalized_poly)
        #计算转向函数并垂直移位（论文C.1节优化）
        tf = self.tf_computer.compute(normalized_poly)
        tf_shifted = self.normalizer.vertical_shift(tf)
        self.turning_functions.append(tf_shifted)
        #均值归约（论文3.2节：用于D₁^⊥/D₂^⊥距离）
        tf_reduced = self.tf_computer.mean_reduce(*tf_shifted)
        self.mean_reduced_tfs.append(tf_reduced)
        #生成LSH哈希并存储到哈希表（论文核心存储逻辑）
        l1_hash = self.lsh.random_point_lsh(tf_shifted, num_hashes=100)
        l2_hash = self.lsh.discrete_sample_lsh(tf_reduced, n_samples=200)
        self.l1_lsh_table[l1_hash].append(idx)
        self.l2_lsh_table[l2_hash].append(idx)

        return idx

    """
        从GDS文件添加所有多边形
        返回：添加的多边形数量
    """
    def add_gds_file(self, gds_path: str) -> int:

        with open(gds_path, 'r', encoding='utf-8') as f:
            gds_content = f.read()
        polygons = self.parser.parse(gds_content)
        for poly in polygons:
            self.add_polygon(poly)
        return len(polygons)

    """
        检索相似多边形
        query_polygon: 查询多边形
        distance_type: 距离类型（'L1'/'L2'/'D1'/'D2'，论文定义的四种距离）
        r: 近邻距离阈值
        c: 近似因子（论文LSH参数）
        返回：[(相似多边形索引, 距离), ...]（按距离升序排序）
    """
    def retrieve_similar(self, query_polygon: np.ndarray, distance_type: str = 'D1',
                         r: float = 0.1, c: float = 2.0) -> List[Tuple[int, float]]:
        #预处理查询多边形
        query_normalized = self.normalizer.normalize(query_polygon)
        query_tf = self.tf_computer.compute(query_normalized)
        query_tf_shifted = self.normalizer.vertical_shift(query_tf)
        query_tf_reduced = self.tf_computer.mean_reduce(*query_tf_shifted)
        #生成查询哈希，获取候选集（论文LSH检索核心：哈希碰撞筛选）
        query_l1_hash = self.lsh.random_point_lsh(query_tf_shifted, num_hashes=100)
        query_l2_hash = self.lsh.discrete_sample_lsh(query_tf_reduced, n_samples=200)
        #合并候选集（去重）
        candidates: Set[int] = set()
        candidates.update(self.l1_lsh_table.get(query_l1_hash, []))
        candidates.update(self.l2_lsh_table.get(query_l2_hash, []))
        # 候选集为空时，返回空列表
        if not candidates:
            return []
        #计算真实距离（论文LSH后验证步骤）
        similar_results = []
        for idx in candidates:
            #过滤顶点数差异过大的多边形
            query_vertex_cnt = len(query_polygon)
            candidate_vertex_cnt = self.polygon_vertex_counts[idx]
            if abs(query_vertex_cnt - candidate_vertex_cnt) > max(5, query_vertex_cnt * 0.3):
                continue

            #根据距离类型计算距离
            if distance_type == 'L1':
                dist = self.tf_computer.l1_distance(query_tf_shifted, self.turning_functions[idx])
            elif distance_type == 'L2':
                dist = self.tf_computer.l2_distance(query_tf_reduced, self.mean_reduced_tfs[idx])
            elif distance_type == 'D1':
                #D1距离：考虑水平平移（论文3.3节slide-clone-LSH逻辑简化）
                candidate_tf = self.turning_functions[idx]
                min_dist = float('inf')
                #水平平移候选断点（这个3是如果断点足够多的话就检测3个断点）『这玩意最小的越小的话肯定匹配的越快，但是也可能失真』
                shift_steps = min(3, len(candidate_tf[0]) - 1)
                for s in range(shift_steps):
                    shifted_x = np.roll(candidate_tf[0], s)
                    shifted_y = np.roll(candidate_tf[1], s)
                    dist = self.tf_computer.l1_distance(query_tf_shifted, (shifted_x, shifted_y))
                    min_dist = min(min_dist, dist)
                dist = min_dist
            elif distance_type == 'D2':
                #D2距离：考虑水平平移（论文B.3节逻辑简化）
                candidate_tf = self.mean_reduced_tfs[idx]
                min_dist = float('inf')
                shift_steps = min(3, len(candidate_tf[0]) - 1)
                for s in range(shift_steps):
                    shifted_x = np.roll(candidate_tf[0], s)
                    shifted_y = np.roll(candidate_tf[1], s)
                    dist = self.tf_computer.l2_distance(query_tf_reduced, (shifted_x, shifted_y))
                    min_dist = min(min_dist, dist)
                dist = min_dist
            else:
                raise ValueError(f"不支持的距离类型：{distance_type}")

            #满足近似近邻条件（论文LSH (r, cr)阈值）
            if dist <= c * r:
                similar_results.append((idx, dist))

        #按距离升序排序
        similar_results.sort(key=lambda x: x[1])
        return similar_results

if __name__ == "__main__":
    retrieval = PolygonSimilarRetrieval(m_max=1000)
    gds_file_path = "test1.txt"
    added_cnt = retrieval.add_gds_file(gds_file_path)
    print(f"从GDS文件成功添加{added_cnt}个多边形")

    if added_cnt > 0:
        while True:
            a=int(input())
            query_polygon = retrieval.raw_polygons[a]
            print(f"\n查询目标：第{a}个多边形（顶点数：{len(query_polygon)}）")

            #检索相似多边形（使用D1距离，r=0.2，c=2.0）
            similar_polygons = retrieval.retrieve_similar(
                query_polygon,
                distance_type='D1',
                r=0.2,
                c=2.0
            )

            #输出结果
            print(f"找到{len(similar_polygons)}个相似多边形：")
            for idx, dist in similar_polygons:
                print(f"  - 索引{idx}：距离={dist:.4f}，顶点数={retrieval.polygon_vertex_counts[idx]}")