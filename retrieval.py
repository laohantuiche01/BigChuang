from queue import Queue
import numpy as np
import re
import hashlib
from scipy.integrate import quad, simpson
from typing import List, Tuple, Dict, Set
from database import PolygonDatabase
from progress_bar import ProgressBar  # 导入进度条类
import threading


class DistanceWorker(threading.Thread):
    def __init__(self, queue, result_list, lock, retrieval_obj, query_data, distance_type, shift_steps, r, c):
        super().__init__()
        self.queue = queue
        self.result_list = result_list
        self.lock = lock
        self.retrieval = retrieval_obj
        self.query_data = query_data
        self.distance_type = distance_type
        self.shift_steps = shift_steps
        self.r = r
        self.c = c
        self.daemon = True

    def run(self):
        while not self.queue.empty():
            idx = self.queue.get()
            try:
                # 获取查询数据
                query_tf_shifted, query_tf_reduced, query_vertex_cnt = self.query_data

                # 获取候选多边形数据
                _, candidate_tf, candidate_tf_reduced, candidate_vertex_cnt = self.retrieval.db.get_polygon_data(idx)

                # 顶点数动态过滤（优化版本）
                vertex_diff_threshold = max(3, min(5, query_vertex_cnt * 0.3))  # 自适应阈值
                if abs(query_vertex_cnt - candidate_vertex_cnt) > vertex_diff_threshold:
                    continue

                # 计算距离
                if self.distance_type == 'L1':
                    dist = self.retrieval.tf_computer.l1_distance(query_tf_shifted, candidate_tf)
                elif self.distance_type == 'L2':
                    dist = self.retrieval.tf_computer.l2_distance(query_tf_reduced, candidate_tf_reduced)
                elif self.distance_type == 'D1':
                    dist = self.retrieval.tf_computer.d1_distance(query_tf_shifted, candidate_tf, self.shift_steps)
                elif self.distance_type == 'D2':
                    dist = self.retrieval.tf_computer.d2_distance(query_tf_reduced, candidate_tf_reduced,
                                                                  self.shift_steps)
                else:
                    raise ValueError(f"不支持的距离类型：{self.distance_type}")

                # 结果过滤与保存
                if dist <= self.c * self.r:
                    with self.lock:
                        self.result_list.append((idx, dist))
            finally:
                self.queue.task_done()


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
        edges = np.vstack([edges, polygon[0] - polygon[-1]])
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        edge_lengths = np.linalg.norm(edges, axis=1)
        total_length = edge_lengths.sum()
        # 归一化周长断点（增加密度，提升匹配精度）
        x = np.cumsum(edge_lengths / total_length)[:-1]
        x = np.insert(x, 0, 0.0)
        # 累积角度
        cumulative_angles = np.cumsum(angles)
        # 将累积角度映射到[-π, π]区间，减少数值差异
        cumulative_angles = (cumulative_angles + np.pi) % (2 * np.pi) - np.pi
        y = np.insert(cumulative_angles, 0, 0.0)
        return x, y

    @staticmethod
    def mean_reduce(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """均值归约：优化积分计算，提升旋转抵消效果"""
        y = y[1:]
        y_mean = np.trapz(y, x)
        y_reduced = y - y_mean
        y_reduced = y_reduced / (np.max(np.abs(y_reduced)) + 1e-6)  # 映射到[-1,1]
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
        distance, _ = quad(integrand, 0, 1, limit=2500)
        return distance

    @staticmethod
    def l2_distance(tf1: Tuple[np.ndarray, np.ndarray], tf2: Tuple[np.ndarray, np.ndarray]) -> float:
        x1, y1 = tf1
        x2, y2 = tf2
        # 合并并去重，确保覆盖所有关键断点
        all_x = np.unique(np.concatenate([x1, x2]))
        # 排序（保证积分顺序）
        all_x = np.sort(all_x)
        # 过滤无效值（确保在[0,1]区间）
        all_x = all_x[(all_x >= 0) & (all_x <= 1)]
        # 若节点过少，补充采样（提升精度）
        if len(all_x) < 100:
            all_x = np.linspace(0, 1, 200)  # 兜底：生成200个均匀节点

        # 对两个TF在统一节点上插值（线性插值，保证连续性）
        def interpolate(x_target, x_source, y_source):
            """线性插值，适配离散点"""
            if len(x_source) == 1:
                return np.full_like(x_target, y_source[0])
            # 线性插值（避免超出边界）
            y_interp = np.interp(x_target, x_source, y_source, left=y_source[0], right=y_source[-1])
            return y_interp

        # 得到统一节点上的y值
        y1_interp = interpolate(all_x, x1, y1)
        y2_interp = interpolate(all_x, x2, y2)

        integrand = (y1_interp - y2_interp) ** 2
        if len(all_x) >= 2:
            distance_sq = simpson(integrand, x=all_x)
        else:
            distance_sq = np.trapz(integrand, x=all_x)
        return np.sqrt(max(distance_sq, 1e-12))  # 防止开方出现NaN

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
                    shift_steps: int = 8) -> float:
        """D₂距离：优化平移策略，提升旋转图形匹配概率"""
        min_dist = float('inf')
        shift_range = list(range(-shift_steps // 2, shift_steps // 2))  # 例：steps=8→[-4,-3,-2,-1,0,1,2,3]
        for s in shift_range:
            if s == 0:
                shifted_x, shifted_y = tf_c[0], tf_c[1]
            else:
                # 循环平移+线性插值（避免平移后断点突变）
                shifted_x = np.roll(tf_c[0], s)
                shifted_y = np.roll(tf_c[1], s)
                # 对平移后的断点进行平滑（消除跳变）
                if abs(s) > 0:
                    # 取首尾各2个点进行平均平滑
                    smooth_window = min(2, len(shifted_y) // 5)
                    if smooth_window > 0:
                        shifted_y[:smooth_window] = np.mean(shifted_y[:2 * smooth_window])
                        shifted_y[-smooth_window:] = np.mean(shifted_y[-2 * smooth_window:])
            # 计算优化后的L2距离
            dist = TurningFunction.l2_distance(tf_q, (shifted_x, shifted_y))
            if dist < min_dist:
                min_dist = dist
                # 若距离已足够小，直接返回
                if min_dist < 1e-3:
                    return min_dist
        return min_dist


"""LSH哈希（论文1核心结构）"""
class LSHFamily:
    def __init__(self, m_max: int = 1000):
        self.m_max = m_max
        self.a_m = -((m_max // 2) - 1) * np.pi
        self.b_m = ((m_max // 2) + 3) * np.pi
        self.span_m = (self.b_m - self.a_m) / 2  # 范围优化

    def random_point_lsh(self, tf: Tuple[np.ndarray, np.ndarray], num_hashes: int = 200) -> str:  # 增加哈希位数
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

    def discrete_sample_lsh(self, tf: Tuple[np.ndarray, np.ndarray], n_samples: int = 500, segments: int = 5) -> List[
        str]:  # 分段哈希
        """discrete-sample-LSH（L₂适配）：返回分段哈希列表"""
        f_x, f_y = tf
        sample_x = np.linspace(0, 1, n_samples)
        sample_y = []
        for x in sample_x:
            idx = np.searchsorted(f_x, x) - 1
            sample_y.append(f_y[idx] if idx >= 0 else f_y[0])
        sample_y = np.array(sample_y)
        vec = sample_y / np.sqrt(n_samples)

        # 分段计算哈希
        segment_size = n_samples // segments
        hashes = []
        for i in range(segments):
            segment = vec[i * segment_size: (i + 1) * segment_size]
            seg_str = ','.join([f"{v:.6f}" for v in segment])
            hashes.append(hashlib.sha256(seg_str.encode()).hexdigest())
        return hashes


"""多边形标准化"""
class PolygonNormalizer:
    @staticmethod
    def normalize(polygon: np.ndarray) -> np.ndarray:
        # 平移
        centroid = np.mean(polygon, axis=0)
        polygon_translated = polygon - centroid
        # 缩放
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
        l1_hash = self.lsh.random_point_lsh(tf_shifted, num_hashes=500)  # 增加哈希位数
        l2_hashes = self.lsh.discrete_sample_lsh(tf_reduced, n_samples=500, segments=2)  # 分段哈希
        return self.db.add_polygon(polygon, normalized_poly, tf_shifted, tf_reduced, l1_hash, l2_hashes)

    def add_gds_file(self, gds_path: str) -> int:
        """从GDS文件批量添加多边形（带进度条）"""
        print(f"正在读取GDS文件：{gds_path}")
        with open(gds_path, 'r', encoding='utf-8') as f:
            gds_content = f.read()
        # 解析GDS得到多边形列表
        polygons = self.parser.parse(gds_content)
        if not polygons:
            print("未解析到有效多边形")
            return 0

        progress = ProgressBar(total=len(polygons), desc="添加多边形到数据库")
        for poly in polygons:
            self.add_polygon(poly)
            progress.update()
        progress.finish()

        #self.db._save_db()
        return len(polygons)

    def retrieve_similar(self, query_polygon: np.ndarray,
                         distance_type: str = 'D2',
                         r: float = 0.3,  # 放松阈值适配遮挡
                         c: float = 2.5,  # 扩大近似因子，提升遮挡召回
                         shift_steps: int = 8,
                         max_threads: int = 16) -> List[Tuple[int, float]]:
        # 查询预处理
        query_normalized = self.normalizer.normalize(query_polygon)
        query_tf = self.tf_computer.compute(query_normalized)
        query_tf_shifted = self.normalizer.vertical_shift(query_tf)
        query_tf_reduced = self.tf_computer.mean_reduce(*query_tf)
        query_vertex_cnt = len(query_polygon)

        query_data = (query_tf_shifted, query_tf_reduced, query_vertex_cnt)

        # 获取候选集
        query_l1_hash = self.lsh.random_point_lsh(query_tf_shifted, num_hashes=0)
        query_l2_hashes = self.lsh.discrete_sample_lsh(query_tf_reduced, n_samples=0, segments=2)
        candidates = self.db.get_candidates(query_l1_hash, query_l2_hashes, distance_type)
        if not candidates:
            print("未找到候选多边形")
            return []
        candidates = list(candidates)
        print(f"找到{len(candidates)}个候选多边形")

        queue = Queue()
        for idx in candidates:
            queue.put(idx)

        result_list = []
        lock = threading.Lock()
        threads = []
        for _ in range(min(max_threads, len(candidates))):
            worker = DistanceWorker(
                queue, result_list, lock, self, query_data,
                distance_type, shift_steps, r, c
            )
            threads.append(worker)
            worker.start()
        queue.join()
        result_list.sort(key=lambda x: x[1])
        return result_list


if __name__ == "__main__":
    retrieval = PolygonSimilarRetrieval(m_max=1000)
    gds_path = "test3.txt"
    #added_cnt = 48755
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
                distance_type='L2',
                r=1.0,
                c=3.5,
                shift_steps=3
            )

            print(f"\n找到{len(similar)}个相似多边形：")
            for idx, dist in similar:
                _, _, _, vertex_cnt = retrieval.db.get_polygon_data(idx)
                print(f"  - 索引{idx}：距离={dist:.4f}，顶点数={vertex_cnt}")
