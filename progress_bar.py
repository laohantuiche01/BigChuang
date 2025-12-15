import sys
import time


class ProgressBar:
    """进度条工具类：支持自定义长度、描述文本，动态刷新显示"""

    def __init__(self, total: int, desc: str = "处理中", length: int = 50):
        self.total = total  # 总任务量
        self.desc = desc  # 任务描述
        self.length = length  # 进度条显示长度（字符数）
        self.current = 0  # 当前完成量
        self.start_time = time.time()  # 开始时间

    def update(self, current: int = None) -> None:
        """更新进度：支持手动指定当前进度或自动递增"""
        if current is not None:
            self.current = min(current, self.total)
        else:
            self.current += 1
        self._display()

    def _display(self) -> None:
        """动态显示进度条"""
        progress = self.current / self.total if self.total > 0 else 1.0
        filled_length = int(self.length * progress)
        # 进度条字符：已完成用'█'，未完成用' '
        bar = '█' * filled_length + ' ' * (self.length - filled_length)
        # 计算耗时和预估剩余时间
        elapsed_time = time.time() - self.start_time
        remaining_time = (elapsed_time / progress) - elapsed_time if progress > 0 else 0
        # 格式化输出：进度条+百分比+耗时+剩余时间
        percent = progress * 100
        sys.stdout.write(
            f"\r{self.desc}: [{bar}] {percent:.1f}% | 耗时: {elapsed_time:.1f}s | 剩余: {remaining_time:.1f}s"
        )
        sys.stdout.flush()
        # 任务完成后换行
        if self.current == self.total:
            print()

    def finish(self) -> None:
        """强制标记任务完成"""
        self.current = self.total
        self._display()
