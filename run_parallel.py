import subprocess
import threading
import time
import itertools
from typing import List, Tuple
import queue
from contextlib import contextmanager
import logging
from datetime import datetime
import os

# GPU池管理器（从之前的代码复制过来）
class GPUPool:
    """GPU资源池管理器"""
    
    def __init__(self, gpu_ids: list = None):
        if gpu_ids is None:
            gpu_ids = list(range(7))  # cuda0-cuda6
        
        self.gpu_ids = gpu_ids
        self.available_gpus = queue.Queue()
        self.used_gpus = set()
        self.lock = threading.Lock()
        
        for gpu_id in self.gpu_ids:
            self.available_gpus.put(gpu_id)
    
    def get_device(self, timeout=None):
        try:
            gpu_id = self.available_gpus.get(timeout=timeout)
            with self.lock:
                self.used_gpus.add(gpu_id)
            device = f"cuda:{gpu_id}"
            return device
        except queue.Empty:
            raise TimeoutError("获取GPU设备超时")
    
    def release_device(self, device: str):
        gpu_id = int(device.split(":")[1])
        with self.lock:
            if gpu_id in self.used_gpus:
                self.used_gpus.remove(gpu_id)
        self.available_gpus.put(gpu_id)
    
    @contextmanager
    def acquire_device(self, timeout=None):
        device = self.get_device(timeout=timeout)
        try:
            yield device
        finally:
            self.release_device(device)
    
    def get_status(self):
        with self.lock:
            return {
                "total_gpus": len(self.gpu_ids),
                "available_gpus": self.available_gpus.qsize(),
                "used_gpus": len(self.used_gpus),
                "used_gpu_ids": sorted(list(self.used_gpus))
            }


class TaskRunner:
    """任务执行器"""
    
    def __init__(self, gpu_pool: GPUPool, result_dir: str = "./test_results_new", max_samples: int = 1000):
        self.gpu_pool = gpu_pool
        self.result_dir = result_dir
        self.max_samples = max_samples
        self.completed_tasks = []
        self.failed_tasks = []
        self.lock = threading.Lock()
        
        # 设置日志
        self.setup_logging()
        
        # 确保结果目录存在
        os.makedirs(result_dir, exist_ok=True)
    
    def setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('task_execution.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_task_combinations(self):
        """生成所有任务组合"""
        datasets = ['iwslt', 'eq', 'nq']
        models = ['google/gemma-2-2b', 'gpt2']
        decode_modes = ['penalty', 'neuron', 'sae_penalty', 'neuron_penalty']
        
        combinations = list(itertools.product(datasets, models, decode_modes))
        self.logger.info(f"生成了 {len(combinations)} 个任务组合")
        return combinations
    
    def create_command(self, dataset: str, model: str, decode_mode: str, device: str) -> List[str]:
        """创建执行命令"""
        cmd = [
            'python', '-m', 'src.evaluation.run_dataset',
            f'--dataset={dataset}',
            f'--model={model}',
            f'--result_dir={self.result_dir}',
            f'--decode-mode={decode_mode}',
            f'--max-samples={self.max_samples}',
            f'--device={device}'
        ]
        return cmd
    
    def execute_task(self, task_id: int, dataset: str, model: str, decode_mode: str):
        """执行单个任务"""
        task_name = f"Task-{task_id:03d}_{dataset}_{model.replace('/', '_')}_{decode_mode}"
        
        with self.gpu_pool.acquire_device() as device:
            self.logger.info(f"[{task_name}] 开始执行，使用设备: {device}")
            
            cmd = self.create_command(dataset, model, decode_mode, device)
            
            start_time = time.time()
            
            try:
                # 执行命令
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600 * 10  # 10小时超时
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                if result.returncode == 0:
                    self.logger.info(f"[{task_name}] 执行成功，耗时: {duration:.2f}秒")
                    with self.lock:
                        self.completed_tasks.append({
                            'task_id': task_id,
                            'task_name': task_name,
                            'dataset': dataset,
                            'model': model,
                            'decode_mode': decode_mode,
                            'device': device,
                            'duration': duration,
                            'status': 'success'
                        })
                else:
                    self.logger.error(f"[{task_name}] 执行失败，返回码: {result.returncode}")
                    self.logger.error(f"[{task_name}] 错误输出: {result.stderr}")
                    with self.lock:
                        self.failed_tasks.append({
                            'task_id': task_id,
                            'task_name': task_name,
                            'dataset': dataset,
                            'model': model,
                            'decode_mode': decode_mode,
                            'device': device,
                            'duration': duration,
                            'status': 'failed',
                            'error': result.stderr
                        })
            
            except subprocess.TimeoutExpired:
                self.logger.error(f"[{task_name}] 执行超时")
                with self.lock:
                    self.failed_tasks.append({
                        'task_id': task_id,
                        'task_name': task_name,
                        'dataset': dataset,
                        'model': model,
                        'decode_mode': decode_mode,
                        'device': device,
                        'duration': 3600,
                        'status': 'timeout',
                        'error': 'Task timeout after 1 hour'
                    })
            
            except Exception as e:
                self.logger.error(f"[{task_name}] 执行异常: {str(e)}")
                with self.lock:
                    self.failed_tasks.append({
                        'task_id': task_id,
                        'task_name': task_name,
                        'dataset': dataset,
                        'model': model,
                        'decode_mode': decode_mode,
                        'device': device,
                        'duration': time.time() - start_time,
                        'status': 'error',
                        'error': str(e)
                    })
    
    def run_all_tasks(self, max_concurrent_tasks: int = None):
        """运行所有任务"""
        if max_concurrent_tasks is None:
            max_concurrent_tasks = len(self.gpu_pool.gpu_ids)
        
        combinations = self.generate_task_combinations()
        total_tasks = len(combinations)
        
        self.logger.info(f"开始执行 {total_tasks} 个任务，最大并发数: {max_concurrent_tasks}")
        
        threads = []
        start_time = time.time()
        
        # 创建线程池
        for task_id, (dataset, model, decode_mode) in enumerate(combinations):
            thread = threading.Thread(
                target=self.execute_task,
                args=(task_id, dataset, model, decode_mode),
                name=f"Task-{task_id}"
            )
            threads.append(thread)
        
        # 启动线程（GPU池会自动控制并发数）
        for thread in threads:
            thread.start()
            time.sleep(0.1)  # 稍微错开启动时间
        
        # 定期打印进度
        progress_thread = threading.Thread(target=self.print_progress, args=(total_tasks,))
        progress_thread.daemon = True
        progress_thread.start()
        
        # 等待所有任务完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 打印最终结果
        self.print_summary(total_duration)
    
    def print_progress(self, total_tasks: int):
        """定期打印进度"""
        while True:
            time.sleep(30)  # 每30秒打印一次
            with self.lock:
                completed = len(self.completed_tasks)
                failed = len(self.failed_tasks)
                finished = completed + failed
                
                if finished >= total_tasks:
                    break
                
                progress = (finished / total_tasks) * 100
                gpu_status = self.gpu_pool.get_status()
                
                self.logger.info(
                    f"进度: {finished}/{total_tasks} ({progress:.1f}%) - "
                    f"成功: {completed}, 失败: {failed} - "
                    f"GPU使用: {gpu_status['used_gpus']}/{gpu_status['total_gpus']}"
                )
    
    def print_summary(self, total_duration: float):
        """打印执行总结"""
        with self.lock:
            total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
            success_rate = (len(self.completed_tasks) / total_tasks) * 100 if total_tasks > 0 else 0
            
            self.logger.info("=" * 60)
            self.logger.info("任务执行总结")
            self.logger.info("=" * 60)
            self.logger.info(f"总任务数: {total_tasks}")
            self.logger.info(f"成功任务: {len(self.completed_tasks)}")
            self.logger.info(f"失败任务: {len(self.failed_tasks)}")
            self.logger.info(f"成功率: {success_rate:.1f}%")
            self.logger.info(f"总耗时: {total_duration:.2f}秒 ({total_duration/3600:.2f}小时)")
            
            if self.failed_tasks:
                self.logger.info("\n失败任务列表:")
                for task in self.failed_tasks:
                    self.logger.info(f"  - {task['task_name']}: {task['status']}")
            
            # 保存详细结果到文件
            self.save_results()
    
    def save_results(self):
        """保存结果到文件"""
        import json
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'summary': {
                'total': len(self.completed_tasks) + len(self.failed_tasks),
                'success': len(self.completed_tasks),
                'failed': len(self.failed_tasks)
            }
        }
        
        result_file = os.path.join(self.result_dir, 'execution_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"详细结果已保存到: {result_file}")


def main():
    """主函数"""
    # 创建GPU池
    gpu_pool = GPUPool()
    
    # 创建任务执行器
    runner = TaskRunner(
        gpu_pool=gpu_pool,
        result_dir="./test_results_new",
        max_samples=1000
    )
    
    # 运行所有任务
    runner.run_all_tasks()


if __name__ == "__main__":
    main()
