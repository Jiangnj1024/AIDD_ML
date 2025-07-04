"""
工具函数模块 - 提供通用的辅助功能
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

class Utils:
    """工具类"""
    
    @staticmethod
    def ensure_dir(directory: str) -> None:
        """
        确保目录存在，如果不存在则创建
        
        参数:
            directory (str): 目录路径
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")
    
    @staticmethod
    def save_model(model, filepath: str) -> None:
        """
        保存模型到文件
        
        参数:
            model: 要保存的模型对象
            filepath (str): 保存路径
        """
        Utils.ensure_dir(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存至: {filepath}")
    
    @staticmethod
    def load_model(filepath: str):
        """
        从文件加载模型
        
        参数:
            filepath (str): 模型文件路径
            
        返回:
            加载的模型对象
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"模型已从 {filepath} 加载")
        return model
    
    @staticmethod
    def save_results(results: Dict, filepath: str) -> None:
        """
        保存结果到JSON文件
        
        参数:
            results (dict): 要保存的结果字典
            filepath (str): 保存路径
        """
        Utils.ensure_dir(os.path.dirname(filepath))
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, tuple):
                serializable_results[key] = list(value)
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存至: {filepath}")
    
    @staticmethod
    def load_results(filepath: str) -> Dict:
        """
        从JSON文件加载结果
        
        参数:
            filepath (str): 结果文件路径
            
        返回:
            dict: 加载的结果字典
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"结果已从 {filepath} 加载")
        return results
    
    @staticmethod
    def get_timestamp() -> str:
        """
        获取当前时间戳字符串
        
        返回:
            str: 格式化的时间戳
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def print_separator(title: str = "", char: str = "=", length: int = 60) -> None:
        """
        打印分隔线
        
        参数:
            title (str): 标题文本
            char (str): 分隔符字符
            length (int): 分隔线长度
        """
        if title:
            print(f"\n{char * length}")
            print(f"{title:^{length}}")
            print(f"{char * length}")
        else:
            print(f"{char * length}")
    
    @staticmethod
    def calculate_statistics(data: List[float]) -> Dict[str, float]:
        """
        计算数据的统计信息
        
        参数:
            data (list): 数值列表
            
        返回:
            dict: 包含统计信息的字典
        """
        if not data:
            return {}
        
        data_array = np.array(data)
        return {
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'median': np.median(data_array),
            'count': len(data_array)
        }
    
    @staticmethod
    def format_performance_table(results: Dict[str, tuple]) -> str:
        """
        格式化性能结果表格
        
        参数:
            results (dict): 性能结果字典
            
        返回:
            str: 格式化的表格字符串
        """
        if not results:
            return "无结果数据"
        
        # 表头
        table = f"{'模型':<15} {'准确率':<8} {'敏感性':<8} {'特异性':<8} {'AUC':<8}\n"
        table += "-" * 55 + "\n"
        
        # 数据行
        for model_name, (acc, sens, spec, auc) in results.items():
            table += f"{model_name:<15} {acc:<8.3f} {sens:<8.3f} {spec:<8.3f} {auc:<8.3f}\n"
        
        return table
    
    @staticmethod
    def validate_data_format(df: pd.DataFrame) -> bool:
        """
        验证数据格式是否正确
        
        参数:
            df (pd.DataFrame): 要验证的数据框
            
        返回:
            bool: 验证结果
        """
        required_columns = ['fp', 'active']
        
        # 检查必需列是否存在
        for col in required_columns:
            if col not in df.columns:
                print(f"错误: 缺少必需列 '{col}'")
                return False
        
        # 检查活性列的值
        unique_active = df['active'].unique()
        if not set(unique_active).issubset({0, 1}):
            print(f"错误: 活性列应只包含0和1，当前包含: {unique_active}")
            return False
        
        # 检查指纹列是否为字符串
        if not df['fp'].dtype == 'object':
            print("错误: 指纹列应为字符串类型")
            return False
        
        print("数据格式验证通过")
        return True
    
    @staticmethod
    def get_memory_usage() -> str:
        """
        获取当前内存使用情况
        
        返回:
            str: 内存使用信息
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return f"内存使用: {memory_info.rss / 1024 / 1024:.2f} MB"
        except ImportError:
            return "需要安装psutil包来获取内存信息"
    
    @staticmethod
    def log_execution_time(func):
        """
        装饰器：记录函数执行时间
        
        参数:
            func: 要装饰的函数
            
        返回:
            装饰后的函数
        """
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            print(f"{func.__name__} 执行时间: {execution_time:.2f} 秒")
            return result
        return wrapper