"""
数据加载和预处理模块
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config

class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, data_path=None):
        """
        初始化数据加载器
        
        参数:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path or Config.DATA_PATH
        self.df = None
        
    def str_to_array(self, fp_str):
        """
        将字符串格式的指纹转换为numpy数组
        
        参数:
            fp_str (str): 字符串格式的指纹
            
        返回:
            np.array: 转换后的数组
        """
        # 去掉中括号和换行符，并按空格分割
        fp_list = fp_str.replace('[', '').replace(']', '').replace('\n', '').split()
        return np.array(list(map(int, fp_list)))
    
    def load_and_preprocess(self):
        """
        加载并预处理数据
        
        返回:
            pd.DataFrame: 预处理后的数据框
        """
        # 读取CSV文件
        self.df = pd.read_csv(self.data_path)
        
        # 转换指纹列
        self.df['fp'] = self.df['fp'].apply(self.str_to_array)
        
        print(f"数据加载完成，共有 {len(self.df)} 条记录")
        print(self.df.head())
        
        return self.df
    
    def prepare_model_data(self):
        """
        准备模型输入数据
        
        返回:
            tuple: (fingerprints, labels) 指纹列表和标签列表
        """
        if self.df is None:
            raise ValueError("请先调用 load_and_preprocess() 方法")
            
        # 准备模型输入的特征数据
        fingerprints = self.df.fp.tolist()
        
        # 准备模型输入的标签数据
        labels = self.df.active.tolist()
        
        print(f"前5个标签: {labels[:5]}")
        
        return fingerprints, labels
    
    def split_data(self, fingerprints, labels, test_size=None, random_state=None):
        """
        拆分训练集和测试集
        
        参数:
            fingerprints (list): 指纹列表
            labels (list): 标签列表
            test_size (float): 测试集比例
            random_state (int): 随机种子
            
        返回:
            tuple: (train_x, test_x, train_y, test_y)
        """
        test_size = test_size or Config.TEST_SIZE
        random_state = random_state or Config.RANDOM_STATE
        
        train_x, test_x, train_y, test_y = train_test_split(
            fingerprints,
            labels,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"训练数据大小: {len(train_x)}")
        print(f"测试数据大小: {len(test_x)}")
        
        return train_x, test_x, train_y, test_y