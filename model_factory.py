"""
模型工厂模块 - 创建和管理不同类型的机器学习模型
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from config import Config

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_random_forest(params=None):
        """
        创建随机森林模型
        
        参数:
            params (dict): 模型参数，如果为None则使用默认配置
            
        返回:
            RandomForestClassifier: 随机森林模型实例
        """
        if params is None:
            params = Config.RF_PARAMS
        return RandomForestClassifier(**params)
    
    @staticmethod
    def create_svm(params=None):
        """
        创建支持向量机模型
        
        参数:
            params (dict): 模型参数，如果为None则使用默认配置
            
        返回:
            SVC: SVM模型实例
        """
        if params is None:
            params = Config.SVM_PARAMS
        return svm.SVC(**params)
    
    @staticmethod
    def create_mlp(params=None):
        """
        创建多层感知机模型
        
        参数:
            params (dict): 模型参数，如果为None则使用默认配置
            
        返回:
            MLPClassifier: MLP模型实例
        """
        if params is None:
            params = Config.ANN_PARAMS
        return MLPClassifier(**params)
    
    @staticmethod
    def create_all_models():
        """
        创建所有模型并返回模型列表
        
        返回:
            list: 包含所有模型的字典列表
        """
        models = [
            {
                "label": "Model_RF",
                "model": ModelFactory.create_random_forest(),
                "name": "Random Forest"
            },
            {
                "label": "Model_SVM", 
                "model": ModelFactory.create_svm(),
                "name": "Support Vector Machine"
            },
            {
                "label": "Model_ANN",
                "model": ModelFactory.create_mlp(),
                "name": "Artificial Neural Network"
            }
        ]
        
        return models
    
    @staticmethod
    def get_model_by_name(name):
        """
        根据名称获取特定模型
        
        参数:
            name (str): 模型名称 ('rf', 'svm', 'mlp')
            
        返回:
            sklearn model: 对应的模型实例
        """
        name = name.lower()
        if name in ['rf', 'random_forest']:
            return ModelFactory.create_random_forest()
        elif name in ['svm', 'support_vector_machine']:
            return ModelFactory.create_svm()
        elif name in ['mlp', 'ann', 'neural_network']:
            return ModelFactory.create_mlp()
        else:
            raise ValueError(f"未知的模型名称: {name}")