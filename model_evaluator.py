"""
模型评估模块
"""

import numpy as np
import time
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn import clone

class ModelEvaluator:
    """模型评估类"""
    
    @staticmethod
    def calculate_performance(ml_model, test_x, test_y, verbose=True):
        """
        计算模型性能指标
        
        参数:
            ml_model: sklearn模型对象
            test_x: 测试集特征
            test_y: 测试集标签
            verbose (bool): 是否打印结果
            
        返回:
            tuple: (accuracy, sensitivity, specificity, auc)
        """
        # 在测试集上预测概率（取正类概率）
        test_prob = ml_model.predict_proba(test_x)[:, 1]
        
        # 在测试集上预测类别
        test_pred = ml_model.predict(test_x)
        
        # 计算性能指标
        accuracy = accuracy_score(test_y, test_pred)
        sensitivity = recall_score(test_y, test_pred)
        specificity = recall_score(test_y, test_pred, pos_label=0)
        auc = roc_auc_score(test_y, test_prob)
        
        if verbose:
            print(f"准确率: {accuracy:.2f}")
            print(f"敏感性: {sensitivity:.2f}")
            print(f"特异性: {specificity:.2f}")
            print(f"AUC: {auc:.2f}")
        
        return accuracy, sensitivity, specificity, auc
    
    @staticmethod
    def train_and_validate(ml_model, name, splits, verbose=True):
        """
        训练模型并验证性能
        
        参数:
            ml_model: sklearn模型对象
            name (str): 模型名称
            splits (list): [train_x, test_x, train_y, test_y]
            verbose (bool): 是否打印结果
            
        返回:
            tuple: (accuracy, sensitivity, specificity, auc)
        """
        train_x, test_x, train_y, test_y = splits
        
        # 训练模型
        ml_model.fit(train_x, train_y)
        
        # 评估性能
        return ModelEvaluator.calculate_performance(ml_model, test_x, test_y, verbose)
    
    @staticmethod
    def cross_validation(ml_model, df, n_folds=5, verbose=False):
        """
        交叉验证
        
        参数:
            ml_model: sklearn模型对象
            df: pandas DataFrame
            n_folds (int): 交叉验证折数
            verbose (bool): 是否打印详细信息
            
        返回:
            tuple: (acc_per_fold, sens_per_fold, spec_per_fold, auc_per_fold)
        """
        t0 = time.time()
        
        # 初始化KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=12345)
        
        # 存储每一折的结果
        acc_per_fold = []
        sens_per_fold = []
        spec_per_fold = []
        auc_per_fold = []
        
        # 执行交叉验证
        for train_index, test_index in kf.split(df):
            # 克隆模型
            fold_model = clone(ml_model)
            
            # 准备训练数据
            train_x = df.iloc[train_index].fp.tolist()
            train_y = df.iloc[train_index].active.tolist()
            
            # 训练模型
            fold_model.fit(train_x, train_y)
            
            # 准备测试数据
            test_x = df.iloc[test_index].fp.tolist()
            test_y = df.iloc[test_index].active.tolist()
            
            # 计算性能
            accuracy, sens, spec, auc = ModelEvaluator.calculate_performance(
                fold_model, test_x, test_y, verbose
            )
            
            # 保存结果
            acc_per_fold.append(accuracy)
            sens_per_fold.append(sens)
            spec_per_fold.append(spec)
            auc_per_fold.append(auc)
        
        # 打印统计结果
        print(
            f"平均准确率: {np.mean(acc_per_fold):.2f} \t"
            f"标准差: {np.std(acc_per_fold):.2f} \n"
            f"平均敏感性: {np.mean(sens_per_fold):.2f} \t"
            f"标准差: {np.std(sens_per_fold):.2f} \n"
            f"平均特异性: {np.mean(spec_per_fold):.2f} \t"
            f"标准差: {np.std(spec_per_fold):.2f} \n"
            f"平均AUC: {np.mean(auc_per_fold):.2f} \t"
            f"标准差: {np.std(auc_per_fold):.2f} \n"
            f"耗时: {time.time() - t0:.2f}秒\n"
        )
        
        return acc_per_fold, sens_per_fold, spec_per_fold, auc_per_fold