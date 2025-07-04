"""
配置文件 - 存储项目的所有配置参数
"""

import os
import pandas as pd




class Config:
    """项目配置类"""
    
    # 数据配置
    DATA_PATH = "/home/ubuntu/Molecular_features/EGFR_compounds_maccs_fingerprints_processed.csv"
    
    # 模型配置
    RANDOM_STATE = 12345
    TEST_SIZE = 0.2
    N_FOLDS = 3
    
    # 随机森林参数
    RF_PARAMS = {
        "n_estimators": 100,
        "criterion": "entropy",
        "random_state": RANDOM_STATE
    }
    
    # SVM参数
    SVM_PARAMS = {
        "kernel": "rbf",
        "C": 1,
        "gamma": 0.1,
        "probability": True
    }
    
    # ANN参数
    ANN_PARAMS = {
        "hidden_layer_sizes": (5, 3),
        "random_state": RANDOM_STATE
    }
    
    # 梯度提升参数
    GB_PARAMS = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": RANDOM_STATE
    }
    
    # AdaBoost参数
    ADA_PARAMS = {
        "n_estimators": 50,
        "learning_rate": 1.0,
        "random_state": RANDOM_STATE
    }
    
    # 逻辑回归参数
    LR_PARAMS = {
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
        "solver": "lbfgs"
    }
    
    # K近邻参数
    KNN_PARAMS = {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto"
    }
    
    # 朴素贝叶斯参数
    NB_PARAMS = {
        # 高斯朴素贝叶斯通常不需要太多参数
    }
    
    # 决策树参数
    DT_PARAMS = {
        "criterion": "entropy",
        "random_state": RANDOM_STATE,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
    
    # 线性判别分析参数
    LDA_PARAMS = {
        "solver": "svd"
    }
    
    # 极端随机树参数
    ET_PARAMS = {
        "n_estimators": 100,
        "criterion": "entropy",
        "random_state": RANDOM_STATE,
        "max_depth": 10
    }
    
    # 图表配置
    PLOT_CONFIG = {
        "font_sans_serif": ['SimHei', 'FangSong'],
        "unicode_minus": False,
        "dpi": 300,
        "bbox_inches": "tight",
        "transparent": True
    }
    
    # 输出配置
    OUTPUT_DIR = "./output"
    ROC_PLOT_PATH = "./output/roc_auc.png"