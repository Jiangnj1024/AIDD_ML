"""
配置文件 - 存储项目的所有配置参数
"""

import os

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