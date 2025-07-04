"""
可视化模块 - 处理图表绘制和数据可视化
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
from config import Config

class Visualizer:
    """可视化类"""
    
    def __init__(self):
        """初始化可视化器，设置中文显示"""
        plt.rcParams['font.sans-serif'] = Config.PLOT_CONFIG['font_sans_serif']
        plt.rcParams['axes.unicode_minus'] = Config.PLOT_CONFIG['unicode_minus']
        
        # 创建输出目录
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    def plot_roc_curves(self, models, test_x, test_y, save_png=False, save_path=None):
        """
        绘制ROC曲线
        
        参数:
            models (list): 模型列表
            test_x: 测试集特征
            test_y: 测试集标签
            save_png (bool): 是否保存图片
            save_path (str): 保存路径
            
        返回:
            matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 遍历所有模型
        for model_info in models:
            ml_model = model_info["model"]
            label = model_info["label"]
            
            # 预测概率
            test_prob = ml_model.predict_proba(test_x)[:, 1]
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(test_y, test_prob)
            auc = roc_auc_score(test_y, test_prob)
            
            # 绘制ROC曲线
            ax.plot(fpr, tpr, label=f"{label} AUC面积 = {auc:.2f}", linewidth=2)
        
        # 绘制对角线（随机分类器基准线）
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="随机分类器")
        
        # 设置图表属性
        ax.set_xlabel("假正例率 (FPR)", fontsize=12)
        ax.set_ylabel("真正例率 (TPR)", fontsize=12)
        ax.set_title("ROC曲线对比", fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # 保存图片
        if save_png:
            save_path = save_path or Config.ROC_PLOT_PATH
            fig.savefig(
                save_path,
                dpi=Config.PLOT_CONFIG['dpi'],
                bbox_inches=Config.PLOT_CONFIG['bbox_inches'],
                transparent=Config.PLOT_CONFIG['transparent']
            )
            print(f"ROC曲线图已保存至: {save_path}")
        
        return fig
    
    def plot_performance_comparison(self, results, save_png=False, save_path=None):
        """
        绘制性能对比图
        
        参数:
            results (dict): 模型性能结果
            save_png (bool): 是否保存图片
            save_path (str): 保存路径
            
        返回:
            matplotlib.figure.Figure: 图形对象
        """
        models = list(results.keys())
        metrics = ['准确率', '敏感性', '特异性', 'AUC']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 准备数据
        x = range(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[model][i] for model in models]
            ax.bar([pos + width * i for pos in x], values, width, label=metric)
        
        # 设置图表属性
        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('性能指标', fontsize=12)
        ax.set_title('模型性能对比', fontsize=14, fontweight='bold')
        ax.set_xticks([pos + width * 1.5 for pos in x])
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置y轴范围
        ax.set_ylim([0, 1])
        
        # 保存图片
        if save_png:
            save_path = save_path or os.path.join(Config.OUTPUT_DIR, "performance_comparison.png")
            fig.savefig(
                save_path,
                dpi=Config.PLOT_CONFIG['dpi'],
                bbox_inches=Config.PLOT_CONFIG['bbox_inches'],
                transparent=Config.PLOT_CONFIG['transparent']
            )
            print(f"性能对比图已保存至: {save_path}")
        
        return fig
    
    @staticmethod
    def show_plots():
        """显示所有图表"""
        plt.show()
    
    @staticmethod
    def close_all_plots():
        """关闭所有图表"""
        plt.close('all')