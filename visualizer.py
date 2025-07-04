"""
可视化模块 - 处理图表绘制和数据可视化 (Nature期刊风格)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import os
from config import Config

class Visualizer:
    """可视化类 - Nature期刊风格"""
    
    def __init__(self):
        """初始化可视化器，设置Nature期刊风格"""
        # Nature期刊风格配置
        self.nature_style = {
            'figure.figsize': (3.5, 2.8),  # Nature单栏宽度
            'font.size': 8,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.linewidth': 0.5,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.labelsize': 8,
            'axes.titlesize': 9,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,
            'legend.fontsize': 7,
            'legend.frameon': False,
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.0,
            'patch.linewidth': 0.5,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        }
        
        # 应用Nature风格
        plt.rcParams.update(self.nature_style)
        
        # Nature期刊常用颜色方案
        self.nature_colors = [
            '#1f77b4',  # 蓝色
            '#ff7f0e',  # 橙色
            '#2ca02c',  # 绿色
            '#d62728',  # 红色
            '#9467bd',  # 紫色
            '#8c564b',  # 棕色
            '#e377c2',  # 粉色
            '#7f7f7f',  # 灰色
            '#bcbd22',  # 黄绿色
            '#17becf'   # 青色
        ]
        
        # 创建输出目录
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    def plot_roc_curves(self, models, test_x, test_y, save_png=False, save_path=None):
        """
        绘制ROC曲线 - Nature风格
        
        参数:
            models (list): 模型列表
            test_x: 测试集特征
            test_y: 测试集标签
            save_png (bool): 是否保存图片
            save_path (str): 保存路径
            
        返回:
            matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        # 遍历所有模型
        for i, model_info in enumerate(models):
            ml_model = model_info["model"]
            label = model_info["label"]
            color = self.nature_colors[i % len(self.nature_colors)]
            
            # 预测概率
            test_prob = ml_model.predict_proba(test_x)[:, 1]
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(test_y, test_prob)
            auc = roc_auc_score(test_y, test_prob)
            
            # 绘制ROC曲线
            ax.plot(fpr, tpr, color=color, linewidth=1.5, 
                   label=f"{label} (AUC = {auc:.3f})")
        
        # 绘制对角线（随机分类器基准线）
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', 
               linewidth=0.8, alpha=0.7, label="Random")
        
        # 设置图表属性
        ax.set_xlabel("False positive rate", fontsize=8)
        ax.set_ylabel("True positive rate", fontsize=8)
        ax.set_title("ROC curves", fontsize=9, fontweight='bold', pad=10)
        
        # 设置坐标轴
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # 设置网格
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # 设置图例
        ax.legend(loc="lower right", fontsize=7, frameon=False)
        
        # 确保坐标轴比例一致
        ax.set_aspect('equal')
        
        # 移除右侧和顶部边框
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # 保存图片
        if save_png:
            save_path = save_path or Config.ROC_PLOT_PATH
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1,
                transparent=False,
                facecolor='white'
            )
            print(f"ROC曲线图已保存至: {save_path}")
        
        return fig
    
    def plot_performance_comparison(self, results, save_png=False, save_path=None):
        """
        绘制性能对比图 - Nature风格
        
        参数:
            results (dict): 模型性能结果
            save_png (bool): 是否保存图片
            save_path (str): 保存路径
            
        返回:
            matplotlib.figure.Figure: 图形对象
        """
        models = list(results.keys())
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
        
        fig, ax = plt.subplots(figsize=(5, 3.5))
        
        # 准备数据
        x = np.arange(len(models))
        width = 0.18
        
        # 绘制柱状图
        for i, metric in enumerate(metrics):
            values = [results[model][i] for model in models]
            color = self.nature_colors[i % len(self.nature_colors)]
            
            bars = ax.bar(x + width * (i - 1.5), values, width, 
                         label=metric, color=color, alpha=0.8,
                         edgecolor='white', linewidth=0.5)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=6)
        
        # 设置图表属性
        ax.set_xlabel('Model', fontsize=8)
        ax.set_ylabel('Performance', fontsize=8)
        ax.set_title('Model Performance Comparison', fontsize=9, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # 设置y轴范围和刻度
        ax.set_ylim([0, 1.1])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # 设置网格
        ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')
        
        # 设置图例
        ax.legend(loc='upper left', fontsize=7, frameon=False, ncol=2)
        
        # 移除右侧和顶部边框
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_png:
            save_path = save_path or os.path.join(Config.OUTPUT_DIR, "performance_comparison.png")
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1,
                transparent=False,
                facecolor='white'
            )
            print(f"性能对比图已保存至: {save_path}")
        
        return fig
    
    def plot_confusion_matrix_heatmap(self, cm, class_names, save_png=False, save_path=None):
        """
        绘制混淆矩阵热力图 - Nature风格
        
        参数:
            cm: 混淆矩阵
            class_names: 类别名称
            save_png (bool): 是否保存图片
            save_path (str): 保存路径
            
        返回:
            matplotlib.figure.Figure: 图形对象
        """
        fig, ax = plt.subplots(figsize=(3, 2.5))
        
        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制热力图
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        
        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel('Normalized frequency', rotation=-90, va="bottom", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        
        # 设置标签
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        
        # 添加数值标签
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                             ha="center", va="center", fontsize=7,
                             color="white" if cm_normalized[i, j] > 0.5 else "black")
        
        ax.set_title("Confusion Matrix", fontsize=9, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('Actual', fontsize=8)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_png:
            save_path = save_path or os.path.join(Config.OUTPUT_DIR, "confusion_matrix.png")
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1,
                transparent=False,
                facecolor='white'
            )
            print(f"混淆矩阵图已保存至: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, feature_names, importance_scores, save_png=False, save_path=None):
        """
        绘制特征重要性图 - Nature风格
        
        参数:
            feature_names: 特征名称列表
            importance_scores: 重要性分数列表
            save_png (bool): 是否保存图片
            save_path (str): 保存路径
            
        返回:
            matplotlib.figure.Figure: 图形对象
        """
        # 排序特征
        sorted_idx = np.argsort(importance_scores)[::-1]
        top_features = min(10, len(feature_names))  # 显示前10个特征
        
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # 绘制水平条形图
        y_pos = np.arange(top_features)
        bars = ax.barh(y_pos, importance_scores[sorted_idx[:top_features]], 
                      color=self.nature_colors[0], alpha=0.8,
                      edgecolor='white', linewidth=0.5)
        
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_features]], fontsize=7)
        ax.set_xlabel('Importance Score', fontsize=8)
        ax.set_title('Feature Importance', fontsize=9, fontweight='bold', pad=10)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontsize=6)
        
        # 设置网格
        ax.grid(True, alpha=0.3, linewidth=0.5, axis='x')
        
        # 移除右侧和顶部边框
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # 翻转y轴，使最重要的特征在顶部
        ax.invert_yaxis()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_png:
            save_path = save_path or os.path.join(Config.OUTPUT_DIR, "feature_importance.png")
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1,
                transparent=False,
                facecolor='white'
            )
            print(f"特征重要性图已保存至: {save_path}")
        
        return fig
    
    @staticmethod
    def show_plots():
        """显示所有图表"""
        plt.show()
    
    @staticmethod
    def close_all_plots():
        """关闭所有图表"""
        plt.close('all')
    
    def reset_style(self):
        """重置到默认matplotlib风格"""
        plt.rcdefaults()
        
    def apply_nature_style(self):
        """应用Nature期刊风格"""
        plt.rcParams.update(self.nature_style)