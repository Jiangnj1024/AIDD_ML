"""
主程序 - 分子活性预测机器学习项目
"""
import sys
import os
from data_loader import DataLoader
from model_factory import ModelFactory
from model_evaluator import ModelEvaluator
from visualizer import Visualizer
from config import Config

def main():
    """主函数"""
    print("="*60)
    print("分子活性预测机器学习项目")
    print("="*60)
    
    try:
        # 1. 数据加载和预处理
        print("\n1. 数据加载和预处理...")
        data_loader = DataLoader()
        df = data_loader.load_and_preprocess()
        
        # 2. 准备模型数据
        print("\n2. 准备模型数据...")
        fingerprints, labels = data_loader.prepare_model_data()
        
        # 3. 拆分数据集
        print("\n3. 拆分数据集...")
        train_x, test_x, train_y, test_y = data_loader.split_data(fingerprints, labels)
        splits = [train_x, test_x, train_y, test_y]
        
        # 4. 创建模型
        print("\n4. 创建模型...")
        models = ModelFactory.create_all_models()
        print(f"创建了 {len(models)} 个模型:")
        for model in models:
            print(f"  - {model['name']} ({model['label']})")
        
        # 5. 模型训练和验证
        print("\n5. 模型训练和验证...")
        performance_results = {}
        
        for model_info in models:
            print(f"\n--- 训练 {model_info['name']} ---")
            model = model_info['model']
            label = model_info['label']
            
            # 训练和验证
            accuracy, sensitivity, specificity, auc = ModelEvaluator.train_and_validate(
                model, label, splits, verbose=True
            )
            
            # 存储结果
            performance_results[label] = (accuracy, sensitivity, specificity, auc)
        
        # 6. 交叉验证
        print("\n6. 交叉验证...")
        for model_info in models:
            print(f"\n======= {model_info['label']} =======")
            ModelEvaluator.cross_validation(
                model_info['model'], df, n_folds=Config.N_FOLDS
            )
        
        # 7. 可视化
        print("\n7. 生成可视化图表...")
        visualizer = Visualizer()
        
        # 绘制ROC曲线
        roc_fig = visualizer.plot_roc_curves(models, test_x, test_y, save_png=True)
        
        # 绘制性能对比图
        performance_fig = visualizer.plot_performance_comparison(
            performance_results, save_png=True
        )
        
        # 8. 输出总结
        print("\n8. 项目总结...")
        print("="*60)
        print("模型性能汇总:")
        print("-"*60)
        print(f"{'模型':<15} {'准确率':<8} {'敏感性':<8} {'特异性':<8} {'AUC':<8}")
        print("-"*60)
        
        for label, (acc, sens, spec, auc) in performance_results.items():
            print(f"{label:<15} {acc:<8.2f} {sens:<8.2f} {spec:<8.2f} {auc:<8.2f}")
        
        print("="*60)
        print("项目执行完成！")
        print(f"图表已保存至: {Config.OUTPUT_DIR}")
        
        # 显示图表
        visualizer.show_plots()
        
    except FileNotFoundError:
        print(f"错误: 数据文件未找到 - {Config.DATA_PATH}")
        print("请检查数据文件路径是否正确")
        sys.exit(1)
        
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()