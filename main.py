"""
主程序 - 分子活性预测机器学习项目
"""
import sys
import os
import time
import logging
from datetime import datetime
from data_loader import DataLoader
from model_factory import ModelFactory
from model_evaluator import ModelEvaluator
from visualizer import Visualizer
from config import Config

def setup_logging():
    """设置日志配置"""
    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 设置日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(Config.OUTPUT_DIR, f"experiment_log_{timestamp}.log")
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def print_header():
    """打印项目头部信息"""
    print("=" * 60)
    print("分子活性预测机器学习项目")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

def print_config_info():
    """打印配置信息"""
    print("\n📋 项目配置信息:")
    print(f"  数据文件: {Config.DATA_PATH}")
    print(f"  随机种子: {Config.RANDOM_STATE}")
    print(f"  测试集比例: {Config.TEST_SIZE}")
    print(f"  交叉验证折数: {Config.N_FOLDS}")
    print(f"  输出目录: {Config.OUTPUT_DIR}")

def print_data_info(df, fingerprints, labels):
    """打印数据信息"""
    print(f"\n📊 数据集信息:")
    print(f"  总样本数: {len(df)}")
    
    # 处理不同类型的fingerprints数据
    if hasattr(fingerprints, 'shape'):
        # NumPy数组或类似对象
        print(f"  特征维度: {fingerprints.shape[1]}")
    elif isinstance(fingerprints, list) and len(fingerprints) > 0:
        # 列表类型
        if isinstance(fingerprints[0], (list, tuple)):
            print(f"  特征维度: {len(fingerprints[0])}")
        else:
            print(f"  特征维度: 未知 (一维数据)")
    else:
        print(f"  特征维度: 未知")
    
    print(f"  活性化合物: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  非活性化合物: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

def print_model_info(models):
    """打印模型信息"""
    print(f"\n🤖 创建了 {len(models)} 个模型:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']} ({model['label']})")

def print_performance_summary(performance_results):
    """打印性能汇总"""
    print("\n📈 模型性能汇总:")
    print("-" * 80)
    print(f"{'模型':<20} {'准确率':<10} {'敏感性':<10} {'特异性':<10} {'AUC':<10} {'排名':<6}")
    print("-" * 80)
    
    # 按AUC排序
    sorted_results = sorted(performance_results.items(), key=lambda x: x[1][3], reverse=True)
    
    for rank, (label, (acc, sens, spec, auc)) in enumerate(sorted_results, 1):
        model_name = label.replace('Model_', '')
        print(f"{model_name:<20} {acc:<10.4f} {sens:<10.4f} {spec:<10.4f} {auc:<10.4f} #{rank}")
    
    print("-" * 80)
    
    # 最佳模型
    best_model = sorted_results[0]
    print(f"🏆 最佳模型: {best_model[0].replace('Model_', '')} (AUC: {best_model[1][3]:.4f})")

def save_results(performance_results, log_file):
    """保存结果到文件"""
    results_file = os.path.join(Config.OUTPUT_DIR, "model_results.txt")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("分子活性预测机器学习项目 - 结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置信息:\n")
        f.write(f"  - 数据文件: {Config.DATA_PATH}\n")
        f.write(f"  - 随机种子: {Config.RANDOM_STATE}\n")
        f.write(f"  - 测试集比例: {Config.TEST_SIZE}\n")
        f.write(f"  - 交叉验证折数: {Config.N_FOLDS}\n\n")
        
        f.write("模型性能结果:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'模型':<20} {'准确率':<10} {'敏感性':<10} {'特异性':<10} {'AUC':<10}\n")
        f.write("-" * 60 + "\n")
        
        # 按AUC排序
        sorted_results = sorted(performance_results.items(), key=lambda x: x[1][3], reverse=True)
        
        for label, (acc, sens, spec, auc) in sorted_results:
            model_name = label.replace('Model_', '')
            f.write(f"{model_name:<20} {acc:<10.4f} {sens:<10.4f} {spec:<10.4f} {auc:<10.4f}\n")
        
        f.write("-" * 60 + "\n")
        best_model = sorted_results[0]
        f.write(f"最佳模型: {best_model[0].replace('Model_', '')} (AUC: {best_model[1][3]:.4f})\n")
    
    print(f"📄 结果已保存至: {results_file}")

def main():
    """主函数"""
    start_time = time.time()
    
    try:
        # 设置日志
        log_file = setup_logging()
        
        # 打印项目信息
        print_header()
        print_config_info()
        
        logging.info("开始执行分子活性预测机器学习项目")
        
        # 1. 数据加载和预处理
        print("\n🔄 1. 数据加载和预处理...")
        logging.info("开始数据加载和预处理")
        
        data_loader = DataLoader()
        df = data_loader.load_and_preprocess()
        
        # 2. 准备模型数据
        print("\n🔄 2. 准备模型数据...")
        logging.info("准备模型数据")
        
        fingerprints, labels = data_loader.prepare_model_data()
        print_data_info(df, fingerprints, labels)
        
        # 3. 拆分数据集
        print("\n🔄 3. 拆分数据集...")
        logging.info("拆分训练集和测试集")
        
        train_x, test_x, train_y, test_y = data_loader.split_data(fingerprints, labels)
        splits = [train_x, test_x, train_y, test_y]
        
        print(f"  训练集: {len(train_x)} 样本")
        print(f"  测试集: {len(test_x)} 样本")
        
        # 4. 创建模型
        print("\n🔄 4. 创建模型...")
        logging.info("创建机器学习模型")
        
        models = ModelFactory.create_all_models()
        print_model_info(models)
        
        # 5. 模型训练和验证
        print("\n🔄 5. 模型训练和验证...")
        logging.info("开始模型训练和验证")
        
        performance_results = {}
        
        for i, model_info in enumerate(models, 1):
            print(f"\n--- 训练模型 {i}/{len(models)}: {model_info['name']} ---")
            logging.info(f"训练模型: {model_info['name']}")
            
            model = model_info['model']
            label = model_info['label']
            
            # 训练和验证
            accuracy, sensitivity, specificity, auc = ModelEvaluator.train_and_validate(
                model, label, splits, verbose=True
            )
            
            # 存储结果
            performance_results[label] = (accuracy, sensitivity, specificity, auc)
            
            print(f"✅ {model_info['name']} 训练完成 (AUC: {auc:.4f})")
        
        # 6. 交叉验证
        print("\n🔄 6. 交叉验证...")
        logging.info("开始交叉验证")
        
        for model_info in models:
            print(f"\n======= {model_info['label']} 交叉验证 =======")
            logging.info(f"交叉验证: {model_info['name']}")
            
            ModelEvaluator.cross_validation(
                model_info['model'], df, n_folds=Config.N_FOLDS
            )
        
        # 7. 可视化
        print("\n🔄 7. 生成可视化图表...")
        logging.info("生成可视化图表")
        
        visualizer = Visualizer()
        
        # 绘制ROC曲线
        print("  📊 绘制ROC曲线...")
        roc_fig = visualizer.plot_roc_curves(models, test_x, test_y, save_png=True)
        
        # 绘制性能对比图
        print("  📊 绘制性能对比图...")
        performance_fig = visualizer.plot_performance_comparison(
            performance_results, save_png=True
        )
        
        # 8. 输出总结
        print("\n🔄 8. 项目总结...")
        logging.info("生成项目总结")
        
        print_performance_summary(performance_results)
        
        # 保存结果
        save_results(performance_results, log_file)
        
        # 计算执行时间
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("✅ 项目执行完成！")
        print(f"⏱️  总执行时间: {execution_time:.2f} 秒")
        print(f"📁 输出目录: {Config.OUTPUT_DIR}")
        print(f"📋 日志文件: {log_file}")
        print("=" * 60)
        
        logging.info(f"项目执行完成，总耗时: {execution_time:.2f} 秒")
        
        # 显示图表
        print("\n📊 显示图表...")
        visualizer.show_plots()
        
    except FileNotFoundError as e:
        error_msg = f"数据文件未找到 - {Config.DATA_PATH}"
        print(f"❌ 错误: {error_msg}")
        print("请检查数据文件路径是否正确")
        logging.error(error_msg)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
        logging.info("用户中断执行")
        sys.exit(0)
        
    except Exception as e:
        error_msg = f"执行过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        logging.error(error_msg)
        import traceback
        traceback.print_exc()
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()