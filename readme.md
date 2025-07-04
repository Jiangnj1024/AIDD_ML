# 分子活性预测机器学习项目
这是一个基于机器学习的分子活性预测项目，使用分子指纹数据来预测化合物的生物活性。
项目结构
molecular_activity_prediction/
├── config.py              # 配置文件
├── data_loader.py          # 数据加载和预处理模块
├── model_factory.py        # 模型工厂
├── model_evaluator.py      # 模型评估模块
├── visualizer.py           # 可视化模块
├── main.py                 # 主程序
├── requirements.txt        # 依赖文件
├── README.md              # 项目说明
└── output/                # 输出目录
    ├── roc_auc.png        # ROC曲线图
    └── performance_comparison.png  # 性能对比图
功能特性
🔬 数据处理

自动加载和预处理分子指纹数据
字符串格式指纹转换为数组格式
数据集自动拆分（训练集/测试集）

🤖 机器学习模型

随机森林 (Random Forest)
支持向量机 (SVM)
人工神经网络 (ANN/MLP)

📊 模型评估

准确率 (Accuracy)
敏感性 (Sensitivity/Recall)
特异性 (Specificity)
AUC (Area Under Curve)
K折交叉验证

📈 可视化

ROC曲线对比图
模型性能对比图
支持中文显示

安装和使用
1. 环境要求

Python 3.7+
推荐使用虚拟环境

2. 安装依赖
bashpip install -r requirements.txt
3. 配置数据路径
编辑 config.py 文件中的 DATA_PATH 变量，设置你的数据文件路径：
pythonDATA_PATH = "/path/to/your/EGFR_compounds_maccs_fingerprints_processed.csv"
4. 运行程序
bashpython main.py
数据格式要求
输入的CSV文件应包含以下列：

fp: 分子指纹（字符串格式，如 "[1 0 1 0 ...]"）
active: 活性标签（0或1）

配置选项
在 config.py 文件中可以调整以下参数：
数据配置

DATA_PATH: 数据文件路径
TEST_SIZE: 测试集比例（默认0.2）
RANDOM_STATE: 随机种子（默认12345）

模型参数

RF_PARAMS: 随机森林参数
SVM_PARAMS: SVM参数
ANN_PARAMS: 神经网络参数

交叉验证

N_FOLDS: 交叉验证折数（默认3）

输出结果
程序运行后会生成：

控制台输出: 详细的训练过程和性能指标
图表文件:

output/roc_auc.png: ROC曲线对比图
output/performance_comparison.png: 性能对比柱状图



模块说明
config.py
存储所有配置参数，包括模型参数、文件路径等。
data_loader.py
负责数据的加载和预处理：

读取CSV文件
转换分子指纹格式
数据集拆分

model_factory.py
模型工厂模式，负责创建不同类型的机器学习模型：

随机森林
支持向量机
多层感知机

model_evaluator.py
模型评估功能：

性能指标计算
模型训练和验证
K折交叉验证

visualizer.py
可视化功能：

ROC曲线绘制
性能对比图
支持中文字体显示

main.py
主程序，协调所有模块的工作流程。
使用示例
基本使用
pythonfrom data_loader import DataLoader
from model_factory import ModelFactory
from model_evaluator import ModelEvaluator

# 加载数据
loader = DataLoader()
df = loader.load_and_preprocess()

# 创建模型
rf_model = ModelFactory.create_random_forest()

# 评估模型
evaluator = ModelEvaluator()
# ... 具体评估代码
自定义模型参数
pythonfrom model_factory import ModelFactory

# 自定义随机森林参数
custom_rf_params = {
    "n_estimators": 200,
    "criterion": "gini",
    "max_depth": 10
}

rf_model = ModelFactory.create_random_forest(custom_rf_params)
性能指标说明

准确率 (Accuracy): 正确预测的样本数占总样本数的比例
敏感性 (Sensitivity): 真正例占所有实际正例的比例
特异性 (Specificity): 真负例占所有实际负例的比例
AUC: ROC曲线下的面积，值越大模型性能越好

扩展功能
添加新模型

在 model_factory.py 中添加新的模型创建方法
在 create_all_models() 中添加新模型
更新配置文件添加新模型的参数

添加新的评估指标

在 model_evaluator.py 中的 calculate_performance() 方法中添加新指标
更新可视化模块以支持新指标的显示

注意事项

确保数据文件路径正确
数据格式必须符合要求
建议使用虚拟环境避免依赖冲突
对于大数据集，可能需要调整交叉验证折数

故障排除
常见问题

数据文件未找到

检查 config.py 中的 DATA_PATH 设置
确认文件存在且有读取权限


中文字体显示问题

确保系统安装了中文字体
可以修改 config.py 中的字体设置


内存不足

减少交叉验证折数
考虑数据采样或分批处理



许可证
本项目基于 MIT 许可证开源。
贡献
欢迎提交 Issue 和 Pull Request 来改进项目。
更新日志
v1.0.0 (2024-01-01)

初始版本发布
支持三种机器学习模型
完整的评估和可视化功能



分子特征生成


