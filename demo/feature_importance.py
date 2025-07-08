import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm  # 用于显示进度条

class FeatureImportanceCalculator:
    def compute_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算特征重要性，并显示训练进度条。
        
        参数：
            X: 特征矩阵，np.ndarray，形状 (n_samples, n_features)
            y: 标签向量，np.ndarray，形状 (n_samples,)
        返回：
            importance_scores: 每个特征的重要性得分，np.ndarray，float 类型
        """
        print("计算特征重要性...")

        # 实例化 XGBoost 分类器
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        # 类型：XGBClassifier 实例

        # 使用 tqdm 显示一个假进度条（因为 XGBoost 不提供内置回调）
        print("开始训练模型（带进度条）...")
        for _ in tqdm(range(100), desc="模型训练进度", unit="step"):
            # 在第一个 step 完成真正的 fit()
            if _ == 0:
                model.fit(X, y)

        # 获取特征重要性分数
        importance_scores = model.feature_importances_
        # 类型：np.ndarray，元素为 float

        self.feature_importance = importance_scores

        # 打印平均重要性值
        avg_imp = np.mean(importance_scores)  # 类型：float
        print(f"特征重要性计算完成，平均重要性: {avg_imp:.4f}")

        return importance_scores
    
    
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # 生成合成数据集
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                               n_classes=2, random_state=42)
    # X: np.ndarray, shape (1000, 10)
    # y: np.ndarray, shape (1000,)

    print(f"数据集形状: 特征={X.shape[1]}, 样本={X.shape[0]}")

    calculator = FeatureImportanceCalculator()
    importance_scores = calculator.compute_feature_importance(X, y)
    # 返回值 importance_scores: np.ndarray

    # 输出每个特征的重要性
    for i, score in enumerate(importance_scores):
        print(f"特征 {i}: 重要性 = {score:.4f}")