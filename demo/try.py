import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class MolecularFingerprintGenerator:
    """生成不同类型的分子指纹"""
    
    def __init__(self):
        self.maccs_length = 167  # MACCS固定长度
    
    def generate_morgan_fingerprint(self, smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
        """生成Morgan指纹"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    
    def generate_topological_fingerprint(self, smiles: str, n_bits: int = 1024) -> np.ndarray:
        """生成拓扑指纹"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = FingerprintMols.FingerprintMol(mol)
        # 转换为固定长度的二进制向量
        fp_bits = np.zeros(n_bits)
        for i in range(min(len(fp), n_bits)):
            fp_bits[i] = fp[i]
        return fp_bits
    
    def generate_maccs_fingerprint(self, smiles: str) -> np.ndarray:
        """生成MACCS指纹"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.maccs_length)
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        return np.array(fp)

class FingerprintOptimizer:
    """优化分子指纹长度"""
    
    def __init__(self, fingerprint_generator: MolecularFingerprintGenerator):
        self.fp_gen = fingerprint_generator
        
    def optimize_fingerprint_length(self, smiles_list: List[str], labels: List[int], 
                                  fp_type: str, length_range: range = None) -> Tuple[int, float]:
        """优化指纹长度 - 快速版本"""
        
        # 使用更小的搜索范围和更大的步长
        if length_range is None:
            if fp_type == 'morgan':
                length_range = range(64, 513, 64)  # [64, 128, 192, 256, 320, 384, 448, 512]
            elif fp_type == 'topological':
                length_range = range(64, 513, 64)  # 同上
            else:
                return self.fp_gen.maccs_length, 0.0  # MACCS固定长度
        
        best_length = None
        best_score = 0
        
        print(f"优化{fp_type}指纹长度（快速模式）...")
        
        for length in length_range:
            if fp_type == 'morgan':
                fingerprints = [self.fp_gen.generate_morgan_fingerprint(smiles, n_bits=length) 
                              for smiles in smiles_list]
            elif fp_type == 'topological':
                fingerprints = [self.fp_gen.generate_topological_fingerprint(smiles, n_bits=length) 
                              for smiles in smiles_list]
            else:
                continue  # MACCS固定长度
            
            X = np.array(fingerprints)
            y = np.array(labels)
            
            # 使用简化的5折交叉验证
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', 
                                    n_estimators=50, max_depth=3)  # 减少树的数量和深度
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5折而不是10折
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_length = length
            
            print(f"  长度 {length}: 准确率 {avg_score:.4f}")
        
        print(f"{fp_type}指纹最佳长度: {best_length}, 最佳准确率: {best_score:.4f}")
        return best_length, best_score

class ElixirFP:
    """ElixirFP：注意力驱动的分子指纹融合"""
    
    def __init__(self, fingerprint_generator: MolecularFingerprintGenerator):
        self.fp_gen = fingerprint_generator
        self.morgan_length = None
        self.topological_length = None
        self.feature_importance = None
        self.pca = None
        self.kpca = None
        self.scaler = StandardScaler()
        
    def generate_optimized_fingerprints(self, smiles_list: List[str], 
                                      morgan_length: int, topological_length: int) -> np.ndarray:
        """生成优化后的分子指纹"""
        fingerprints = []
        
        for smiles in smiles_list:
            morgan_fp = self.fp_gen.generate_morgan_fingerprint(smiles, n_bits=morgan_length)
            topological_fp = self.fp_gen.generate_topological_fingerprint(smiles, n_bits=topological_length)
            maccs_fp = self.fp_gen.generate_maccs_fingerprint(smiles)
            
            # 连接三种指纹
            combined_fp = np.concatenate([morgan_fp, topological_fp, maccs_fp])
            fingerprints.append(combined_fp)
        
        return np.array(fingerprints)
    
    def compute_feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算特征重要性 - 快速版本"""
        print("计算特征重要性...")
        # 使用更少的树和更浅的深度
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', 
                                n_estimators=50, max_depth=3)
        model.fit(X, y)
        
        # 获取特征重要性分数
        importance_scores = model.feature_importances_
        self.feature_importance = importance_scores
        
        print(f"特征重要性计算完成，平均重要性: {np.mean(importance_scores):.4f}")
        return importance_scores
    
    def apply_conventional_pca(self, X: np.ndarray, n_components: int = 20) -> np.ndarray:
        """应用传统PCA - 减少组件数量"""
        print(f"应用传统PCA，降维至{n_components}维...")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA降维
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA解释方差比: {explained_variance:.4f}")
        
        return X_pca
    
    def apply_attention_kpca(self, X: np.ndarray, n_components: int = 50, 
                           kernel: str = 'rbf', gamma: float = 1.0) -> np.ndarray:
        """应用注意力驱动的KPCA - 减少组件数量"""
        print(f"应用注意力驱动的KPCA，降维至{n_components}维...")
        
        # 使用特征重要性作为权重
        if self.feature_importance is not None:
            # 根据特征重要性加权特征
            weights = self.feature_importance / np.sum(self.feature_importance)
            X_weighted = X * weights
        else:
            X_weighted = X
            print("警告：未找到特征重要性，使用原始特征")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X_weighted)
        
        # KPCA降维
        self.kpca = KernelPCA(n_components=n_components, kernel=kernel, 
                             gamma=gamma, random_state=42)
        X_kpca = self.kpca.fit_transform(X_scaled)
        
        return X_kpca
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray, method_name: str) -> Dict:
        """评估模型性能 - 快速版本"""
        print(f"评估{method_name}性能...")
        
        # 使用更快的模型配置
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', 
                                n_estimators=50, max_depth=3)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5折而不是10折
        
        results = {
            'method': method_name,
            'accuracy_mean': np.mean(scores),
            'accuracy_std': np.std(scores),
            'scores': scores
        }
        
        print(f"{method_name} - 准确率: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        return results

def main_pipeline():
    """主要流程 - 快速版本"""
    # 示例数据（实际使用时需要替换为真实的DrugAge数据）
    print("=== ElixirFP: 注意力驱动的分子指纹融合（快速版本）===\n")
    
    # 模拟更多样本以获得更好的结果
    np.random.seed(42)
    sample_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # 布洛芬
        "CC(=O)OC1=CC=CC=C1C(=O)O",        # 阿司匹林
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # 咖啡因
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34", # 睾酮
        "C1=CC=C(C=C1)C(C(=O)O)N",         # 苯丙氨酸
        "CC(C)C1=CC=C(C=C1)C(C)C",         # 对异丙基甲苯
        "C1=CC=C2C(=C1)C(=O)C3=CC=CC=C3C2=O", # 蒽醌
        "CCN(CC)C1=CC=C(C=C1)N=NC2=CC=CC=C2", # 二甲基氨基偶氮苯
        "C1=CC=C(C=C1)C=CC(=O)O",          # 肉桂酸
        "C1=CC=C(C=C1)C(=O)C2=CC=CC=C2",   # 二苯甲酮
    ] * 5  # 扩展到50个样本
    
    # 模拟标签（抗衰老效果）
    sample_labels = np.random.choice([0, 1], size=len(sample_smiles), p=[0.6, 0.4])
    
    print(f"数据集大小: {len(sample_smiles)} 个化合物")
    print(f"标签分布: {np.bincount(sample_labels)}")
    
    # 初始化组件
    fp_generator = MolecularFingerprintGenerator()
    optimizer = FingerprintOptimizer(fp_generator)
    elixir_fp = ElixirFP(fp_generator)
    
    # 1. 优化指纹长度（快速版本）
    print("\n1. 优化分子指纹长度（快速模式）")
    morgan_length, morgan_score = optimizer.optimize_fingerprint_length(
        sample_smiles, sample_labels, 'morgan'
    )
    
    topological_length, topological_score = optimizer.optimize_fingerprint_length(
        sample_smiles, sample_labels, 'topological'
    )
    
    print(f"\n最优长度 - Morgan: {morgan_length}, Topological: {topological_length}")
    
    

    
    # 2. 生成优化后的指纹
    print("\n2. 生成优化后的分子指纹")
    X_combined = elixir_fp.generate_optimized_fingerprints(
        sample_smiles, morgan_length, topological_length
    )
    print(f"组合指纹维度: {X_combined.shape}")
    
    
    
    # 3. 计算特征重要性
    print("\n3. 计算特征重要性")
    importance_scores = elixir_fp.compute_feature_importance(X_combined, sample_labels)
    
    # 4. 应用传统PCA融合
    print("\n4. 应用传统PCA融合 (ElixirFP)")
    X_pca = elixir_fp.apply_conventional_pca(X_combined, n_components=20)
    pca_results = elixir_fp.evaluate_model(X_pca, sample_labels, "ElixirFP (PCA)")
    
    # 5. 应用注意力驱动的KPCA融合
    print("\n5. 应用注意力驱动的KPCA融合 (Attention-ElixirFP)")
    X_kpca = elixir_fp.apply_attention_kpca(X_combined, n_components=50)
    kpca_results = elixir_fp.evaluate_model(X_kpca, sample_labels, "Attention-ElixirFP (KPCA)")
    
    # 6. 结果比较
    print("\n6. 结果比较")
    print("="*60)
    print(f"Morgan指纹 (单独):                {morgan_score:.4f}")
    print(f"Topological指纹 (单独):           {topological_score:.4f}")
    print(f"ElixirFP (PCA):                  {pca_results['accuracy_mean']:.4f} ± {pca_results['accuracy_std']:.4f}")
    print(f"Attention-ElixirFP (KPCA):       {kpca_results['accuracy_mean']:.4f} ± {kpca_results['accuracy_std']:.4f}")
    
    # 7. 特征重要性分析
    print("\n7. 特征重要性分析")
    print(f"特征重要性统计:")
    print(f"  最大重要性: {np.max(importance_scores):.6f}")
    print(f"  平均重要性: {np.mean(importance_scores):.6f}")
    print(f"  标准差:     {np.std(importance_scores):.6f}")
    
    # 找出最重要的特征
    top_features = np.argsort(importance_scores)[-10:]
    print(f"\n前10个最重要特征的索引: {top_features}")
    
    print("对应的重要性分数:", [f"{score:.6f}" for score in importance_scores[top_features]])
    
    # 8. 性能提升分析
    print("\n8. 性能提升分析")
    print("="*60)
    baseline_score = max(morgan_score, topological_score)
    pca_improvement = pca_results['accuracy_mean'] - baseline_score
    kpca_improvement = kpca_results['accuracy_mean'] - baseline_score
    
    print(f"基线性能（最好单一指纹）:    {baseline_score:.4f}")
    print(f"ElixirFP提升:               {pca_improvement:+.4f}")
    print(f"Attention-ElixirFP提升:     {kpca_improvement:+.4f}")
    
    print(f"\n总运行时间大幅缩短，推荐使用此快速版本进行实验！")

if __name__ == "__main__":
    main_pipeline()