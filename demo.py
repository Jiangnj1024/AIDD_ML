import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import KFold, RandomizedSearchCV
import xgboost as xgb
from scipy.stats import uniform, randint

# Step 1: 构造一个简单的示例数据集
data = {
    'SMILES': [
        'CCO',          # Ethanol
        'CC(=O)OC',     # Acetic acid methyl ester
        'CNC',          # Methylamine
        'CCCC',         # Butane
        'CC(=O)O',      # Acetic Acid
        'CN1CCCC1',     # Piperidine
        'C1CCCCC1',     # Cyclohexane
        'CC(C)(C)O'     # tert-Butyl alcohol
    ],
    'Value': [0.7, 1.2, 0.9, 2.1, 1.5, 1.8, 2.0, 1.6]  # 假设是 logP 或溶解度等数值属性
}

df = pd.DataFrame(data)

# Step 2: 实现 generate_fingerprints 函数（以 Morgan 指纹为例）
def generate_fingerprints(smiles, fp_type='Morgan', n_bits=128):
    mol = Chem.MolFromSmiles(smiles)
    if fp_type == 'Morgan':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return list(fp)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")

# Step 3: 调用 optimize_fingerprint 函数
def optimize_fingerprint(df, fp_type, n_bits_range):
    best_score = float('inf')
    best_n_bits = 0
    best_params = None
    best_importances = None

    for n_bits in range(n_bits_range[0], n_bits_range[1] + 1, 8):
        fingerprints = [generate_fingerprints(smiles, fp_type, n_bits) for smiles in df['SMILES']]
        X = np.array(fingerprints)
        y = df['Value'].values

        model = xgb.XGBRegressor()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        search = RandomizedSearchCV(model, param_grid, cv=kf, n_iter=10,
                                    scoring='neg_mean_squared_error',
                                    random_state=42)
        search.fit(X, y)

        mean_score = -search.best_score_
        if mean_score < best_score:
            best_score = mean_score
            best_n_bits = n_bits
            best_params = search.best_params_
            best_importances = search.best_estimator_.feature_importances_

    return best_n_bits, best_params, best_score, best_importances

# Step 4: 运行 Demo
best_n_bits, best_params, best_score, best_importances = optimize_fingerprint(
    df=df,
    fp_type='Morgan',
    n_bits_range=(64, 1024)  # 尝试从64到128位指纹长度
)

print("✅ Best n_bits:", best_n_bits)
print("✅ Best Parameters:", best_params)
print("✅ Best MSE Score:", best_score)
print("✅ Top 10 Feature Importances:", best_importances[:10])