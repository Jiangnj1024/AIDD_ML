import sys
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
import numpy as np
import pandas as pd


############## 分子指纹生成脚本##############
# 该脚本用于从 CSV 文件中读取 SMILES 字符串，生成分子指纹，并将结果保存到 Excel 文件中。
# 支持的指纹方法包括 MACCS、Morgan2 和 Morgan3。
# 使用方法：python molecular_fingerprint.py <input.csv> [method] [n_bits]
# 示例：python molecular_fingerprint.py data.csv morgan2 2048
# 默认方法为 MACCS，指纹长度默认为 2048 位。    
###############

def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    """将 SMILES 转换为分子指纹数组。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[警告] 无效的 SMILES: {smiles}")
        return np.zeros(167 if method == "maccs" else n_bits, dtype=int)

    try:
        if method == "maccs":
            return np.array(MACCSkeys.GenMACCSKeys(mol))
        elif method == "morgan2":
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
        elif method == "morgan3":
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits))
        else:
            print(f"[警告] 未知方法 {method}，使用默认 maccs。")
            return np.array(MACCSkeys.GenMACCSKeys(mol))
    except Exception as e:
        print(f"[错误] 处理失败: {e}")
        return "0" * (167 if method == "maccs" else n_bits)

def process_file(input_file, output_file="fingerprints.xlsx", method="maccs", n_bits=2048):
    """读取 CSV，生成指纹列，并保存到 Excel。"""
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"[错误] 无法读取文件 {input_file}: {e}")
        sys.exit(1)

    if "smiles" not in df.columns:
        print("[错误] 文件中缺少 'smiles' 列。")
        sys.exit(1)

    print(f"[信息] 正在生成指纹（方法: {method}）...")
    
    df["fp"] = df["smiles"].apply(lambda smi: smiles_to_fp(smi, method, n_bits))

    try:
        df.to_csv(output_file, index=False)
        print(f"[成功] 已保存到 {output_file}")
    except Exception as e:
        print(f"[错误] 写入失败: {e}")

if __name__ == "__main__":  # 如果脚本是作为主程序运行（而不是被导入），则执行以下代码
    if len(sys.argv) < 2:  # 如果命令行参数不足 2 个（只包含脚本名），提示用法并退出
        print("用法: python molecular_fingerprint.py <input.csv> [method] [n_bits]")  # 提示正确用法
        print("示例: python molecular_fingerprint.py data.csv morgan2 2048")  # 示例命令
        sys.exit(1)  # 程序异常退出，状态码为 1
        # python molecular_fingerprint.py EGFR_compounds.csv maccs 2048
    input_file = sys.argv[1]  # 获取第一个参数，即输入的 CSV 文件名

    method = sys.argv[2] if len(sys.argv) > 2 else "maccs"  # 获取第二个参数（指纹方法），如果没有提供则默认为 "maccs"

    n_bits = int(sys.argv[3]) if len(sys.argv) > 3 else 2048  # 获取第三个参数（指纹长度），默认值为 2048

    output_file = input_file.replace(".csv", f"_{method}_fingerprints.csv")  # 自动生成输出文件名：替换 .csv 为 _<method>_fingerprints.xlsx

    process_file(input_file, output_file, method, n_bits)  # 调用处理函数，生成指纹并写入输出文件
