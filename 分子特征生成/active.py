import pandas as pd
import numpy as np
import os

# 检查文件是否存在
csv_file = "EGFR_compounds_maccs_fingerprints.csv"
if not os.path.exists(csv_file):
    print(f"错误：找不到文件 {csv_file}")
    print("请确保文件在当前目录下，或提供完整的文件路径")
    exit()

try:
    # 读取CSV文件
    chembl_df = pd.read_csv(csv_file)
    
    # 检查是否有pIC50列
    if 'pIC50' not in chembl_df.columns:
        print("错误：CSV文件中没有找到 'pIC50' 列")
        print("可用的列名：", list(chembl_df.columns))
        exit()
    
    # 初始化active列
    chembl_df["active"] = np.zeros(len(chembl_df))
    
    # 设置活性标记 (pIC50 >= 6.3 为活性化合物)
    chembl_df.loc[chembl_df[chembl_df.pIC50 >= 6.3].index, "active"] = 1.0
    
    # 打印结果
    print("=" * 50)
    print("ChEMBL数据处理结果")
    print("=" * 50)
    print(f"活性化合物数量: {int(chembl_df.active.sum())}")
    print(f"非活性化合物数量: {len(chembl_df) - int(chembl_df.active.sum())}")
    print(f"化合物总数: {len(chembl_df)}")
    
    # 计算活性化合物比例
    active_ratio = chembl_df.active.sum() / len(chembl_df) * 100
    print(f"活性化合物比例: {active_ratio:.2f}%")
    
    # 显示前几行数据
    print("\n前5行数据预览:")
    print(chembl_df[['pIC50', 'active']].head())
    
    # 可选：保存处理后的数据
    output_file = "EGFR_compounds_maccs_fingerprints_processed.csv"
    chembl_df.to_csv(output_file, index=False)
    print(f"\n处理后的数据已保存至: {output_file}")
    
except FileNotFoundError:
    print(f"错误：无法找到文件 {csv_file}")
except pd.errors.EmptyDataError:
    print("错误：CSV文件为空")
except Exception as e:
    print(f"处理过程中发生错误: {e}")