from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np
# 允许的氨基酸字母
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

def calculate_molecular_volume(analysed_seq):
    # 使用每种氨基酸的平均体积估算分子体积（单位：立方埃）。
    # 这些体积是经验值，可以根据需要调整。
    aa_volumes = {
        'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
        'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
        'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
        'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
    }
    volume = sum(aa_volumes.get(aa, 0) for aa in analysed_seq.sequence)
    return volume
# 定义要计算的理化性质
def calculate_properties(sequence):
    try:
        # 检查序列中是否有无效字符
        if not set(sequence).issubset(VALID_AMINO_ACIDS):
            raise ValueError("Sequence contains invalid amino acid letters")

        analysed_seq = ProteinAnalysis(str(sequence))
        properties = {
            "Molecular Weight": analysed_seq.molecular_weight(),
            "Isoelectric Point (pI)": analysed_seq.isoelectric_point(),
            "GRAVY": analysed_seq.gravy(),
            #"Amino Acid Composition": analysed_seq.get_amino_acids_percent(),
            "Aromaticity": analysed_seq.aromaticity(),
            "Instability Index": analysed_seq.instability_index(),
            #"Secondary Structure Fraction": analysed_seq.secondary_structure_fraction(),
            #"Flexibility": analysed_seq.flexibility(),
            "Disulfide Bonds": analysed_seq.molar_extinction_coefficient()[1],  # 预测二硫键
            "Molecular Volume": calculate_molecular_volume(analysed_seq),
            "Secondary Structure Fraction": analysed_seq.secondary_structure_fraction(),
        }
        return properties
    except ValueError as e:
        print(f"Skipping sequence due to error: {e}")
        return None


# 读取FASTA文件
# def read_fasta(file_path):
#     sequences = []
#     for record in SeqIO.parse(file_path, "fasta"):
#         sequences.append(record)
#     return sequences
#
#
# # 将理化性质结果保存为CSV文件
# def save_to_csv(results, output_file):
#     df = pd.DataFrame(results)
#     df.to_csv(output_file, index=False)
#
#
# # 主函数
# def main(fasta_file, output_file):
#     sequences = read_fasta(fasta_file)
#     results = []
#
#     for seq_record in sequences:
#         seq_properties = calculate_properties(seq_record.seq)
#         if seq_properties:  # 只有在没有错误的情况下才添加结果
#             seq_properties["Sequence ID"] = seq_record.id
#             seq_properties["Sequence"] = str(seq_record.seq)
#             results.append(seq_properties)
#
#     # 重新排列列的顺序，确保ID和序列在前两列
#     if results:
#         columns = ["Sequence ID", "Sequence"] + [col for col in results[0] if col not in ["Sequence ID", "Sequence"]]
#         results = [{k: v for k, v in sorted(item.items(), key=lambda x: columns.index(x[0]))} for item in results]
#
#     save_to_csv(results, output_file)
#     print(f"Results saved to {output_file}")
# 使用示例
# fasta_file1 = "data/train.fasta"  # 替换为你的FASTA文件路径
# output_file1 = "data/xingzhi/train.csv"  # 输出CSV文件路径
# main(fasta_file1, output_file1)
# fasta_file2 = "data/test.fasta"  # 替换为你的FASTA文件路径
# output_file2 = "data/xingzhi/test.csv"  # 输出CSV文件路径
# main(fasta_file2, output_file2)
# fasta_file3 = "data/valid.fasta"  # 替换为你的FASTA文件路径
# output_file3 = "data/xingzhi/valid.csv"  # 输出CSV文件路径
# main(fasta_file3, output_file3)
def normalize_matrix(matrix):
    """
        将一个形状为 (64, 10) 的矩阵归一化到 [0, 1] 范围内。

        参数:
        - matrix: 需要归一化的 numpy 数组，形状为 (64, 10)
        - min_values: 每列的最小值，形状为 (10,)
        - max_values: 每列的最大值，形状为 (10,)

        返回:
        - 归一化后的 numpy 数组，形状为 (64, 10)
    """
   

    min_values = np.array([172.1818, 4.05, -4.5, 0.0, -89.84, 0.0, 172.8, 0.0, 0.0, 0.0])
    max_values = np.array([6250, 12.0, 4.2, 1.0, 541.0, 49500.0, 7523, 1.0, 1.0, 1.0])
    normalized_matrix = (matrix - min_values) / (max_values - min_values)

    return normalized_matrix

def read_fasta(file_path):
    """读取FASTA文件"""
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(record)
    return sequences

def save_to_csv(results, output_file):
    """将理化性质结果保存为CSV文件"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def main(fasta_file, output_file):
    sequences = read_fasta(fasta_file)
    results = []

    for seq_record in sequences:
        seq_properties = calculate_properties(seq_record.seq)
        
        if seq_properties:  # 只有在没有错误的情况下才添加结果
            flattened_properties = {}
            for key, value in seq_properties.items():
                if isinstance(value, tuple):
                    for i, v in enumerate(value):
                        flattened_properties[f"{key} {i+1}"] = v
                else:
                    flattened_properties[key] = value
            flattened_properties["Sequence ID"] = seq_record.id
            flattened_properties["Sequence"] = str(seq_record.seq)
            results.append(flattened_properties)

    # 重新排列列的顺序，确保ID和序列在前两列
    if results:
        columns = ["Sequence ID", "Sequence"] + [col for col in results[0] if col not in ["Sequence ID", "Sequence"]]
        results = [{k: v for k, v in sorted(item.items(), key=lambda x: columns.index(x[0]))} for item in results]

    save_to_csv(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    a= calculate_properties('GGLYRLKKVLGK')
    #[2268.7577, 9.930983924865721, -0.5176470588235295, 0.23529411764705882, 13.223529411764709, 8730, 2641.2999999999997, 0.35294117647058826, 0.058823529411764705, 0.0]
    properties_array = []
    if a is None:
        print(f"Warning: properties_dict is None for sequence {seq}. Using default zero array.")
        properties_array = [0.0] * 10  # 使用包含10个零的数组
    else:
        for value in a.values():

            if isinstance(value, tuple):
                properties_array.extend(value)  # 如果是元组，展开并添加到列表中
            else:
                properties_array.append(value)  # 否则，直接添加到列表中
    print(normalize_matrix(properties_array))
    print(properties_array)
    # 示例使用
    # fasta_file = "generated_sequences4.fasta"  # 替换为你的FASTA文件路径
    # output_file = "generated_sequences4.csv"  # 输出CSV文件路径
    # main(fasta_file, output_file)
    