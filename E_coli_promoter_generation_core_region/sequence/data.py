import pandas as pd


def kmer_string_to_dna(kmer_string, k):
    """
    将以空格隔开的 k-mer 字符串还原为 DNA 序列。

    参数:
        kmer_string (str): 以空格隔开的 k-mer 字符串。
        k (int): k-mer 的长度。

    返回:
        str: 还原后的 DNA 序列。
    """
    # 将字符串拆分为 k-mer 列表
    kmer_list = kmer_string.split()

    # 初始化序列为第一个 k-mer
    reconstructed_sequence = kmer_list[0]

    # 遍历后续的 k-mers，并逐步拼接
    for kmer in kmer_list[1:]:
        # 拼接 k-mer 的最后一个字符
        reconstructed_sequence += kmer[-1]

    return reconstructed_sequence


k = 3
df = pd.read_csv('train.tsv', sep='\t')
# df = pd.read_csv('dev.csv')
# 获取'data'和'label'列，并转换为列表

data_list = df['sequence'].tolist()
label_list = df['label'].tolist()
promoters = [kmer_string_to_dna(kmer_string, k) for kmer_string in data_list]

print(label_list.count(0))
print(label_list.count(1))

# 1代表strong
flags = ['weak' if item == 0 else 'strong' if item == 1 else item for item in label_list]
data = pd.DataFrame({
    'flags': flags,
    'promoters': promoters
})

# 将 DataFrame 保存为 CSV 文件
data.to_csv('train_data.csv', index=False)
print(data_list[0])
print(promoters[0])