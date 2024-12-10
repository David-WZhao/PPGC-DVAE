import pandas as pd

# 读取CSV文件
df = pd.read_csv('test_last.csv')

# 计算第二列每一行的字符数
df['char_count'] = df.iloc[:, 1].astype(str).apply(len)

# 筛选出字符数小于等于50的行
filtered_df = df[df['char_count'] <= 50]

# 保存筛选结果为新的CSV文件
filtered_df.to_csv('test_last_new.csv', index=False)

print("已将字符数大于50的行剔除，并保存为train_last_new.csv")