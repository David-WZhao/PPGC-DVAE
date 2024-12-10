import csv

# 定义文件路径
file_path = 'data/train_last.csv'  # 替换为你的CSV文件路径

# 需要统计的字符列表
chars_to_count = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# 初始化一个字典来存储字符的计数
char_count = {char: 0 for char in chars_to_count}

# 初始化总字符数
total_chars = 0

# 读取CSV文件并统计第二列中字符的出现次数
with open(file_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) > 1:  # 确保第二列存在
            second_column = row[1]
            for char in second_column:
                if char in char_count:
                    char_count[char] += 1
                total_chars += 1

# 打印统计结果
for char, count in char_count.items():
    proportion = (count / total_chars) * 100  # 计算比例
    print(f"Character '{char}' appears {count} times in the second column, which is {proportion:.2f}% of the total characters in the second column.")
