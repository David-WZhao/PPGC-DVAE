import os

# 定义文件路径（相对路径）
file_path = 'utils/generated_sequences3.txt'  # 直接使用文件名

# 打印当前工作目录
current_working_directory = os.getcwd()
print(f"Current working directory: {current_working_directory}")

# 打印文件路径
absolute_file_path = os.path.abspath(file_path)
print(f"Absolute file path: {absolute_file_path}")

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # 打开并读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 需要统计的字符列表
    chars_to_count = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # 初始化一个字典来存储字符的计数
    char_count = {char: 0 for char in chars_to_count}

    # 遍历文件内容，统计每个字符的出现次数
    total_chars = 0
    for char in content:
        if char in char_count:
            char_count[char] += 1
        total_chars += 1

    # 打印统计结果
    for char, count in char_count.items():
        proportion = count / total_chars * 100  # 计算比例
        print(f"Character '{char}' appears {count} times, which is {proportion:.2f}% of the total characters in the file.")
