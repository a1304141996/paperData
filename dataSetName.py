"""将需要用到的数据集的名字全部拉下来，好让姚老师去拿数据"""

import os
import pandas as pd


def extract_filenames_without_extension(directory, output_dir):
    # 存储文件名的列表
    filenames = []

    # 遍历目录中的文件
    for filename in os.listdir(directory):
        # 检查是否是文件
        if os.path.isfile(os.path.join(directory, filename)):
            # 去掉文件后缀
            name_without_extension = os.path.splitext(filename)[0]
            # 添加到列表中
            filenames.append(name_without_extension)

    # 创建DataFrame
    df = pd.DataFrame(filenames, columns=['Filename'])

    # 将DataFrame存储到Excel文件中
    output_file = output_dir
    df.to_excel(output_file, index=False)

    print(f"文件名已存储到 {output_file}")


# 调用函数
directory = r'D:\毕业论文\处理后的数据\合并后的对话数据\付费'  # 替换为你的目录路径
output_dir = r'D:\毕业论文\处理后的数据\合并后的对话数据\filenames.xlsx'
extract_filenames_without_extension(directory, output_dir)
