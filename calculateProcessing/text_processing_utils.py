"""
由于Calculate_FW与Calculate_rlsm对文件中对话的前期处理方式完全相同，为避免使用重复的代码，同时增加代码的可阅读性以及逻辑的清晰性
将处理方式相同的部分单独拿出来作为一个文件，后续两个代码只需要导入该文件并调用相关函数就行了
"""
'''
将对话中的标点符号以及nan标识去掉，并整合为一段纯粹的句子，再将这个句子分词，并输出句子中的分词数据：
total_func_word_count:功能词总数
total_words：分词总数
counts：匹配的功能词数量
matched_function_words：匹配的功能词字典
'''

import re
import jieba


# 获取功能词词典
def load_function_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        function_words = [line.strip() for line in file.readlines()]
        function_words.sort(key=len, reverse=True)
    return function_words


# 按空行分割文本以区分不同有效咨询
def split_paragraphs(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()
    paragraphs = contents.split('\n\n\n')
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


# 清理句子中的‘nan'数据和标点符号，并合并为一个句子
def clean_and_merge(sentence):
    cleaned_parts = []
    parts = sentence.split()
    for part in parts:
        cleaned_part = re.sub(r'[^\w\s]', '', part)
        if part.lower() == 'nan':
            continue
        else:
            cleaned_parts.append(cleaned_part)
    merged_sentence = ''.join(cleaned_parts)
    return merged_sentence


# 用于计算FW值
def calculate_fw_values(total_func_count, total_words):
    if total_words == 0:
        return 0  # 避免除数为0
    return total_func_count / total_words


# 处理单个句子，统计词语数量和功能词数量
def process_sentence(sentence, function_words):
    total_func_word_count = 0
    counts = {}
    total_words = len(sentence)
    seg_words = jieba.lcut(sentence, cut_all=False)
    word_dict = {}
    matched_function_words = []
    for word in seg_words:
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    for func_word in function_words:
        if func_word in word_dict:
            matched_function_words.append(func_word)
            counts[func_word] = word_dict[func_word]
            total_func_word_count += word_dict[func_word]

    return total_func_word_count, total_words, counts, matched_function_words  # 四个参数，但是FW引用的只有三个参数


