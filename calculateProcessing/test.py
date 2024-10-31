import jieba
import text_processing_utils


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


sentence = '你好我是你的男朋友'
function_words_file = r'D:\数据集\LIWC中文词典\功能词典.txt'
function_words = text_processing_utils.load_function_words(function_words_file)

total_func_word_count, total_words, counts, matched_function_words = process_sentence(sentence, function_words)
print(total_func_word_count)
print(total_words)
print(counts)
print(matched_function_words)
