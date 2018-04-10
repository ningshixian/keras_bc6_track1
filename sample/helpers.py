import re
import string

def get_answer():
    """Get an answer."""
    return True


# 对单词进行清洗
def wordNormalize(word):
    word = word.lower()
    word = re.sub(u'\s+', '', word, flags=re.U)  # 匹配任何空白字符
    word = word.replace("--", "-")
    word = re.sub("\"+", '"', word)

    # 特殊符号归一化
    temp = word
    for char in word:
        if char not in string.printable:
            temp = temp.replace(char, '')
    word = temp
    return word


def createCharDict():
    '''
    创建字符字典
    '''
    # charSet = set()
    # with open(trainCorpus + '/' + 'train.out', encoding='utf-8') as f:
    #     for line in f:
    #         if not line == '\n':
    #             a = line.strip().split('\t')
    #             charSet.update(a[0])  # 获取字符集合

    char2idx = {}
    for char in string.printable:
        char2idx[char] = len(char2idx)
    char2idx['**'] = len(char2idx)  # 用于那些未收录的字符
    print(char2idx)
    return char2idx