import codecs
import time
# import esm  # pip install esmre
import numpy as np
from tqdm import tqdm
from config import LABLES
import pickle as pkl


def load_dic(train_label, dicfile):
    """
    1、加载蛋白质字典 =  训练语料标签文件 + UMLS中抽取出的实体文件（蛋白质、氨基酸、基因）
    2、扩展实体
    参考论文 Unsupervised gene/protein named entity normalization using automatically ext
    :param train_label:训练语料标签文件
    :param dicfile: ProteinDic.txt
    :return: 基因列表 words(有待改进成 trie )
    """
    words = []
    with codecs.open(train_label, 'r', 'utf-8') as l:
        for line in l:
            splited = line.strip().split('|')
            gene = str(splited[-1])
            # gene = gene.replace('genes', 'gene')
            if not gene.isdigit() and not gene.isalpha() and not gene in words:
                words.append(gene)
                if ' ' in gene:
                    words.append(gene.replace(' ', '-'))
                    words.append(gene.replace(' ', ''))
                if '-' in gene:
                    words.append(gene.replace('-', ' '))
                    words.append(gene.replace('-', ''))

    with codecs.open(dicfile, 'r', 'utf-8') as dicfile:
        lines = dicfile.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i].strip().strip('\n') # .encode('utf-8')
        if not line in words:
            words.append(line)
            if ' ' in line:
                words.append(line.replace(' ', '-'))
                words.append(line.replace(' ', ''))
            if '-' in line:
                words.append(line.replace('-', ' '))
                words.append(line.replace('-', ''))

    # 以词组数目逆序排序
    words = sorted(words, cmp=None, key=lambda x: len(x), reverse=True)
    words = sorted(words, cmp=None, key=lambda x: len(x.split()), reverse=True)

    # dic = esm.Index()
    # for i in tqdm(range(len(words))):
    #     word = words[i]
    #     dic.enter(word)
    # dic.fix()

    with codecs.open('data/dic.pkl', 'wb') as f:
        pkl.dump(words, f, -1)

    return words


def getDicFeature(data, dic):
    """
    字典特征获取
    :param data: 文本数据
    :param dic: 蛋白质字典
    :return:
    """
    features = []
    total_train_time = 0
    for i in tqdm(range(len(data))):
        sentence = data[i].replace('proteins', 'protein')
        start_time = time.time()
        result = max_match_cut(sentence, dic)
        time_diff = time.time() - start_time
        total_train_time += time_diff
        features.append(result)
    print("the feature calculation took %.2f sec total." % total_train_time)
    features = np.asarray(features)
    return features



def max_match_cut(sentence, dic):
    """
    将字典中的所有词与输入句子进行完全匹配，若匹配，则分配"IBES"；否则，分配"O"
    :param sentence:输入句子
    :param dic:蛋白质词典
    :return:字典特征
    """
    temp = sentence
    splited_s = sentence.split()
    result = dic.query(sentence)
    for tuple in result:
        ind0, ind1 = tuple[0]
        if sentence[ind1] == ' ':
            entity = tuple[1]
            temp2 = temp[:ind0] + '@ ' * (len(entity.split()) - 1) + '@' + temp[ind1:]
            # if not len(temp2.split()) == len(sentence.split()):
            assert len(temp2.split()) == len(splited_s)
            for i in range(len(splited_s)):
                if '@' in temp2.split()[i]:
                    splited_s[i] = '@'

    sentence = ' '.join(splited_s)

    feature = [LABLES.index('O') for i in range(len(splited_s))]  # initial label array
    k = 0
    while k < len(splited_s):
        total = 0
        temp = k
        while k < len(splited_s) and splited_s[k] in ['@', '@,', '@.']:
            total += 1
            k += 1
        else:
            if total == 0:
                k += 1
            elif total == 1:
                feature[temp] = LABLES.index('S')
            elif total > 1:  # 包含两个字及以上的词
                feature[temp] = LABLES.index('B')
                for i in range(1, (total - 1)):
                    temp = temp + 1
                    feature[temp] = LABLES.index('I')
                feature[temp + 1] = LABLES.index('E')
            else:
                print('Exception!!')
    assert len(splited_s) == len(feature)

    # feature = []
    # splited_line = sentence.split()
    # for word in splited_line:
    #     num = 0
    #     if word.isalpha() or word.isdigit():
    #         feature.append(LABLES.index('O'))
    #         continue
    #     num += word.count('@')
    #     if num == 1:
    #         feature.append(LABLES.index('S'))
    #     elif num > 1:  # 包含两个字及以上的词
    #         feature.append(LABLES.index('B'))
    #         for k in range(1, (num - 1)):
    #             feature.append(LABLES.index('I'))
    #         feature.append(LABLES.index('E'))
    #     else:
    #         feature.append(LABLES.index('O'))
    #
    # if not len(temp.split())==len(feature):
    #     print('\n', len(temp.split()), len(feature))
    #     print(temp)
    #     print(result)
    #     print(sentence)
    #     print(feature)
    # assert len(temp.split())==len(feature)
    return feature



def get(train_texts, valid_texts, test_texts):
    print("添加字典特征（训练语料 + UMLS抽取出的蛋白质词典）")
    # dic = load_dic('data/train/GENE.eval', 'data/ProteinDic.txt')
    with codecs.open('data/pkl/words.pkl', 'rb') as f:
        words = pkl.load(f)

    dic = esm.Index()
    for i in range(len(words)):
        word = words[i]
        dic.enter(word)
    dic.fix()

    dicFeature_train = getDicFeature(train_texts, dic)
    dicFeature_val = getDicFeature(valid_texts, dic)
    dicFeature_test = getDicFeature(test_texts, dic)
    return dicFeature_train, dicFeature_val, dicFeature_test