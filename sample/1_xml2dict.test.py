#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
    从 trian.out.txt 抽取字典特征
'''
from xml.dom.minidom import parse
import re
import codecs
import os
from tqdm import tqdm
import esm
import csv


def readKB():
    '''
    读取蛋白质/基因字典
    '''
    word_list=set()
    pro_path = '/Users/ningshixian/Desktop/bc6_data_big/uniprot_sprot.dat2'
    gene_path = '/Users/ningshixian/Desktop/bc6_data_big/gene_info2'
    csv_path = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/annotations.csv'
    csv_path_test = r'/Users/ningshixian/Desktop/BC6_Track1/test_corpus_20170804/annotations.csv'

    with open(pro_path) as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        splited = line.split('\t')
        e_list = splited[1].replace('\n', '').split(';')
        # word_list.extend(e_list)
        for e in e_list:
            e = e.strip().lower()
            if len(e)>3 and not e.isdigit():
                word_list.add(e)

    with open(gene_path) as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        splited = line.split('\t')
        e_list = splited[1].replace('\n', '').split(';')
        # word_list.extend(e_list)
        for e in e_list:
            e = e.strip().lower()
            if len(e)>3 and not e.isdigit():
                word_list.add(e)

    with open(csv_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            id = row['obj']
            e = row['text']
            # text = row['text'].lower()
            if id.startswith('NCBI gene:') or id.startswith('Uniprot:') or \
                    id.startswith('gene:') or id.startswith('protein:'):
                if len(e)>3 and not e.isdigit():
                    word_list.add(e)

    with open(csv_path_test) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            id = row['obj']
            e = row['text']
            # text = row['text'].lower()
            if id.startswith('NCBI gene:') or id.startswith('Uniprot:') or \
                    id.startswith('gene:') or id.startswith('protein:'):
                if len(e)>3 and not e.isdigit():
                    word_list.add(e)

    word_list = list(word_list)
    xx = sorted(enumerate(word_list), key = lambda x:len(x[1]), reverse=True)
    xx = [item[0] for item in xx]
    word_list = [word_list[i] for i in xx]

    return list(word_list)


def max_match_cut(sentence, dic):
    """
    最大匹配，标记实体
    :param sentence:输入句子
    :param dic:词典
    :return:字典特征
    """
    tmp = sentence
    splited_s = sentence.split()
    result = dic.query(sentence.lower())
    result = list(set(result))
    # # 先按照实体长度排序
    # en_list = [len(r[1]) for r in result]
    # en_sorted = sorted(enumerate(en_list), key=lambda x:x[1], reverse=True)
    # en_idx = [x[0] for x in en_sorted]  # 数组下标
    # result = [result[idx] for idx in en_idx]

    # 按照offset排序
    offset_list = [r[0][0] for r in result]
    offset_sorted = sorted(enumerate(offset_list), key=lambda x:x[1], reverse=True)
    offset_idx = [x[0] for x in offset_sorted]  # 排序后的数组下标
    result = [result[idx] for idx in offset_idx]  # 排序后的结果

    def filter(result):
        '''
        对排序后的结果进行过滤，留下可能的实体结果
        '''
        p1 = 0
        p2 = 0
        id_list = []
        temp = None
        for i in range(len(result)):
            start = result[i][0][0]
            end = result[i][0][1]
            if start == p1:
                if end > p2:
                    if id_list:
                        id_list.pop()
                    temp = i
                    p1 = start
                    p2 = end
            else:
                if end >= p2 and start < p1:
                    if id_list:
                        id_list.pop()
                p1 = start
                p2 = end
                if temp:
                    id_list.append(temp)
                    id_list.append(i)
                else:
                    id_list.append(i)
                temp = None

        if temp:
            id_list.append(temp)
        result = [result[idx] for idx in id_list]
        return result

    result = filter(result)
    result = filter(result)
    # if result:
    #     print(result)

    '''
    对句子中的实体进行标记
    '''
    prex = None
    end = None
    lenth = None
    for tuple in result:
        ind0, ind1 = tuple[0]
        if sentence[ind0-1] == ' ' and sentence[ind1] == ' ':
            # print(ind0, ind1)
            entity = tuple[1]
            if prex and end:
                if ind0==prex or ind1==end:
                    if not ind1-ind0<=lenth:
                        continue
            prex = ind0
            end = ind1
            lenth = ind1-ind0

            tmp_tmp = tmp
            left = tmp[:ind0]
            mid = tmp[ind0:ind1]
            right = tmp[ind1:]
            tmp = left + ' ' + str(B_flag) + mid + str(I_flag) + ' ' + right
            tmp = tmp.replace('   ', ' ').replace('  ', ' ')

            if not len(tmp.split()) == len(splited_s):
                # 实体标记错误，跳过
                tmp = tmp_tmp
                continue
            assert len(tmp.split()) == len(splited_s)
    return tmp


def readXML():
    word_list = readKB()
    # word_list.append('PPCA')

    print('获取字典树trie')
    dic = esm.Index()
    for i in range(len(word_list)):
        word = word_list[i].lower()
        dic.enter(word)
    dic.fix()

    print('最大匹配')
    results = []
    with open('/Users/ningshixian/PycharmProjects/keras_bc6_track1/sample/data/BIBIO/test/test.txt') as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        for tag in tag_list:
            line = line.replace(tag, '')
        line = max_match_cut(line, dic)
        # label = max_match_cut2(line, word_list)
        results.append(line)

    with open('/Users/ningshixian/PycharmProjects/keras_bc6_track1/sample/data/BIBIO/test/test2.txt', 'w') as f:
        for sentence in results:
            f.write(sentence)


# 利用GENIA tagger工具对标记过的语料进行预处理（分词+POS+CHUNK+NER）
# 得到 test.genia 文件


def judge(word, label_sen, flag):
    '''
    0:实体结束    1:实体开头或内部  2:特殊实体[] 

    情况1：嵌套实体，如：pHA，其标注为: B-p B-ha-I-I
        对于后者的标注形式较难解决：
        ①实体包含多个标签B，如：B-PROTEIN-B-GENE-Spi-I-GENE
        ②实体包含多个标签I，如：B-GENE/RARα/I-GENE/I-PROTEIN
    解决1：针对特定情况，比较B-和I-的数量，选择多的一方

    情况2：实体词本身最后一个字母是B，如：B-GENE-SipB-I-GENE
    解决2：改变标记形式<B-XXX-><-I-XXX>为<B-*/></I-*>
          丢弃那些长度为1的golden实体

    情况3：特殊实体，如：[14C]，其标注为: B-[ 14C ]-I
        GENIA tagger 分词后变为 B- [ 14C ] -I。标记被分离了
    解决3：在获取BIO标签时进行处理

    '''  
    previous = None
    changable = False
    if B_flag in word or I_flag in word:
        if word==B_flag:
            # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
            flag=2
            changable = 1
            print('B')
        elif word==I_flag:
            # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
            flag=0
            changable = 1
            print(word)
        if not changable:
            if word.startswith(B_flag):
                if word.count(B_flag) > word.count(I_flag):
                    # 嵌套实体①
                    label_sen.append('B')
                    flag=1
                    changable = 1
                elif word.count(B_flag) < word.count(I_flag):  # 实体结尾
                    # 嵌套实体②
                    label_sen.append('I')
                    flag=0
                    changable = 1
                else: # 单个实体
                    if flag:
                        label_sen.append('I')
                        flag=1
                    else:
                        label_sen.append('B')
                        flag=0
                    changable = 1
            elif word.endswith(I_flag):
                # 对应两种结尾情况：①/I-XXX ②/I-XXX/I-XXX
                label_sen.append('I')
                flag=0
                changable = 1
            else:
                # 非实体词
                pass
            
    if changable:
        word = word.replace(B_flag, '').replace(I_flag, '')
    else:
        if flag:
            if flag==2: # 针对‘[entity]’这种实体形式
                # print(word, flag)
                label_sen.append('B')
                flag=1
            else:   # flag=1
                label_sen.append('I')
                flag=1
        else:
            label_sen.append('O')
            flag=0

    return word, flag


# 根据预处理语料的标记 <B-xxx-></-I-xxx> 获取BIO标签
def getLabel():
    flag = 0    # 0:实体结束    1:实体内部  2:特殊实体[]
    label_sen = []
    sent = []
    geniaPath = '/Users/ningshixian/PycharmProjects/keras_bc6_track1/sample/data/BIBIO/test/test2.genia.txt'
    outputPath = '/Users/ningshixian/PycharmProjects/keras_bc6_track1/sample/data/BIBIO/test/test2.out.txt'

    with codecs.open(geniaPath, 'r', encoding='utf-8') as data:
        for line in data:
            if not line=='\n':
                words = line.split('\t')[0]
                word, flag = judge(words, label_sen, flag)
                if not word:
                    # 跳过单纯的标签 B^ 和 ^I
                    continue
                sent.append(word + '\t' + '\t'.join(line.split('\t')[2:-1]) + '\t' + label_sen[-1] + '\n')
            else:
                # label.append(label_sen)
                flag = 0
                label_sen = []
                sent.append('\n')

    with codecs.open(outputPath, 'w', encoding='utf-8') as f:
        for line in sent:
            f.write(line)



if __name__ == '__main__':
    
    tag_list = ['B‐^^', '^^‐I', 'B‐^', '^‐I']
    B_flag = 'B^'
    I_flag = '^I'

    # readXML()
    # print("完结撒花====")

    '''
    % cd geniatagger-3.0.2
    % ./geniatagger  /Users/ningshixian/PycharmProjects/keras_bc6_track1/sample/data/BIBIO/test/test2.txt \
    > /Users/ningshixian/PycharmProjects/keras_bc6_track1/sample/data/BIBIO/test/test2.genia.txt
    '''

    getLabel()
    print("完结撒花====")
    
    '''
    将词典特征加入到训练文件中
    '''
    outputPath = '/Users/ningshixian/PycharmProjects/keras_bc6_track1/sample/data/BIBIO/test/test.out.txt'
    outputPath2 = '/Users/ningshixian/PycharmProjects/keras_bc6_track1/sample/data/BIBIO/test/test2.out.txt'
    finalPath = '/Users/ningshixian/PycharmProjects/keras_bc6_track1/sample/data/BIBIO/test/test.final.txt'
    
    with codecs.open(outputPath, 'r', encoding='utf-8') as data:
        output = data.readlines()
    with codecs.open(outputPath2, 'r', encoding='utf-8') as data:
        output2 = data.readlines()
    
    results = []
    for i in range(len(output)):
        if output[i]=='\n':
            results.append('')
            continue
        line1 = output[i].replace('\n', '').strip()
        line2 = output2[i].replace('\n', '').strip()
        tmp = line1.split('\t')[:-1] + [line2.split('\t')[-1]] + [line1.split('\t')[-1]]
        results.append('\t'.join(tmp))
    with codecs.open(finalPath, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line)
            f.write('\n')