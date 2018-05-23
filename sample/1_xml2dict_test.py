#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
    BioC(XML)格式ConNLL格式
    从 .XML 原始document文件中解析获取训练数据和标签文件

    1、通过offset，将句子中所有的实体用一对标签 <B^>entity<^I> 进行标记
       注意：offset 是在二进制编码下索引的，要对句子进行编码 s=s.encode(‘utf-8’)
    2、对于嵌套实体（offset 相同），仅保留其中长度较长的实体
    3、对句子中的标点符号  !\”#$%‘()*+,-./:;<=>?@[\\]_`{|}~ 进行切分；^ 保留用于实体标签！！
    4、利用GENIA tagger工具对标记过的语料进行分词和词性标注
    5、根据预处理语料的标记 <B^><^I> 获取BIO标签
'''
from xml.dom.minidom import parse
import re
import codecs
import os
from tqdm import tqdm

# def entityReplace(splited_sen, splited_tagged, i, item, sen_length):
#     '''
#     将句子中的实体用<></>标签包裹
#     '''
#     # 1、先处理嵌套实体的问题
#     if B_tag in splited_sen[i] and I_tag in splited_sen[i]:
#         k1 = splited_sen[i].index(B_tag)+4
#         k2 = splited_sen[i].index(I_tag)
#         splited_sen[i] = splited_sen[i][:k1] + item + splited_sen[i][k2:]
#     elif B_tag in splited_sen[i]:
#         k1 = splited_sen[i].index(B_tag)+4
#         splited_sen[i] = splited_sen[i][:k1] + item
#     elif I_tag in splited_sen[i]:
#         k2 = splited_sen[i].index(I_tag)
#         splited_sen[i] = item + splited_sen[i][k2:]
#     else:
#         splited_sen[i] = item
#     # 2、对于嵌入在单词内部的实体，包裹标签后 需要重新调整句子的长度
#     gap = i+1
#     diff = len(splited_tagged) - sen_length    # 标记后的句子与原始句子的长度差
#     while diff:
#         splited_sen.insert(gap, splited_tagged[gap])
#         diff-=1
#         gap+=1


def xx(entity):
    if entity.startswith('NCBI gene:') or entity.startswith('Uniprot:') or \
    entity.startswith('gene:') or entity.startswith('protein:'):
        return True
    else:
        return False


def readKB():
    word_list=[]
    pro_path = '/home/administrator/PycharmProjects/embedding/uniprot_sprot.dat2'
    gene_path = '/home/administrator/PycharmProjects/embedding/gene_info2'

    with open(pro_path) as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        splited = line.split('\t')
        e_list = splited[1].replace('\n', '').split(';')
        # word_list.extend(e_list)
        for e in e_list:
            e = e.strip()
            if len(e)>1 and not e.isdigit():
                word_list.append(e)

    with open(gene_path) as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        splited = line.split('\t')
        e_list = splited[1].replace('\n', '').split(';')
        # word_list.extend(e_list)
        for e in e_list:
            e = e.strip()
            if len(e)>2 and not e.isdigit():
                word_list.append(e)

    return list(set(word_list))


def max_match_cut(sentence, dic):
    """
    将字典中的所有词与输入句子进行完全匹配，若匹配，则分配"IBES"；否则，分配"O"
    :param sentence:输入句子
    :param dic:蛋白质词典
    :return:字典特征
    """
    sentence = sentence     # .decode("utf-8", errors='ignore')
    tmp = sentence
    splited_s = sentence.split()
    result = dic.query(sentence)
    result = list(set(result))
    # # 先按照实体长度排序
    # en_list = [len(r[1]) for r in result]
    # en_sorted = sorted(enumerate(en_list), key=lambda x:x[1], reverse=True)
    # en_idx = [x[0] for x in en_sorted]  # 数组下标
    # result = [result[idx] for idx in en_idx]

    # 再按照offset排序
    offset_list = [r[0][0] for r in result]
    offset_sorted = sorted(enumerate(offset_list), key=lambda x:x[1], reverse=True)
    offset_idx = [x[0] for x in offset_sorted]  # 数组下标
    result = [result[idx] for idx in offset_idx]

    def filter(result):
        p1 = 0
        p2 = 0
        id_list = []
        temp = None
        for i in range(len(result)):
            start = result[i][0][0]
            end = result[i][0][1]
            # if i+1<len(result):
            #     start_next = result[i+1][0][0]
            #     end_next = result[i+1][0][1]
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
    if result:
        print(result)

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
                # print(len(tmp.split()) , len(splited_s))
                # print(tmp.split())
                # print(splited_s)
                tmp = tmp_tmp
                continue
            assert len(tmp.split()) == len(splited_s)

    return tmp



def readXML():
    
    import esm

    word_list = readKB()
    word_list.append('PPCA')
    # word_list = ['PPCA']

    print('获取字典树trie')
    dic = esm.Index()
    for i in range(len(word_list)):
        word = word_list[i]
        dic.enter(word)
    dic.fix()

    # word_list = readKB()
    # word_list.append('PPCA')

    print('最大匹配')
    results = []
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test.txt') as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        for tag in tag_list:
            line = line.replace(tag, '')
        line = max_match_cut(line, dic)
        # label = max_match_cut2(line, word_list)
        results.append(line)

    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test2.txt', 'w') as f:
        for sentence in results:
            f.write(sentence)



# 利用GENIA tagger工具对标记过的语料进行预处理（分词+POS+CHUNK+NER）
# 得到 train.genia 文件


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
def getLabel(dataPath):
    flag = 0    # 0:实体结束    1:实体内部  2:特殊实体[]
    label_sen = []
    sent = []
    geniaPath = dataPath+ '/' + 'train.genia.txt'
    outputPath = dataPath+ '/' + 'train.out.txt'

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

    # 生成单独的BIO标签文件
    label_sen = []
    ff = open(dataPath + '/' +'label.txt', 'w')
    with codecs.open(dataPath + '/' +'train.out.txt', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        if line=='\n':
            ff.write(''.join(label_sen))
            ff.write('\n')
            label_sen = []
        else:
            label = line.split('\t')[-1]
            label_sen.append(label.strip('\n')[0])
    ff.close()



if __name__ == '__main__':
    tag_list = ['B‐^^', '^^‐I', 'B‐^', '^‐I']
    B_flag = 'B^'
    I_flag = '^I'

    readXML()
    print("完结撒花====")

    '''
    % cd geniatagger-3.0.2
    % ./geniatagger  /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/train/train.txt \
    > /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/train/train.genia.txt
    '''

    # getLabel(train_path)
    # print("完结撒花====")

    # with codecs.open(train_path + "/" + 'train_goldenID.txt', encoding='utf-8') as f:
    #     lines1 = f.readlines()
    # with codecs.open(train_path + '/' + 'label.txt', encoding='utf-8') as f:
    #     lines2 = f.readlines()

    # for i in range(len(lines1)):
    #     sentence1 = lines1[i].strip('\n')
    #     sentence2 = lines2[i].strip('\n')
    #     count1 = len(sentence1.split('\t')) if sentence1 else 0
    #     count2 = sentence2.count('B')
    #     if not count1 == count2:
    #         print(i)
    #         print(count1, count2)
    #         print(sentence1)
    #         print(sentence2)
