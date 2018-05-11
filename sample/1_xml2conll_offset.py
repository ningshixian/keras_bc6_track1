#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
     1、从 .XML 原始文件中解析获取数据和标签文件
     2、将570篇 document 分为训练455篇 document 和验证115篇 document
     3、通过offset，将句子中所有的实体用一对标签 <B*/>entity</I*> 进行标记
     4、利用GENIA tagger工具对标记过的语料进行预处理
     5、根据预处理语料的标记 <B*/></I*> 获取BIO标签
'''
from xml.dom.minidom import parse
import re
import codecs
import os


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


def readXML(files, BioC_PATH):
    num_passage = 0
    num_file = 0
    passages_list = []
    raw_passages_list = []
    for file in files:  #遍历文件夹
        if not os.path.isdir(file):  #判断是否是文件夹，不是文件夹才打开
            f = BioC_PATH + "/" + file
            try:
                DOMTree = parse(f) # 使用minidom解析器打开 XML 文档
                collection = DOMTree.documentElement  # 得到了根元素
            except:
                print(f)
                continue

            # 在集合中获取所有 document
            documents = collection.getElementsByTagName("document")

            for document in documents:
                passages = document.getElementsByTagName("passage")
                # print("*****passage*****")
                for passage in passages:
                    text = passage.getElementsByTagName('text')[0]
                    sentence = text.childNodes[0].data.encode("utf-8")  # byte
                    sen_new = sentence.decode("utf-8")  # str
                    raw_passages_list.append(sen_new)
                    entity_list = []
                    id_list = []
                    offset_list = []
                    length_list = []
                    num_passage += 1
                    annotations = passage.getElementsByTagName('annotation')
                    for annotation in annotations:
                        info = annotation.getElementsByTagName("infon")[0]
                        ID = info.childNodes[0].data
                        location = annotation.getElementsByTagName("location")[0]
                        offset = int(location.getAttribute("offset"))
                        length = int(location.getAttribute("length"))
                        annotation_txt = annotation.getElementsByTagName("text")[0]
                        entity = annotation_txt.childNodes[0].data
                        assert len(sentence[offset:offset+length].decode('utf-8'))==len(entity) 
                        id_list.append(ID)
                        offset_list.append(offset)
                        length_list.append(length)
                    
                    # 根据offset的大小对数组进行逆序排序
                    offset_sort = sorted(enumerate(offset_list), key=lambda x:x[1], reverse=True)
                    offset_list = [x[1] for x in offset_sort]
                    offset_idx = [x[0] for x in offset_sort]  # 下标
                    length_list = [length_list[idx] for idx in offset_idx]
                    id_list = [id_list[idx] for idx in offset_idx]

                    # if num_passage==111:
                    #     print(offset_list)
                    #     print(id_list)

                    # 针对实体嵌套的情况（即两个实体的offset相同，而长度不同）
                    # 保留其中长的实体的 offset
                    idx = []
                    idx_remove = []
                    for i in range(len(offset_list)):
                        offset1 = offset_list[i]
                        if i+1<len(offset_list):
                            offset2 = offset_list[i+1]
                        else:
                            continue
                        if offset1==offset2:
                            idx.append([i, i+1])
                    while 1:
                        if idx:
                            idx1 = idx[0][0]
                            idx2 = idx[0][1]
                            idx3 = idx2 if length_list[idx1]>length_list[idx2] else idx1
                            idx_remove.append(idx3)
                            idx = idx[1:]
                        else:
                            break
                    if idx_remove:
                        print(num_passage)
                        print(offset_list)
                        # print(id_list)
                        idx_remove = sorted(idx_remove, reverse=True)
                        for idxidx in idx_remove:
                            offset_list.pop(idxidx)
                            length_list.pop(idxidx)
                            id_list.pop(idxidx)
                        print(offset_list)
                        # print(id_list)

                    # 将句子中的所有实体分别用一对标签 <B*/>entity</I*> 进行标记
                    tmp = sentence
                    for i in range(len(offset_list)):
                        offset = offset_list[i]
                        length = length_list[i]
                        ID = id_list[i]
                        if isinstance(tmp, str):
                            tmp = tmp.encode("utf-8")
                        
                        # This solution will strip out (ignore) the characters in 
                        # question returning the string without them. 
                        left = tmp[:offset].decode("utf-8", errors='ignore')
                        mid = tmp[offset:offset+length].decode("utf-8", errors='ignore')
                        right = tmp[offset+length:].decode("utf-8", errors='ignore')
                        
                        if ID.startswith('NCBI gene:') or ID.startswith('Uniprot:') or \
                            ID.startswith('gene:') or ID.startswith('protein:'):
                            tmp = left + ' ' + B_tag + mid + I_tag + ' ' + right
                            tmp = tmp.replace('   ', ' ').replace('  ', ' ')
                        # elif ID.startswith('Rfam:') or ID.startswith('mRNA:'):
                        #     tmp = left + ' ' + B_tag + mid + I_tag + ' ' + right
                        #     tmp = tmp.replace('   ', ' ').replace('  ', ' ')
                        else:
                            # 暂时不考虑其他类别的实体
                            continue

                    if isinstance(tmp, bytes):
                        tmp = tmp.decode("utf-8")
                    if num_passage==4628:    # 101
                        print(file)

                    for specific_symbol in ['-', '°C']:
                        tmp = tmp.replace(specific_symbol, ' '+specific_symbol+' ')
                        tmp = tmp.replace('  ', ' ')
                    passages_list.append(tmp)

    with codecs.open(train_path + "/" + 'train.txt', 'w', encoding='utf-8') as f:
        for sentence in passages_list:
            f.write(sentence)
            f.write('\n')
    with codecs.open(train_path + "/" + 'train_raw.txt', 'w', encoding='utf-8') as f:
        for sentence in raw_passages_list:
            f.write(sentence)
            f.write('\n')
    passages_list = []
    print('passage 总数： {}'.format(num_passage)) # 13697


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
    changable = False
    if 'B*' in word or 'I*' in word:
        if word==B_tag:
            # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
            flag=2
            changable = 1
            print('B_tag')
        elif word==I_tag:
            # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
            flag=0
            changable = 1
            print('I_tag')
        if not changable:
            if word.startswith(B_tag):
                if word.count(B_tag) > word.count(I_tag):
                    # 嵌套实体①
                    label_sen.append('B')
                    flag=1
                    changable = 1
                elif word.count(B_tag) < word.count(I_tag):  # 实体结尾
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
            elif I_tag in word:
                # 对应两种结尾情况：①/I-XXX ②/I-XXX/I-XXX
                label_sen.append('I')
                flag=0
                changable = 1
            else:
                # 非实体词
                pass
            
    if changable:
        word = word.replace(B_tag, '').replace(I_tag, '')
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
                    # 跳过单纯的标签 B*/ 和 /I*
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

    # entityTypes = ['**', '*']
    # star2Type = {'*':'GENE', '**':'PROTEIN'}
    B_tag = 'B*/'
    I_tag = '/I*'

    train_path = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/train'
    BioC_PATH = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/caption_bioc'
    files = os.listdir(BioC_PATH)  # 得到文件夹下的所有文件名称
    files.sort()
    
    readXML(files, BioC_PATH)

    '''
    % cd geniatagger-3.0.2
    % ./geniatagger  /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/train/train.txt \
    > /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/train/train.genia.txt
    '''

    # getLabel(train_path)

    print("完结撒花====")
