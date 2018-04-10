#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
     1、从 .XML 原始文件中解析获取数据和标签文件
     2、将句子中所有的实体用一对标签 <B-xxx>entity</I-xxx> 进行标记
     3、利用GENIA tagger工具对标记过的语料进行预处理
     4、根据预处理语料的标记 <B-xxx></I-xxx> 获取BIO标签
'''
from xml.dom.minidom import parse
import re
import codecs
import os

data = []
label = []
train_path = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtest/test'
BioC_PATH = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtest/caption_bioc_unannotated'
files = os.listdir(BioC_PATH)  #得到文件夹下的所有文件名称


def readXML(files):
    num_sentence = 0
    s = []
    for file in files:  #遍历文件夹
        if not os.path.isdir(file):  #判断是否是文件夹，不是文件夹才打开
            f = BioC_PATH + "/" + file
            try:
                DOMTree = parse(f) # 使用minidom解析器打开 XML 文档
                collection = DOMTree.documentElement  # 得到了根元素
                # for node in collection.childNodes:  # 子结点的访问
                #     if node.nodeType == node.ELEMENT_NODE:
                #         print node.nodeName
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
                    sentence = text.childNodes[0].data
                    num_sentence +=1
                    s.append(sentence)

    with open(BioC_PATH + "/" + 'test.txt', 'w') as f:
        for sentence in s:
            f.write(sentence)
            f.write('\n')
    print('句子总数：{}'.format(num_sentence))


# 利用GENIA tagger工具对标记过的语料进行预处理（分词+POS+CHUNK+NER）
# 得到 train.genia 文件


# 根据预处理语料的标记 <B-xxx></I-xxx> 获取BIO标签
def getLabel(dataPath):
    flag = 0    # 0:实体结束    1:实体内部  2:特殊实体[]
    preType = None
    sent = []
    with open(dataPath+ '/' + 'test.genia', 'r') as data:
        for line in data:
            if not line=='\n':
                word = line.split('\t')[0]
                sent.append(word + '\t' + '\t'.join(line.split('\t')[2:-1]) + '\n')
                # sent.append(word + '\t' + '\t'.join(line.split('\t')[1:-1]) + '\t' + label_sen[-1] + '\n')
            else:
                # label.append(label_sen)
                label_sen = []
                sent.append('\n')

    with open(dataPath+ '/' + 'test.out', 'w') as f:
        for line in sent:
            f.write(line)
    return label


if __name__ == '__main__':
    
    # readXML(files)

    # GENIA tagger
    '''
    cd geniatagger-3.0.2
    ./geniatagger  /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtest/test/test.txt \
    > /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtest/test/test.genia
    '''

    getLabel(train_path)

    print("完结撒花====")
