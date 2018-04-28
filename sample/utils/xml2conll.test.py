#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
     1、从196篇 .XML 原始文件中解析获取数据和标签文件
     3、通过offset，将句子中所有的实体用一对标签 <B-*/>entity</I-*> 进行标记
     4、利用GENIA tagger工具对标记过的语料进行预处理
     5、根据预处理语料的标记 <B-*/></I-*> 获取BIO标签
'''
from xml.dom.minidom import parse
import re
import codecs
import os


def readXML(files, BioC_PATH):
    num_passage = 0
    num_file = 0
    passages_list = []
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
                    # for specific_symbol in ['-', '°C']:
                    #     sentence = sentence.replace(specific_symbol, ' '+specific_symbol+' ')
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

                    # 将句子中的所有实体分别用一对标签 <B-*/>entity</I-*> 进行标记
                    for i in range(len(offset_list)):
                        tmp = sentence
                        offset = offset_list[i]
                        length = length_list[i]
                        ID = id_list[i]
                        # print(tmp[offset:offset+length].decode("utf-8"))
                        if ID.startswith('NCBI gene:') or ID.startswith('Uniprot:') or \
                            ID.startswith('gene:') or ID.startswith('protein:'):
                            tmp = tmp[:offset].decode("utf-8") + ' ' + left + tmp[offset:offset+length].decode("utf-8") + right + ' ' + tmp[offset+length:].decode("utf-8")
                            tmp = tmp.replace('   ', ' ').replace('  ', ' ')
                        elif ID.startswith('Rfam:') or ID.startswith('mRNA:'):
                            tmp = tmp[:offset].decode("utf-8") + ' ' + left + tmp[offset:offset+length].decode("utf-8") + right + ' ' + tmp[offset+length:].decode("utf-8")
                            tmp = tmp.replace('   ', ' ').replace('  ', ' ')
                        else:
                            # 暂时不考虑其他类别的实体
                            continue

                        # 将标记的实体替换到新句子中
                        splited_tmp = tmp.split()   # str
                        splited_sen = sen_new.split()   # str
                        for i in range(len(splited_sen)):
                            item = splited_tmp[i]
                            if left in item or right in item:
                                if left in splited_sen[i] and right in splited_sen[i]:
                                    k1 = splited_sen[i].index(left)+4
                                    k2 = splited_sen[i].index(right)
                                    splited_sen[i] = splited_sen[i][:k1] + item + splited_sen[i][k2:]
                                elif left in splited_sen[i]:
                                    k1 = splited_sen[i].index(left)+4
                                    splited_sen[i] = splited_sen[i][:k1] + item
                                elif right in splited_sen[i]:
                                    k2 = splited_sen[i].index(right)
                                    splited_sen[i] = item + splited_sen[i][k2:]
                                else:
                                    splited_sen[i] = item
                        sen_new = ' '.join(splited_sen)
                        # if not len(splited_sen)==len(splited_tmp):
                        #     print(splited_tmp)
                        #     print(sen_new.split())
                    sen_new = sen_new.replace('   ', ' ').replace('  ', ' ')
                    passages_list.append(sen_new)

    with codecs.open(test_path + "/" + 'test.txt', 'w', encoding='utf-8') as f:
        for sentence in passages_list:
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
    changable = None
    if 'B-' in word or '/I-' in word:
        if word==left:
            # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
            flag=2
            changable = 1
            print('left')
        elif word==right:
            # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
            flag=0
            changable = 1
            print('right')
        if not changable:
            if word.startswith('B-'):
                # 获取实体B-XXX的类型
                entityType = None
                if word.startswith('B-'):
                    entityType = 1
                if entityType:
                    if word.count('B-') > word.count('/I-'):
                        # 嵌套实体①
                        label_sen.append('B')
                        flag=1
                        changable = 1
                    elif word.count('B-') < word.count('/I-'):  # 实体结尾
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
                else:
                    # 针对普通词，如：B-PRO
                    print(word)
                    flag=0
            elif '/I-' in word:
                # 对应两种结尾情况：①/I-XXX ②/I-XXX/I-XXX
                label_sen.append('I')
                flag=0
                changable = 1
            else:
                # 非实体词
                pass
            
    if changable:
        word = word.replace('B-*/', '').replace('/I-*', '')
    else:
        if flag:
            if flag==2: # 针对‘[entity]’这种实体形式
                # print(word, flag)
                label_sen.append('B')
                flag=1
            else:
                label_sen.append('I')
        else:
            label_sen.append('O')
            flag=0

    return word, flag


# 根据预处理语料的标记 <B-xxx-></-I-xxx> 获取BIO标签
def getLabel(dataPath, train_or_devel):
    flag = 0    # 0:实体结束    1:实体内部  2:特殊实体[]
    preType = None
    label_sen = []
    sent = []
    if train_or_devel=='test':
        geniaPath = dataPath+ '/' + 'test.genia.txt'
        outputPath = dataPath+ '/' + 'test.out.txt'

    with codecs.open(geniaPath, 'r', encoding='utf-8') as data:
        for line in data:
            if not line=='\n':
                word = line.split('\t')[0]
                word, flag = judge(word, label_sen, flag)
                if not word:
                    # 跳过 B-xxx-和-I-xxx
                    continue
                sent.append(word + '\t' + '\t'.join(line.split('\t')[2:-1]) + '\t' + label_sen[-1] + '\n')
            else:
                # label.append(label_sen)
                flag, preType = 0, None
                label_sen = []
                sent.append('\n')

    with codecs.open(outputPath, 'w', encoding='utf-8') as f:
        for line in sent:
            f.write(line)


if __name__ == '__main__':

    # entityTypes = ['**', '*']
    # star2Type = {'*':'GENE', '**':'PROTEIN'}
    left = 'B-*/'
    right = '/I-*'

    test_path = r'/Users/ningshixian/Desktop/BC6_Track1/test_corpus_20170804/test'
    BioC_PATH = r'/Users/ningshixian/Desktop/BC6_Track1/test_corpus_20170804/caption_bioc'
    files = os.listdir(BioC_PATH)  # 得到文件夹下的所有文件名称
    files.sort()
    
    # readXML(files, BioC_PATH)

    '''
    % cd geniatagger-3.0.2

    % ./geniatagger  /Users/ningshixian/Desktop/'BC6_Track1'/test_corpus_20170804/test/test.txt \
    > /Users/ningshixian/Desktop/'BC6_Track1'/test_corpus_20170804/test/test.genia.txt
    '''

    getLabel(test_path, 'test')

    print("完结撒花====")
