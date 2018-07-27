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


def readXML(files, BioC_PATH):
    num_sentence = 0
    num_file = 0
    num_annotations_pro = 0
    num_entitytype_pro = 0
    num_annotations_gene = 0
    num_entitytype_gene = 0
    passages_list = []
    raw_sentence_list = []
    id_list_list = []
    for file in files:  #遍历文件夹
        if not os.path.isdir(file):  #判断是否是文件夹，不是文件夹才打开
            f = BioC_PATH + "/" + file
            DOMTree = parse(f) # 使用minidom解析器打开 XML 文档
            collection = DOMTree.documentElement  # 得到了根元素

            # 在集合中获取所有 document
            documents = collection.getElementsByTagName("document")

            for document in documents:
                doc_id = document.getElementsByTagName("id")[0].childNodes[0].data
                passages = document.getElementsByTagName("passage")
                # print("*****passage*****")
                for passage in passages:
                    text = passage.getElementsByTagName('text')[0]
                    sentence_byte = text.childNodes[0].data.encode("utf-8")  # byte
                    sentence_str = sentence_byte.decode("utf-8")  # str
                    raw_sentence_list.append(sentence_str)
                    num_sentence += 1
                    id_list = []
                    offset_list = []
                    length_list = []
                    entity_list = []

                    annotations = passage.getElementsByTagName('annotation')
                    for annotation in annotations:
                        info = annotation.getElementsByTagName("infon")[0]
                        ID = info.childNodes[0].data
                        location = annotation.getElementsByTagName("location")[0]
                        offset = int(location.getAttribute("offset"))
                        length = int(location.getAttribute("length"))
                        txt = annotation.getElementsByTagName("text")[0]
                        entity = txt.childNodes[0].data
                        assert len(sentence_byte[offset:offset+length].decode('utf-8'))==len(entity) 
                        id_list.append(ID)
                        offset_list.append(offset)
                        length_list.append(length)
                        entity_list.append(entity)
                    
                    # 根据offset的大小对数组进行逆序排序
                    offset_sorted = sorted(enumerate(offset_list), key=lambda x:x[1], reverse=True)
                    offset_list = [x[1] for x in offset_sorted]  # 新数组
                    offset_idx = [x[0] for x in offset_sorted]  # 数组下标
                    length_list = [length_list[idx] for idx in offset_idx]
                    id_list = [id_list[idx] for idx in offset_idx]
                    entity_list = [entity_list[idx] for idx in offset_idx]

                    # 针对实体嵌套的情况
                    # 即两个实体的start/end相同，而长度不同，保留长的哪个
                    offset_temp = []
                    offset_remove = []
                    for i in range(len(offset_list)):
                        offset1 = offset_list[i]
                        length1 = length_list[i]
                        for j in range(i+1, len(offset_list)):
                            offset2 = offset_list[j]
                            length2 = length_list[j]
                            if offset1==offset2:  # 实体的start相同
                                offset_temp.append([i, j])
                            elif offset1+length1==offset2+length2:  # 实体的end相同
                                offset_temp.append([i, j])
                    while 1:
                        if offset_temp:
                            idx1 = offset_temp[0][0]
                            idx2 = offset_temp[0][1]
                            ID1 = id_list[idx1]
                            ID2 = id_list[idx2]
                            # 保留其中的蛋白质或基因实体，否则保留其中长度较长的实体
                            if xx(ID1) and not xx(ID2):
                                offset_remove.append(idx2)
                            elif not xx(ID1) and xx(ID2):
                                offset_remove.append(idx1)
                            else:
                                idx3 = idx2 if length_list[idx1]>length_list[idx2] else idx1
                                offset_remove.append(idx3)
                            offset_temp = offset_temp[1:]
                        else:
                            break

                    # 丢弃 offset_remove 中的实体
                    if offset_remove:
                        # print('{}: {}'.format(doc_id, offset_list))
                        offset_remove = sorted(offset_remove, reverse=True)
                        for idxidx in offset_remove:
                            offset_list.pop(idxidx)
                            length_list.pop(idxidx)
                            id_list.pop(idxidx)
                            entity_list.pop(idxidx)
                        # print('{}: {}'.format(doc_id, offset_list))
                        # print(entity_list)

                    # 用一对标签 <B^>entity<^I> 包裹筛选后的所有实体
                    tmp = sentence_byte
                    id_list_only = []   # 仅保留gene or protein的ID
                    for i in range(len(offset_list)):
                        offset = offset_list[i]
                        length = length_list[i]
                        ID = id_list[i]
                        entity = entity_list[i]

                        if isinstance(tmp, str):
                            tmp = tmp.encode("utf-8")
                            
                        if ID.startswith('Uniprot:') or ID.startswith('protein:'):
                            if ID.startswith('Uniprot:'):
                                num_annotations_pro+=1
                            elif ID.startswith('protein:'):
                                num_entitytype_pro+=1
                            id_list_only.append(ID.strip('\n').strip())
                            # # This solution will strip out (ignore) the characters in
                            # # question returning the string without them.
                            left = tmp[:offset].decode("utf-8", errors='ignore')
                            mid = tmp[offset:offset + length].decode("utf-8", errors='ignore')
                            right = tmp[offset + length:].decode("utf-8", errors='ignore')
                            tmp = left + ' ' + B_tag[0] + mid + I_tag[0] + ' ' + right
                            tmp = tmp.replace('   ', ' ').replace('  ', ' ')
                        elif ID.startswith('NCBI gene:') or ID.startswith('gene:'):
                            if ID.startswith('NCBI gene:'):
                                num_annotations_gene+=1
                            elif ID.startswith('gene:'):
                                num_entitytype_gene+=1
                            id_list_only.append(ID.strip('\n').strip())
                            # # This solution will strip out (ignore) the characters in
                            # # question returning the string without them.
                            left = tmp[:offset].decode("utf-8", errors='ignore')
                            mid = tmp[offset:offset + length].decode("utf-8", errors='ignore')
                            right = tmp[offset + length:].decode("utf-8", errors='ignore')
                            tmp = left + ' ' + B_tag[1] + mid + I_tag[1] + ' ' + right
                            tmp = tmp.replace('   ', ' ').replace('  ', ' ')
                        else:
                            # 暂时不考虑其他类别的实体
                            continue
                    if not id_list_only:
                        id_list_only.append('') # 不包含实体也要占位

                    if isinstance(tmp, bytes):
                        tmp = tmp.decode("utf-8")

                    tmp = ' '.join(tmp.split())  # 重构
                    # 对标点符号进行切分，但保留 ^ 用作标记识别符
                    for special in "!\"#$%'()*+,-./:;<=>?@[\\]_`{|}~":
                        tmp = tmp.replace(special, ' '+special+' ')
                    tmp = tmp.replace('°C', ' °C ')
                    tmp = tmp.replace('   ', ' ').replace('  ', ' ')
                    if '' in tmp.split():
                        print('tmp中存在空字符error\n')

                    passages_list.append(tmp)
                    id_list_list.append(id_list_only)

    with codecs.open(test_path + "/" + 'test.txt', 'w', encoding='utf-8') as f:
        for sentence in passages_list:
            f.write(sentence)
            f.write('\n')
    with codecs.open(test_path + "/" + 'test_goldenID.txt', 'w', encoding='utf-8') as f:
        for sentence in id_list_list:
            f.write('\t'.join(sentence))
            f.write('\n')
    with codecs.open(test_path + "/" + 'test_raw.txt', 'w', encoding='utf-8') as f:
        for sentence in raw_sentence_list:
            f.write(sentence)
            f.write('\n')
    passages_list = []
    del passages_list

    print('标注proID的实体的个数：{}'.format((num_annotations_pro)))
    print('标注pro类型的实体的个数：{}'.format((num_entitytype_pro)))
    print('标注geneID的实体的个数：{}'.format((num_annotations_gene)))
    print('标注gene类型的实体的个数：{}'.format((num_entitytype_gene)))
    print('passage 总数： {}'.format(num_sentence)) # 13697


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
    if B_tag[0] in word or I_tag[0] in word or B_tag[1] in word or I_tag[1] in word:
        if word==B_tag[0]:
            # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
            flag=2
            changable = 1
            print('B_protein')
        elif word==I_tag[0] or word==I_tag[1]:
            # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
            flag=0
            changable = 1
            print(word)
        elif word==B_tag[1]:
            flag=21
            changable = 1
            print('B_gene')
        if not changable:
            if word.startswith(B_tag[1]):
                if word.count(B_tag[1]) > word.count(I_tag[1]):
                    # 嵌套实体①
                    label_sen.append('B-gene')
                    flag=1
                    changable = 1
                elif word.count(B_tag[1]) < word.count(I_tag[1]):  # 实体结尾
                    # 嵌套实体②
                    label_sen.append('I-gene')
                    flag=0
                    changable = 1
                else: # 单个实体
                    if flag:
                        label_sen.append('I-gene')
                        flag=1
                    else:
                        label_sen.append('B-gene')
                        flag=0
                    changable = 1
            elif word.startswith(B_tag[0]):
                if word.count(B_tag[0]) > word.count(I_tag[0]):
                    # 嵌套实体①
                    label_sen.append('B-protein')
                    flag=1
                    changable = 1
                elif word.count(B_tag[0]) < word.count(I_tag[0]):  # 实体结尾
                    # 嵌套实体②
                    label_sen.append('I-protein')
                    flag=0
                    changable = 1
                else: # 单个实体
                    if flag:
                        label_sen.append('I-protein')
                        flag=1
                    else:
                        label_sen.append('B-protein')
                        flag=0
                    changable = 1
            elif word.endswith(I_tag[1]):
                # 对应两种结尾情况：①/I-XXX ②/I-XXX/I-XXX
                label_sen.append('I-gene')
                flag=0
                changable = 1
            elif word.endswith(I_tag[0]):
                # 对应两种结尾情况：①/I-XXX ②/I-XXX/I-XXX
                label_sen.append('I-protein')
                flag=0
                changable = 1
            else:
                # 非实体词
                pass
            
    if changable:
        word = word.replace(B_tag[1], '').replace(I_tag[1], '')
        word = word.replace(B_tag[0], '').replace(I_tag[0], '')
    else:
        if flag:
            if flag==2: # 针对‘[entity]’这种实体形式
                # print(word, flag)
                label_sen.append('B-protein')
                flag=1
            elif flag==21: 
                label_sen.append('B-gene')
                flag=1
            else:   # flag=1
                if label_sen[-1]=='B-protein':
                    label_sen.append('I-protein')
                elif label_sen[-1]=='B-gene':
                    label_sen.append('I-gene')
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
    geniaPath = dataPath+ '/' + 'test.genia.txt'
    outputPath = dataPath+ '/' + 'test.out.txt'

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
    with codecs.open(dataPath + '/' +'test.out.txt', encoding='utf-8') as f:
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

    B_tag = ['B‐^', 'B‐^^']   # '‐' != '-'
    I_tag = ['^‐I', '^^‐I']

    test_path = r'/Users/ningshixian/Desktop/BC6_Track1/test_corpus_20170804/test'
    BioC_PATH = r'/Users/ningshixian/Desktop/BC6_Track1/test_corpus_20170804/caption_bioc'
    files = os.listdir(BioC_PATH)  # 得到文件夹下的所有文件名称
    files.sort()
    
    readXML(files, BioC_PATH)
    print("完结撒花====")

    '''
    % cd geniatagger-3.0.2

    % ./geniatagger  /Users/ningshixian/Desktop/'BC6_Track1'/test_corpus_20170804/test/test.txt \
    > /Users/ningshixian/Desktop/'BC6_Track1'/test_corpus_20170804/test/test.genia.txt
    '''

    # getLabel(test_path)
    # print("完结撒花====")
    #
    # counts1 = []
    # with codecs.open(test_path + "/" + 'test_goldenID.txt', encoding='utf-8') as f:
    #     lines1 = f.readlines()
    # with open(test_path + '/' + 'label.txt') as f:
    #     lines2 = f.readlines()
    #
    # for i in range(len(lines1)):
    #     sentence1 = lines1[i].strip('\n')
    #     sentence2 = lines2[i].strip('\n')
    #     count1 = len(sentence1.split('\t')) if sentence1 else 0
    #     count2 = sentence2.count('B')
    #     if not count1 == count2:
    #         print(sentence1)
    #         print(sentence2)

