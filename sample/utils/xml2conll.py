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
train_path = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/train'
BioC_PATH = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/caption_bioc'
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
                    entity_list = []
                    entity2id = {}
                    num_sentence += 1
                    # print("text: %s" % sentence)
                    annotations = passage.getElementsByTagName('annotation')
                    for annotation in annotations:
                        info = annotation.getElementsByTagName("infon")[0]
                        ID = info.childNodes[0].data
                        location = annotation.getElementsByTagName("location")[0]
                        offset = location.getAttribute("offset")
                        length = location.getAttribute("length")
                        # print "offset: %s" % offset
                        # print "length: %s" % length
                        annotation_txt = annotation.getElementsByTagName("text")[0]
                        entity = annotation_txt.childNodes[0].data
                        if entity not in entity_list:
                            entity_list.append(entity)  # 每个golden实体仅访问一次
                            entity2id[entity] = ID

                    entity_list = sorted(entity_list, key=lambda x:len(x), reverse=True)
                    for i in range(len(entity_list)):
                        entity = entity_list[i]
                        ID = entity2id[entity]
                        if ID.startswith('NCBI taxon:'):
                            new_entity = ' B-ORGANISMS-'+entity+'-I-ORGANISMS '
                        elif ID.startswith('NCBI gene:'):
                            new_entity = ' B-GENE-'+entity+'-I-GENE '
                        elif ID.startswith('GO:'):
                            new_entity = ' B-CELLULAR-'+entity+'-I-CELLULAR '
                        elif ID.startswith('Rfam:'):
                            new_entity = ' B-RNA-'+entity+'-I-RNA '
                        elif ID.startswith('PubChem:'):
                            new_entity = ' B-MOLECULES-'+entity+'-I-MOLECULES '
                        elif ID.startswith('Uniprot:'):
                            new_entity = ' B-PROTEIN-'+entity+'-I-PROTEIN '
                        elif ID.startswith('Uberon:'):
                            new_entity = ' B-TISSUES-'+entity+'-I-TISSUES '
                        elif ID.startswith('CHEBI:'):
                            new_entity = ' B-MOLECULES-'+entity+'-I-MOLECULES '
                        elif ID.startswith('CL:'):
                            new_entity = ' B-CELLTYPE-'+entity+'-I-CELLTYPE '
                        elif ID.startswith('CVCL_'):
                            new_entity = ' B-CELLTYPE-'+entity+'-I-CELLTYPE '

                        elif ID.startswith('organism:'):
                            new_entity = ' B-ORGANISMS-'+entity+'-I-ORGANISMS '
                        elif ID.startswith('gene:'):
                            new_entity = ' B-GENE-'+entity+'-I-GENE '
                        elif ID.startswith('protein:'):
                            new_entity = ' B-PROTEIN-'+entity+'-I-PROTEIN '
                        elif ID.startswith('tissue:'):
                            new_entity = ' B-TISSUES-'+entity+'-I-TISSUES '
                        elif ID.startswith('cell:'):
                            new_entity = ' B-CELLULAR-'+entity+'-I-CELLULAR '
                        elif ID.startswith('subcellular:'):
                            new_entity = ' B-CELLULAR-'+entity+'-I-CELLULAR '
                        elif ID.startswith('molecule:'):
                            new_entity = ' B-MOLECULES-'+entity+'-I-MOLECULES '
                        elif ID.startswith('Corum:'):
                            pass
                        elif ID.startswith('BAO:'):
                            pass
                        else:
                            print(ID)
                        sentence = sentence.replace(entity, new_entity)  # 对句子中所有的实体做上标记（空格切分嵌套实体）
                    # 处理嵌套实体惹的祸
                    sentence = sentence.replace('   ', ' ').replace('  ', ' ')
                    sentence = sentence.replace(' B-CELLULAR- ', ' B-CELLULAR-').replace(' -I-CELLULAR ', '-I-CELLULAR ')
                    sentence = sentence.replace(' B-MOLECULES- ', ' B-MOLECULES-').replace(' -I-MOLECULES ', '-I-MOLECULES ')
                    sentence = sentence.replace(' B-TISSUES- ', ' B-TISSUES-').replace(' -I-TISSUES ', '-I-TISSUES ')
                    sentence = sentence.replace(' B-PROTEIN- ', ' B-PROTEIN-').replace(' -I-PROTEIN ', '-I-PROTEIN ')
                    sentence = sentence.replace(' B-GENE- ', ' B-GENE-').replace(' -I-GENE ', '-I-GENE ')
                    sentence = sentence.replace(' B-ORGANISMS- ', ' B-ORGANISMS-').replace(' -I-ORGANISMS ', '-I-ORGANISMS ')
                    sentence = sentence.replace(' B-CELLTYPE- ', ' B-CELLTYPE-').replace(' -I-CELLTYPE ', '-I-CELLTYPE ')
                    sentence = sentence.replace(' B-RNA- ', ' B-RNA-').replace(' -I-RNA ', '-I-RNA ')
                    s.append(sentence)

    with open(BioC_PATH + "/" + 'train.txt', 'a') as f:
        for sentence in s:
            f.write(sentence)
            f.write('\n')
    print('句子总数：{}'.format(num_sentence))


# 利用GENIA tagger工具对标记过的语料进行预处理（分词+POS+CHUNK+NER）
# 得到 train.genia 文件


def judge(word, label_sen, entityTypes, flag, preType):
    changable = None
    for entityType in entityTypes:
        if word==('B-'+entityType+'-'):
            flag=2
            changable = entityType
            break
        elif word==('-I-'+entityType):
            flag=0
            changable = entityType
            break
        elif word.startswith('B-'+entityType+'-') and word.endswith('-I-'+entityType):
            if flag:
                label_sen.append('I-'+entityType)
                flag = 1
            else:
                label_sen.append('B-'+entityType)
                flag = 0
            changable = entityType
            break
        # elif word.endswith('-INSIDE'):
        elif '-I-'+entityType in word:
            label_sen.append('I-'+entityType)
            flag=0
            changable = entityType
            break
        elif word.startswith('B-'+entityType+'-'):
            label_sen.append('B-'+entityType)
            flag=1
            changable = entityType
            break

    if changable:
        preType = changable
        word = word.replace('B-'+changable+'-', '').replace('-I-'+changable, '')
    else:
        if flag:
            if flag==2:# 针对‘[entity]’这种实体形式下的'B-XX['标注
                label_sen.append('B-'+preType)
                flag=1
            else:
                label_sen.append('I-'+preType)
        else:
            label_sen.append('O')
    return word, flag, preType


# 根据预处理语料的标记 <B-xxx></I-xxx> 获取BIO标签
def getLabel(dataPath):
    flag = 0    # 0:实体结束    1:实体内部  2:特殊实体[]
    preType = None
    # label = []
    label_sen = []
    sent = []
    with open(dataPath+ '/' + 'train.genia', 'r') as data:
        for line in data:
            if not line=='\n':
                word = line.split('\t')[0]
                entityTypes = ['ORGANISMS', 'GENE', 'CELLULAR', 'RNA', 'MOLECULES', 
                                'PROTEIN', 'TISSUES', 'CELLTYPE']
                word, flag, preType = judge(word, label_sen, entityTypes, flag, preType)
                if not word:
                    continue
                sent.append(word + '\t' + '\t'.join(line.split('\t')[2:-1]) + '\t' + label_sen[-1] + '\n')
                # sent.append(word + '\t' + '\t'.join(line.split('\t')[1:-1]) + '\t' + label_sen[-1] + '\n')
            else:
                # label.append(label_sen)
                label_sen = []
                sent.append('\n')

    with open(dataPath+ '/' + 'train.out', 'w') as f:
        for line in sent:
            f.write(line)
    return label


if __name__ == '__main__':
    
    # readXML(files)

    # GENIA tagger
    '''
    cd geniatagger-3.0.2
    ./geniatagger  /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/train/train.txt \
    > /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/train/train.genia
    '''

    getLabel(train_path)

    print("完结撒花====")
