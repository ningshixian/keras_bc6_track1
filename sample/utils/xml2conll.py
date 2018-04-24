#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
     1、从 .XML 原始文件中解析获取数据和标签文件
     2、将570篇 document 分为训练455篇 document 和验证115篇 document
     3、将句子中所有的实体用一对标签 <B-*/>entity</I-*> 进行标记
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
                    for specific_symbol in ['-', '°C']:
                        sentence = sentence.replace(specific_symbol, ' '+specific_symbol+' ')
                    entity_list = []
                    entity2id = {}
                    num_passage += 1
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
                        if entity not in entity_list and len(entity)>1:
                            '''每个golden实体仅访问一次，且丢弃那些长度为1的实体！'''
                            entity_list.append(entity)  
                            entity2id[entity] = ID
                    
                    # 按实体长度逆序排序
                    entity_list = sorted(entity_list, key=lambda x:len(x), reverse=True)
                    for i in range(len(entity_list)):
                        entity = entity_list[i]
                        new_entity = entity
                        ID = entity2id[entity]
                        if ID.startswith('NCBI gene:'):
                            new_entity = ' B-*/'+entity+'/I-* '
                        elif ID.startswith('Uniprot:'):
                            new_entity = ' B-**/'+entity+'/I-** '
                        # elif ID.startswith('NCBI taxon:'):
                        #     new_entity = ' B-ORGANISMS-'+entity+'-I-ORGANISMS '
                        # elif ID.startswith('GO:'):
                        #     new_entity = ' B-CELLULAR-'+entity+'-I-CELLULAR '
                        # elif ID.startswith('Rfam:'):
                        #     new_entity = ' B-RNA-'+entity+'-I-RNA '
                        # elif ID.startswith('PubChem:'):
                        #     new_entity = ' B-MOLECULES-'+entity+'-I-MOLECULES '
                        # elif ID.startswith('Uberon:'):
                        #     new_entity = ' B-TISSUES-'+entity+'-I-TISSUES '
                        # elif ID.startswith('CHEBI:'):
                        #     new_entity = ' B-MOLECULES-'+entity+'-I-MOLECULES '
                        # elif ID.startswith('CL:'):
                        #     new_entity = ' B-CELLTYPE-'+entity+'-I-CELLTYPE '
                        # elif ID.startswith('CVCL_'):
                        #     new_entity = ' B-CELLTYPE-'+entity+'-I-CELLTYPE '

                        elif ID.startswith('gene:'):
                            new_entity = ' B-*/'+entity+'/I-* '
                        elif ID.startswith('protein:'):
                            new_entity = ' B-**/'+entity+'/I-** '
                        # elif ID.startswith('organism:'):
                        #     new_entity = ' B-ORGANISMS-'+entity+'-I-ORGANISMS '
                        # elif ID.startswith('tissue:'):
                        #     new_entity = ' B-TISSUES-'+entity+'-I-TISSUES '
                        # elif ID.startswith('cell:'):
                        #     new_entity = ' B-CELLULAR-'+entity+'-I-CELLULAR '
                        # elif ID.startswith('subcellular:'):
                        #     new_entity = ' B-CELLULAR-'+entity+'-I-CELLULAR '
                        # elif ID.startswith('molecule:'):
                        #     new_entity = ' B-MOLECULES-'+entity+'-I-MOLECULES '
                        # elif ID.startswith('Corum:'):
                        #     pass
                        # elif ID.startswith('BAO:'):
                        #     pass
                        # else:
                        #     print('未考虑{}'.format(ID))
                        sentence = sentence.replace(entity, new_entity)  # 对句子中所有的实体做上标记（空格切分嵌套实体）
                    # 处理嵌套实体惹的祸
                    sentence = sentence.replace('   ', ' ').replace('  ', ' ')
                    sentence = sentence.replace(' B-*/ ', ' B-*/').replace(' /I-* ', '/I-* ')
                    sentence = sentence.replace(' B-**/ ', ' B-**/').replace(' /I-** ', '/I-** ')
                    # sentence = sentence.replace(' B-CELLULAR- ', ' B-CELLULAR-').replace(' -I-CELLULAR ', '-I-CELLULAR ')
                    # sentence = sentence.replace(' B-MOLECULES- ', ' B-MOLECULES-').replace(' -I-MOLECULES ', '-I-MOLECULES ')
                    # sentence = sentence.replace(' B-TISSUES- ', ' B-TISSUES-').replace(' -I-TISSUES ', '-I-TISSUES ')
                    # sentence = sentence.replace(' B-ORGANISMS- ', ' B-ORGANISMS-').replace(' -I-ORGANISMS ', '-I-ORGANISMS ')
                    # sentence = sentence.replace(' B-CELLTYPE- ', ' B-CELLTYPE-').replace(' -I-CELLTYPE ', '-I-CELLTYPE ')
                    # sentence = sentence.replace(' B-RNA- ', ' B-RNA-').replace(' -I-RNA ', '-I-RNA ')
                    passages_list.append(sentence)

        num_file+=1
        if num_file==455:
            print(file) # 4864890.xml
            with codecs.open(train_path + "/" + 'train.txt', 'w', encoding='utf-8') as f:
                for sentence in passages_list:
                    f.write(sentence)
                    f.write('\n')
            passages_list = []
            print('passage 总数： {}'.format(num_passage)) # 10966
            num_passage = 0

        if num_file==570:
            with codecs.open(devel_path + "/" + 'devel.txt', 'w', encoding='utf-8') as f:
                for sentence in passages_list:
                    f.write(sentence)
                    f.write('\n')
            passages_list = []
            print('passage 总数： {}'.format(num_passage)) # 2731


# 利用GENIA tagger工具对标记过的语料进行预处理（分词+POS+CHUNK+NER）
# 得到 train.genia 文件


def judge(word, label_sen, entityTypes, flag, preType):
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
        for entityType in entityTypes:
            if word==('B-'+entityType+'/'):
                # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
                flag=2
                changable = entityType
                break
            elif word==('/I-'+entityType):
                # 处理 B-[]-I 实体在分词后标签B-和-I被分离的情况
                flag=0
                changable = entityType
                break
        if not changable:
            if word.startswith('B-'):
                # 获取实体B-XXX的类型
                entityType = None
                for item in entityTypes:
                    if word.startswith('B-'+item):
                        entityType = item
                        break
                if entityType:
                    if word.count('B-') > word.count('/I-'):
                        # 嵌套实体①
                        label_sen.append('B-'+star2Type.get(entityType))
                        flag=1
                        changable = entityType
                    elif word.count('B-') < word.count('/I-'):  # 实体结尾
                        # 嵌套实体②
                        label_sen.append('I-'+star2Type.get(preType))
                        flag=0
                        changable = preType
                    else: # 单个实体
                        if flag:
                            label_sen.append('I-'+star2Type.get(entityType))
                            flag=1
                        else:
                            label_sen.append('B-'+star2Type.get(entityType))
                            flag=0
                        changable = entityType
                else:
                    # 针对普通词，如：B-PRO
                    print(word)
                    flag=0
            # elif word.startswith('B-') and word.endswith('/I-'):
            #     if flag:
            #         if word.count('B-') >= word.count('/I-'):
            #             # 实体的内部
            #             label_sen.append('I-'+preType)
            #             flag = 1
            #         else:
            #             # 实体的结尾
            #             label_sen.append('I-'+preType)
            #             flag = 0
            #     else:
            #         if word.count('B-') > word.count('/I-'):
            #             # 实体的开头
            #             label_sen.append('B-'+entityType)
            #             flag = 1
            #         elif word.count('B-') < word.count('/I-'e):
            #             # 实体的结尾
            #             label_sen.append('I-'+entityType)
            #             flag = 0
            #         else:
            #             # 单个实体
            #             label_sen.append('B-'+entityType)
            #             flag = 0
            #     changable = entityType
            #     break
            elif '/I-' in word:
                # 对应两种结尾情况：①/I-XXX ②/I-XXX/I-XXX
                if word=='B-*/RARα/I-*/I-**':
                    print(preType)
                # 实体的结尾
                label_sen.append('I-'+star2Type.get(preType))
                flag=0
                changable = preType
            else:
                print('这就是个普通的词？')
            
    if changable:
        preType = changable
        for entityType in entityTypes:
            word = word.replace('B-'+entityType+'/', '').replace('/I-'+entityType, '')
    else:
        if flag:
            if flag==2: # 针对‘[entity]’这种实体形式
                # print(word, flag)
                label_sen.append('B-'+star2Type.get(preType))
                flag=1
            else:
                label_sen.append('I-'+star2Type.get(preType))
        else:
            label_sen.append('O')
            flag=0

    return word, flag, preType


# 根据预处理语料的标记 <B-xxx-></-I-xxx> 获取BIO标签
def getLabel(dataPath, train_or_devel):
    flag = 0    # 0:实体结束    1:实体内部  2:特殊实体[]
    preType = None
    label_sen = []
    sent = []
    if train_or_devel=='train':
        geniaPath = dataPath+ '/' + 'train.genia.txt'
        outputPath = dataPath+ '/' + 'train.out.txt'
    elif train_or_devel=='devel':
        geniaPath = dataPath+ '/' + 'devel.genia.txt'
        outputPath = dataPath+ '/' + 'devel.out.txt'

    with codecs.open(geniaPath, 'r', encoding='utf-8') as data:
        for line in data:
            if not line=='\n':
                word = line.split('\t')[0]
                word, flag, preType = judge(word, label_sen, entityTypes, flag, preType)
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

    entityTypes = ['**', '*']
    star2Type = {'*':'GENE', '**':'PROTEIN'}
    # entityTypes = ['ORGANISMS', 'GENE', 'CELLULAR', 'RNA', 'MOLECULES', 
    #                             'PROTEIN', 'TISSUES', 'CELLTYPE']

    train_path = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/train_455'
    devel_path = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/devel_115'
    BioC_PATH = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/caption_bioc'
    files = os.listdir(BioC_PATH)  # 得到文件夹下的所有文件名称
    files.sort()
    
    # readXML(files, BioC_PATH)

    '''
    % cd geniatagger-3.0.2
    % ./geniatagger  /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/train_455/train.txt \
    > /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/train_455/train.genia.txt
    % ./geniatagger  /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/devel_115/devel.txt \
    > /Users/ningshixian/Desktop/'BC6_Track1'/BioIDtraining_2/devel_115/devel.genia.txt
    '''

    getLabel(train_path, 'train')
    getLabel(devel_path, 'devel')

    print("完结撒花====")
