'''
抽取实体特征训练SVM

'''
import codecs
import os
import pickle as pkl
import string
from xml.dom.minidom import parse

import numpy as np
from bioservices import UniProt
from sample.utils.write_test_result import readSynVec, getCSVData
from tqdm import tqdm

from sample.utils.helpers import cos_sim

u = UniProt()


def idFilter(res):
    '''
        将字典匹配或API匹配得到的Ids集合中，与type不符的部分去掉
        取第一个结果作为实体ID
    '''
    temp = []
    if res == 400:
        print('请求无效\n')
        return temp
    results = res.split('\n')[1:-1]  # 去除开头一行和最后的''
    for line in results:
        Id = line.split('\t')[-1]
        temp.append(Id)
        break
    return temp


def getID(BioC_path):
    idx_line=0
    IDs = {}
    IDs_list = []
    files = os.listdir(BioC_path)
    files.sort()
    print('获取goldenID集合')
    for j in tqdm(range(len(files))):  # 遍历文件夹
        file = files[j]
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = BioC_path + "/" + file
            DOMTree = parse(f)  # 使用minidom解析器打开 XML 文档
            collection = DOMTree.documentElement  # 得到了根元素对象

            source = collection.getElementsByTagName("source")[0].childNodes[0].data
            date = collection.getElementsByTagName("date")[0].childNodes[0].data  # 时间
            key = collection.getElementsByTagName("key")[0].childNodes[0].data
            # 在集合中获取所有 document 的内容
            documents = collection.getElementsByTagName("document")
            for doc in documents:
                id = doc.getElementsByTagName("id")[0].childNodes[0].data
                sourcedata_document = doc.getElementsByTagName("infon")[0].childNodes[0].data
                doi = doc.getElementsByTagName("infon")[1].childNodes[0].data
                pmc_id = doc.getElementsByTagName("infon")[2].childNodes[0].data
                figure = doc.getElementsByTagName("infon")[3].childNodes[0].data
                sourcedata_figure_dir = doc.getElementsByTagName("infon")[4].childNodes[0].data

                passages = doc.getElementsByTagName("passage")
                for passage in passages:
                    idx_line += 1
                    text = passage.getElementsByTagName('text')[0].childNodes[0].data
                    annotations = passage.getElementsByTagName('annotation')
                    for annotation in annotations:
                        info = annotation.getElementsByTagName("infon")[0]
                        ID = info.childNodes[0].data
                        location = annotation.getElementsByTagName("location")[0]
                        offset = location.getAttribute("offset")
                        length = location.getAttribute("length")
                        annotation_txt = annotation.getElementsByTagName("text")[0]
                        entity = annotation_txt.childNodes[0].data
                        # if len(entity) == 1:
                        #     continue
                        if ID.startswith('Uniprot:') or ID.startswith('protein:') \
                                or ID.startswith('NCBI gene:') or ID.startswith('gene:') \
                                or ID.startswith('Rfam:') or ID.startswith('mRNA:'):
                            IDs[offset] = ID
                    IDs_sen = [IDs[key] for key in sorted(IDs.keys())]  # 按照 key 排序
                    IDs_list.append(IDs_sen)
                    IDs = {}
    return IDs_list


def searchEntityId(entity, entity2id):
    '''
    对识别的实体进行标准化，并链接ID
    '''
    Ids = []
    id_list = {}

    # 实体标准化
    for char in string.punctuation:
        if char in entity:
            entity = entity.replace(' '+char+' ', char)
            entity = entity.replace(char+' ', char)
            entity = entity.replace(' '+char, char)

    # 去掉所有标点符号
    temp = entity
    for char in string.punctuation:
        if char in temp:
            temp = temp.replace(char, '')

    # 词典精确匹配1
    if entity.lower() in entity2id:
        Ids = entity2id[entity.lower()]
        # Ids = idFilter(type, Ids)     # 不进行筛选，否则ID几乎全被干掉了
        id_list[entity] = Ids
        return id_list

    # 词典精确匹配2
    if temp.lower() in entity2id:
        Ids = entity2id[temp.lower()]
        # Ids = idFilter(type, Ids)
        id_list[entity] = Ids
        return id_list

    # 数据库API查询
    res = u.search(entity + '+reviewed:yes', frmt="tab", columns="genes, id", limit=3)
    if res:  # 若是有返回结果
        Ids = idFilter(res)  # 不进行筛选
        if not Ids:
            print('实体ID结合为空: ', entity)
        id_list[entity] = Ids
        entity2id[entity.lower()] = Ids  # 将未登录实体添加到实体ID词典中
        return id_list

    # 数据库API查询2
    res = u.search(temp + '+reviewed:yes', frmt="tab", columns="genes, id", limit=3)
    if res:
        Ids = idFilter(res)  # 不进行筛选
        if not Ids:
            print('实体ID结合为空: ', temp)
        id_list[entity] = Ids
        entity2id[entity.lower()] = Ids  # 将未登录实体添加到实体ID词典中
        return id_list

    return []


def getFeatures(entity, goldenID, zhixin, entity2id, method='S-product'):
    '''
    获取正负例的 S_product 特征
    '''
    if method=='S-cosine':
        func = lambda x1,x2: cos_sim(x1, x2)
    elif method=='S-product':
        func = lambda x1,x2: np.multiply(x1, x2)    # x * y
    elif method=='S-raw':
        func = lambda x1, x2: list(dim for dim in x1) + list(dim for dim in x2)
    else:
        func = None

    x = []
    y = []
    temp = goldenID
    goldenID = goldenID.split('|')[0] if '|' in goldenID else goldenID
    goldenID1 = goldenID.split(':')[1]
    if goldenID1 in geneId2vec:
        syn_vec = geneId2vec.get(goldenID1)
    elif goldenID1 in proteinId2vec:
        syn_vec = proteinId2vec.get(goldenID1)
    else:
        # print('未找到golden对应同义词集向量，随机初始化')
        syn_vec = np.random.uniform(-0.1, 0.1, 200)
    x.append(list(func(zhixin, syn_vec)))
    y.append(1)

    candidtaIds = searchEntityId(entity, entity2id)
    candidtaIds.remove(temp) if temp in candidtaIds else candidtaIds
    for candidta in candidtaIds:
        candidta = candidta.split('|')[0] if '|' in candidta else candidta
        if ':' in candidta:
            print(':')
            candidta = candidta.split(':')[1]
        if candidta in geneId2vec:
            syn_vec = geneId2vec.get(candidta)
        elif candidta in proteinId2vec:
            syn_vec = proteinId2vec.get(candidta)
        else:
            # print('未找到对应同义词集向量，随机初始化')
            syn_vec = np.random.uniform(-0.1, 0.1, 200)
        x.append(list(func(zhixin, syn_vec)))
        y.append(0)
    return x, y


def get_train_out_data(path):
    # 获取训练集每个句子
    sen_line = []
    sen_list = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                sen_list.append(sen_line)
                sen_line = []
            else:
                token = line.replace('\n', '').split('\t')
                sen_line.append(token[0])
    return sen_list


def get_x_y(tokenIdx, entity, idx, x, y):
    '''
    获取实体周围的窗口为3的上下文及其对应的pos标记
    +  S_production 特征 = 组成SVM的输入特征
    '''
    index1 = tokenIdx - 3 - len(entity.split())
    index2 = tokenIdx - len(entity.split())

    left_x = x_sen[index1:index2] if index1 >= 0 else [0] * abs(index1) + x_sen[0:index2]
    right_x = x_sen[tokenIdx:tokenIdx + 3] if tokenIdx + 3 <= len(x_data) \
        else x_sen[tokenIdx:] + [0] * abs(tokenIdx + 3 - len(x_data))
    surroundding_word.append(left_x + right_x)

    left_pos = [0] * abs(index1) + pos_sen[0:index2] if index1 < 0 else pos_sen[index1:index2]
    right_pos = pos_sen[tokenIdx:tokenIdx + 3] + [0] * abs(tokenIdx + 3 - len(x_sen)) if tokenIdx + 3 > len(
        x_sen) else pos_sen[tokenIdx:tokenIdx + 3]
    pos_list.append(left_pos + right_pos)

    entity_list.append(entity)

    assert len(pos_list[-1])==6
    assert len(surroundding_word[-1])==6

    # 获取正负例的 S_product 特征
    # print('{},{}'.format(entity, IDs[idx]))
    feas, labels = getFeatures(entity, IDs[idx], zhixin, entity2id)
    for i in range(len(feas)):
        fea = feas[i]
        label = labels[i]
        # 获取SVM的特征输入
        x.append(pos_list[-1] + surroundding_word[-1] + fea)
        y.append(label)




base = r'/home/administrator/桌面/BC6_Track1'
BioC_path = base + '/' + 'BioIDtraining_2/caption_bioc'
dic_path = base + '/' + 'BioIDtraining_2/annotations.csv'   # 实体ID查找词典文件
embeddingPath = r'/home/administrator/PycharmProjects/embedding'
dataPath = '../data/train.out.txt'

entity2id = getCSVData(dic_path)    # 读取实体ID查找词典
geneId2vec, proteinId2vec, stop_word = readSynVec()
source_data = get_train_out_data(dataPath)
IDs_list = getID(BioC_path)
print(IDs_list[25])
print(source_data[25])
assert len(IDs_list)==len(source_data)


with codecs.open('../data/train.pkl', "rb") as f:
    train_x, train_y, train_char, train_cap, train_pos, train_chunk = pkl.load(f)
label_list = []
for y in train_y:
    y = np.array(y).argmax(axis=-1)
    label_list.append(y)

assert len(source_data)==len(train_x)
assert len(source_data)==len(train_y)

with open(embeddingPath+'/emb.pkl', "rb") as f:
    embedding_matrix = pkl.load(f)
with open(embeddingPath+'/length.pkl', "rb") as f:
    word_maxlen, sentence_maxlen = pkl.load(f)

prex = 0
x = []
y = []
print('\n开始获取特征')
for senIdx in tqdm(range(len(train_x))):
    zhixin_list = []
    entity = ''
    entity_list = []
    surroundding_word = []
    pos_list = []

    x_sen = train_x[senIdx]
    x_data = source_data[senIdx][:sentence_maxlen]    # 截断
    y_sen = label_list[senIdx]
    pos_sen = train_pos[senIdx]
    IDs = IDs_list[senIdx]

    zhixin = np.zeros(200)  # 计算质心向量
    for wordId in x_sen:
        # if word not in stop_word:
        vector = embedding_matrix[wordId]
        if vector is None:
            vector = np.random.uniform(-0.1, 0.1, 200)
        zhixin += vector
    zhixin_list.append(zhixin)

    idx = 0
    if not len(x_data)==len(y_sen):
        print(x_data, y_sen)
        print(len(x_data), len(y_sen))
    assert len(x_data) == len(y_sen)
    for tokenIdx in range(len(x_data)):
        label = y_sen[tokenIdx]
        # wordId = x_sen[tokenIdx]
        word = x_data[tokenIdx]
        if label == 1:
            if entity:
                get_x_y(tokenIdx, entity, idx, x, y)
                idx += 1
                entity=''
            prex = label
            entity = str(word) + ' '
        elif label == 2:
            if prex == 1 or prex == 2:
                entity += str(word) + ' '
            else:
                print('标签错误！跳过')
            prex = label
        else:
            if entity:
                get_x_y(tokenIdx, entity, idx, x, y)
                idx += 1
                entity = ''
            prex = 0
    if not entity == '':
        get_x_y(tokenIdx, entity, idx, x, y)
        idx += 1
        entity = ''


with open('train.pkl', "wb") as f:
    pkl.dump((x, y), f, -1)