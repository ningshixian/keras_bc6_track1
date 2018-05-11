'''
获取实体周围的窗口为3的上下文及其对应的pos标记
+  S_production 特征 = 组成SVM的输入特征进行训练

'''
import codecs
import os
import csv
import pickle as pkl
import string
from tqdm import tqdm
import numpy as np
from xml.dom.minidom import parse
from sample.utils.write_test_result import readSynVec, get_stop_dic
from sample.utils.write_test_result import extract_id_from_res
from sample.utils.helpers import cos_sim, pos_surround
import Levenshtein
from bioservices import UniProt
u = UniProt()


def getID(BioC_path):
    '''
    获取训练集中所有实体的正确ID集合
    :param BioC_path:
    :return:
    '''
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
                        if ID.startswith('Uniprot:') or ID.startswith('NCBI gene:') \
                                or ID.startswith('protein:') or ID.startswith('gene:'):
                            IDs[offset] = ID
                    IDs_sen = [IDs[key] for key in sorted(IDs.keys())]  # 按照 key 排序
                    IDs_list.append(IDs_sen)
                    IDs = {}

    return IDs_list


def searchEntityId(entity, entity2id):
    '''
    通过词典匹配/知识库查询获取实体的IDs
    '''
    id_list = []

    # 词典精确匹配
    if entity.lower() in entity2id:
        id_list.extend(entity2id[entity.lower()])
        return id_list
    # print('词典中没找到ID')
    # print(entity)

    # 数据库API查询1-reviewed
    res_reviewed = u.search(entity + '+reviewed:yes', frmt="tab", columns="id", limit=3)
    if res_reviewed == 400:
        print('请求无效\n')
    elif res_reviewed:  # 若是有返回结果
        Ids = extract_id_from_res(res_reviewed)
        for item in Ids:
            id_list.extend(['Uniprot:' + item])
        # Ids = ['Uniprot:' + Ids[0]]  # 取第一个结果作为ID
        entity2id[entity.lower()] = id_list  # 将未登录实体添加到实体ID词典中
        return id_list
    # print('数据库API没找到ID')

    # 模糊匹配--计算 Jaro–Winkler 距离
    max_score = -1
    max_score_key = ''
    for key in entity2id.keys():
        score = Levenshtein.jaro_winkler(key, entity.lower())
        if score > max_score:
            max_score = score
            max_score_key = key
    Ids = entity2id.get(max_score_key)
    return Ids


def get_s_features(entity, goldenID, zhixin, entity2id, method='S-product'):
    '''
    获取一个实体的正负例的 S_product 特征
    '''
    if method=='S-cosine':
        func = lambda x1,x2: cos_sim(x1, x2)
    elif method=='S-product':
        func = lambda x1,x2: np.multiply(x1, x2)    # x * y
    elif method=='S-raw':
        func = lambda x1, x2: list(dim for dim in x1) + list(dim for dim in x2)
    else:
        func = None
        print('没有这个函数')

    # 实体标准化
    entity = entity.strip()
    for char in string.punctuation + '-':
        if char in entity:
            entity = entity.replace(' ' + char + ' ', char)
            entity = entity.replace(char + ' ', char)
            entity = entity.replace(' ' + char, char)

    x = []
    y = []
    temp = goldenID
    goldenID = goldenID.split('|')[0] if '|' in goldenID else goldenID
    golden_id = goldenID.split(':')[1]
    if golden_id in geneId2vec:
        syn_vec = geneId2vec.get(golden_id)
    elif golden_id in proteinId2vec:
        syn_vec = proteinId2vec.get(golden_id)
    else:
        # print(golden_id)
        # print('未找到golden对应同义词集向量，随机初始化')
        syn_vec = np.round(np.random.uniform(-0.1, 0.1, 200), 6)
    x.append(list(func(zhixin, syn_vec)))
    y.append(1)

    if golden_id not in synId2entity:
        synId2entity[golden_id] = []
    if entity not in synId2entity[golden_id]:
        synId2entity[golden_id].append(entity)

    candidtaIds = searchEntityId(entity, entity2id)
    if temp in candidtaIds:
        candidtaIds.remove(temp)
    if golden_id in candidtaIds:
        candidtaIds.remove(golden_id)
    for candidta in candidtaIds:
        candidta = candidta.split('|')[0] if '|' in candidta else candidta
        candidta = candidta.split(':')[1]
        if candidta in geneId2vec:
            syn_vec = geneId2vec.get(candidta)
        elif candidta in proteinId2vec:
            syn_vec = proteinId2vec.get(candidta)
        else:
            # print(candidta)
            # print('未找到golden对应同义词集向量，随机初始化')
            syn_vec = np.round(np.random.uniform(-0.1, 0.1, 200), 6)
        x.append(list(func(zhixin, syn_vec)))
        y.append(0)

        if candidta not in synId2entity:
            synId2entity[candidta] = []
        if entity not in synId2entity[candidta]:
            synId2entity[candidta].append(entity)

    return x, y


def get_train_out_data(path):
    '''
    获取训练集的原始句子集合
    '''
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


def get_x_y(position, entity, entityId, id, zhixin, x, y):
    '''
    获取实体周围的窗口为3的上下文及其对应的pos标记
    +  S_production 特征 = 组成SVM的输入特征
    '''
    pos, local_collocations_fea = pos_surround(x_sen, pos_sen, position, entityId)
    s_feas, labels = get_s_features(entity, id, zhixin, entity2id)
    for i in range(len(s_feas)):
        # 获取SVM的特征输入
        fea = s_feas[i]
        label = labels[i]
        # x.append(pos + fea)
        x.append(pos + local_collocations_fea + fea)
        y.append(label)


def getCSVData(csv_path):
    '''
    获取实体ID词典 'entity':[id1, id2, ...]
    实体全部小写!!
    只用到gene和protein类别的部分
    '''
    num_word_multiID = 0
    entity2id_new = {}
    entity2id = {}
    with open(csv_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            if row['obj'].startswith('NCBI gene:') or \
                    row['obj'].startswith('Uniprot:'):

                text = row['text'].lower()
                if text not in entity2id:
                    entity2id[text] = []
                if row['obj'] not in entity2id[text]:
                    entity2id[text].append(row['obj'])
                # entity2id[row['text']] = list(set(entity2id[row['text']]))
        print('entity2id字典总长度：{}'.format(len(entity2id)))   # 4221
    return entity2id


base = r'/home/administrator/桌面/BC6_Track1'
BioC_path = base + '/' + 'BioIDtraining_2/caption_bioc'
dic_path = base + '/' + 'BioIDtraining_2/annotations.csv'   # 实体ID查找词典文件
embeddingPath = r'/home/administrator/PycharmProjects/embedding'
dataPath = '../data/train.out.txt'

entity2id = getCSVData(dic_path)    # 读取实体ID查找词典
geneId2vec, proteinId2vec = readSynVec()
stop_word = get_stop_dic()
train_data = get_train_out_data(dataPath)
IDs_list = getID(BioC_path)


with codecs.open('../data/train.pkl', "rb") as f:
    train_x, train_y, train_char, train_cap, train_pos, train_chunk = pkl.load(f)
label_list = []
for y in train_y:
    y = np.array(y).argmax(axis=-1)
    label_list.append(y)

assert len(IDs_list)==len(train_data)
assert len(train_data)==len(train_x)
assert len(train_data)==len(train_y)

with open(embeddingPath+'/emb.pkl', "rb") as f:
    embedding_matrix = pkl.load(f)
with open(embeddingPath+'/length.pkl', "rb") as f:
    word_maxlen, sentence_maxlen = pkl.load(f)

prex = 0
sen_remove_list = [725, 749, 1410, 1412, 3273, 3973, 4364, 7550, 8326, 8695, 10079, 10592, 10773, 11844]
x = []
y = []
synId2entity = {}

print('\n开始获取SVM特征...')
for senIdx in tqdm(range(len(train_x))):

    if str(senIdx) in sen_remove_list:
        continue

    entity_list = []
    x_sen = train_x[senIdx]
    x_data = train_data[senIdx][:sentence_maxlen]    # 原始数据截断
    y_sen = label_list[senIdx]
    pos_sen = train_pos[senIdx]
    IDs = IDs_list[senIdx]

    assert len(x_sen) == len(x_data) == len(y_sen)

    # 计算质心向量
    zhixin = np.zeros(200)
    for i in range(len(x_sen)):
        wordId = x_sen[i]
        word = x_data[i]
        label = y_sen[i]
        if word not in stop_word and word not in string.punctuation:
            vector = embedding_matrix[wordId]
            # if vector is None:
            #     print('no')
            #     vector = np.random.uniform(-0.1, 0.1, 200)
            zhixin += vector

    k = 0
    entity = ''
    entityId = ''
    for j in range(len(x_data)):
        wordId = x_sen[j]
        word = x_data[j]
        label = y_sen[j]
        if label == 1 or label==0:
            if entity:
                entity_list.append(j)
                if IDs[k].startswith('gene:') or IDs[k].startswith('protein:'):
                    k += 1
                    entity = ''
                    entityId = ''
                    prex = label
                else:
                    position = j - len(entity.split())
                    get_x_y(position, entity, entityId.strip(), IDs[k], zhixin, x, y)
                    k += 1
                    entity=''
                    entityId = ''
            if label==1:
                entity = str(word) + ' '
                entityId += " " + str(wordId)
        else:
            if prex == 1 or prex == 2:
                entity += str(word) + ' '
                entityId += " " + str(wordId)
            else:
                print('标签错误！跳过 {}-->{}'.format(prex, label))
        prex = label

    if not entity == '':
        entity_list.append(j)
        if IDs[k].startswith('gene:') or IDs[k].startswith('protein:'):
            k += 1
            entity = ''
            entityId = ''
        else:
            position = j - len(entity.split())
            get_x_y(position, entity, entityId.strip(), IDs[k], zhixin, x, y)
            k += 1
            entity = ''
            entityId = ''

#     if not len(IDs) == k:
#         sen_remove_list.append(senIdx)
#         print(y_sen)
#         print(entity_list)
#
# print(len(sen_remove_list))
# print(sen_remove_list)


with open('train.pkl', "wb") as f:
    pkl.dump((x, y), f, -1)
with open('synId2entity.txt', "w") as f:
    for key,value in synId2entity.items():
        f.write('{}\t{}'.format(key, '::,'.join(value)))
        f.write('\n')
