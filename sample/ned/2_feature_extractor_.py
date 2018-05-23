'''
抽取SVM的输入特征

实体周围的窗口为3的上下文的pos标记 + LocalCollocation + S_production 特征
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


def getCSVData(csv_path, entity2id):
    '''
    获取实体ID词典 {'entity':[id1, id2, ...]}
    实体区分大小写
    '''
    with open(csv_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            id = row['obj']
            entity = row['text']
            # text = row['text'].lower()
            if id.startswith('NCBI gene:') or id.startswith('Uniprot:') or \
                    id.startswith('gene:') or id.startswith('protein:'):
                if entity not in entity2id:
                    entity2id[entity] = []
                if id not in entity2id[entity]:
                    entity2id[entity].append(id)
        print('entity2id字典总长度：{}'.format(len(entity2id)))  # 5096

    num_word_multiID = 0
    entity2id_new = entity2id.copy()
    # 拓展实体词典
    for key, value in entity2id.items():
        if len(value)>1:
            num_word_multiID+=1
        for char in string.punctuation:
            if char in key:
                key = key.replace(char, ' ' + char + ' ')
        key = key.strip().replace('  ', ' ')
        if key not in entity2id_new:
            entity2id_new[key] = value
        key = key.strip().replace(' ', '')  # 去掉所有空格
        if key not in entity2id_new:
            entity2id_new[key] = value
    entity2id = {}
    del entity2id
    print('F4/80: {}'.format(entity2id_new['F4/80']))
    print('其中，多ID实体的个数：{}'.format(num_word_multiID))    # 1538
    return entity2id_new


def getGoldenID(path):
    '''
    获取所有实体的 golden ID 集合
    '''
    IDs_list = []
    with open(path) as f:
        for line in f:
            splited = line.strip('\n').split('\t')
            IDs_list.append(splited[::-1])  # 按offset顺序   ['']
    return IDs_list


# def getID(BioC_path):
#     '''
#     获取训练集中所有实体的正确ID集合
#     :param BioC_path:
#     :return:
#     '''
#     idx_line=0
#     IDs = {}
#     IDs_list = []
#     files = os.listdir(BioC_path)
#     files.sort()
#     print('获取goldenID集合')
#     for j in tqdm(range(len(files))):  # 遍历文件夹
#         file = files[j]
#         if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
#             f = BioC_path + "/" + file
#             DOMTree = parse(f)  # 使用minidom解析器打开 XML 文档
#             collection = DOMTree.documentElement  # 得到了根元素对象
#
#             source = collection.getElementsByTagName("source")[0].childNodes[0].data
#             date = collection.getElementsByTagName("date")[0].childNodes[0].data  # 时间
#             key = collection.getElementsByTagName("key")[0].childNodes[0].data
#             # 在集合中获取所有 document 的内容
#             documents = collection.getElementsByTagName("document")
#             for doc in documents:
#                 id = doc.getElementsByTagName("id")[0].childNodes[0].data
#                 sourcedata_document = doc.getElementsByTagName("infon")[0].childNodes[0].data
#                 doi = doc.getElementsByTagName("infon")[1].childNodes[0].data
#                 pmc_id = doc.getElementsByTagName("infon")[2].childNodes[0].data
#                 figure = doc.getElementsByTagName("infon")[3].childNodes[0].data
#                 sourcedata_figure_dir = doc.getElementsByTagName("infon")[4].childNodes[0].data
#
#                 passages = doc.getElementsByTagName("passage")
#                 for passage in passages:
#                     idx_line += 1
#                     text = passage.getElementsByTagName('text')[0].childNodes[0].data
#                     annotations = passage.getElementsByTagName('annotation')
#                     for annotation in annotations:
#                         info = annotation.getElementsByTagName("infon")[0]
#                         ID = info.childNodes[0].data
#                         location = annotation.getElementsByTagName("location")[0]
#                         offset = location.getAttribute("offset")
#                         length = location.getAttribute("length")
#                         annotation_txt = annotation.getElementsByTagName("text")[0]
#                         entity = annotation_txt.childNodes[0].data
#                         # if len(entity) == 1:
#                         #     continue
#                         if ID.startswith('Uniprot:') or ID.startswith('NCBI gene:') \
#                                 or ID.startswith('protein:') or ID.startswith('gene:'):
#                             IDs[offset] = ID
#                     IDs_sen = [IDs[key] for key in sorted(IDs.keys())]  # 按照 key 排序
#                     IDs_list.append(IDs_sen)
#                     IDs = {}
#
#     return IDs_list


def searchEntityId(entity, entity2id):
    '''
    通过词典匹配/知识库查询获取实体的IDs
    '''
    id_list = []

    temp = entity
    for char in string.punctuation:
        if char in temp:
            temp = temp.replace(char, ' ' + char + ' ')
    temp = temp.strip()

    # 词典精确匹配
    if entity in entity2id:
        Ids = entity2id[entity]
        id_list.extend(Ids)
        return id_list
    if entity.lower() in entity2id:
        Ids = entity2id[entity.lower()]
        id_list.extend(Ids)
        return id_list
    if temp in entity2id:
        Ids = entity2id[temp]
        id_list.extend(Ids)
        return id_list
    if entity.replace(' ', '') in entity2id:
        Ids = entity2id[entity.replace(' ', '')]
        id_list.extend(Ids)
        return id_list

    print('{} 不在 entity2id 词典中'.format(entity))

    # 数据库API查询1-reviewed
    res_reviewed = u.search(entity + '+reviewed:yes', frmt="tab", columns="id", limit=3)
    if res_reviewed == 400:
        print('请求无效\n')
    elif res_reviewed:  # 若是有返回结果
        Ids = extract_id_from_res(res_reviewed)
        for item in Ids:
            id_list.extend(['Uniprot:' + item])
        entity2id[entity] = id_list  # 将未登录实体添加到实体ID词典中
        return id_list

    # # 模糊匹配--计算 Jaro–Winkler 距离
    # max_score = -1
    # max_score_key = ''
    # for key in entity2id.keys():
    #     score = Levenshtein.jaro_winkler(key, entity)
    #     if score > max_score:
    #         max_score = score
    #         max_score_key = key
    # id_list.extend(entity2id.get(max_score_key))

    return id_list


def get_s_features(entity, goldenID, zhixin, entity2id, method='S-product'):
    '''
    获取一个实体的正负例的 S_product 特征和标签
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
    entity = entity.replace(' °C', '°C')
    entity = entity.strip()
    for char in string.punctuation:
        if char in entity:
            entity = entity.replace(' ' + char + ' ', char)
            entity = entity.replace(char + ' ', char)
            entity = entity.replace(' ' + char, char)

    x_one = []
    y_one = []
    golden_id = goldenID
    if '|' in golden_id:
        golden_id = golden_id.split('|')
        golden_id = '|'.join([item.split(':')[1] for item in golden_id])
    else:
        golden_id = golden_id.split(':')[1]
    if golden_id in Id2synvec:
        syn_vec = Id2synvec.get(golden_id)
    else:
        print('未找到 {} 对应同义词集向量，随机初始化1'.format(golden_id))
        syn_vec = np.round(np.random.uniform(-0.1, 0.1, 200), 6)
    can_not = syn_vec
    x_one.append(list(func(zhixin, syn_vec)))
    y_one.append(1)

    # 收集需要的实体ID
    if golden_id not in synId2entity:
        synId2entity[golden_id] = []
    if entity not in synId2entity[golden_id]:
        synId2entity[golden_id].append(entity)

    candidtaIds = searchEntityId(entity, entity2id)
    if goldenID in candidtaIds:
        candidtaIds.remove(goldenID)
    # print(goldenID)
    # print(candidtaIds)
    for candidta in candidtaIds:
        if candidta.startswith('gene:') or candidta.startswith('protein:'):
            continue
        if '|' in candidta:
            candidta = candidta.split('|')
            candidta = '|'.join([item.split(':')[1] for item in candidta])
        else:
            candidta = candidta.split(':')[1]
        if candidta in Id2synvec:
            syn_vec = Id2synvec.get(candidta)
        else:
            print('未找到{}对应同义词集向量，随机初始化2'.format(candidta))
            syn_vec = np.round(np.random.uniform(-0.1, 0.1, 200), 6)

        # 若不同ID，同义词集向量相同，跳过（more than 10,000）
        if (syn_vec==can_not).all():
            print('same')
            continue

        x_one.append(list(func(zhixin, syn_vec)))
        y_one.append(0)

        # 收集需要的实体ID
        if candidta not in synId2entity:
            synId2entity[candidta] = []
        if entity not in synId2entity[candidta]:
            synId2entity[candidta].append(entity)

    return x_one, y_one


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


def get_x_y(position, entity, entityId, id, zhixin, x_sen, pos_sen, x, y):
    '''
    获取实体周围的窗口为3的上下文及其对应的pos标记
    +  S_production 特征 = 组成SVM的输入特征
    '''
    pos, local_collocations_fea = pos_surround(x_sen, pos_sen, position, entityId, idx2pos, features_dict)
    s_feas, labels = get_s_features(entity, id, zhixin, entity2id)
    for i in range(len(s_feas)):
        fea = s_feas[i]
        label = labels[i]
        # x.append(pos + fea)
        x.append(pos + local_collocations_fea + fea)
        y.append(label)



def main(data1, data2, label_list, pos, id_list, train_num_remove):
    for senIdx in tqdm(range(len(data1))):
        entity_list = []
        x_sen = data1[senIdx]
        x_data = data2[senIdx][:sentence_maxlen]  # 原始数据截断
        y_sen = label_list[senIdx]
        pos_sen = pos[senIdx]
        IDs = id_list[senIdx]

        # 如果当前句子中没有实体，跳过
        if IDs==['']:
            continue

        assert len(x_sen) == len(x_data) == len(y_sen)

        # 计算质心向量
        zhixin = np.zeros(200)
        for i in range(len(x_sen)):
            wordId = x_sen[i]
            word = x_data[i]
            label = y_sen[i]
            if word not in stop_word and word not in string.punctuation:
                vector = embedding_matrix[wordId]
                if vector is None:
                    print('no')
                    vector = np.random.uniform(-0.1, 0.1, 200)
                zhixin += vector

        k = 0
        entity = ''
        entityId = ''
        for j in range(len(x_data)):
            wordId = x_sen[j]
            word = x_data[j]
            label = y_sen[j]
            if label == 1 or label == 0:
                if entity:
                    entity_list.append(j)
                    if IDs[k].startswith('gene:') or IDs[k].startswith('protein:'):
                        pass
                    elif IDs[k].startswith('NCBI gene:') or IDs[k].startswith('Uniprot:'):
                        position = j - len(entity.split())
                        get_x_y(position, entity, entityId.strip(), IDs[k], zhixin, x_sen, pos_sen, x, y)
                    else:
                        print('IDs error!')
                    entity = ''
                    entityId = ''
                    k += 1
                if label == 1:
                    entity = str(word) + ' '
                    entityId += " " + str(wordId)
                prex = label
            else:
                if prex == 1 or prex == 2:
                    entity += str(word) + ' '
                    entityId += " " + str(wordId)
                    prex = label
                else:
                    print('标签错误！跳过 {}-->{}'.format(prex, label))


        if not entity == '':
            entity_list.append(j)
            if IDs[k].startswith('gene:') or IDs[k].startswith('protein:'):
                pass
            elif IDs[k].startswith('NCBI gene:') or IDs[k].startswith('Uniprot:'):
                position = j - len(entity.split())
                get_x_y(position, entity, entityId.strip(), IDs[k], zhixin, x_sen, pos_sen, x, y)
            else:
                print('IDs error!')
            entity = ''
            entityId = ''
            k += 1
            prex = label

        if not len(IDs) == k:
            print('{}: {}\t{}'.format(senIdx, len(IDs), k))
            print('因为训练数据的句子被截断，后面的golden实体丢失，不影响')


if __name__ == '__main__':

    base = r'/home/administrator/桌面/BC6_Track1'
    # train_path = base + '/' + 'BioIDtraining_2/caption_bioc'
    # test_path = base + '/' + 'test_corpus_20170804/caption_bioc'
    train_path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/train_goldenID.txt'
    test_path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test_goldenID.txt'
    dic_path = base + '/' + 'BioIDtraining_2/annotations.csv'
    dic_path2 = base + '/' + 'test_corpus_20170804/annotations.csv'
    embeddingPath = r'/home/administrator/PycharmProjects/embedding'
    train_out_path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/train.out.txt'
    test_out_path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test.out.txt'
    synsetsVec_path = '/home/administrator/PycharmProjects/embedding/data/synsetsVec.txt'

    # 读取golden实体的ID词典
    entity2id = {}
    entity2id = getCSVData(dic_path, entity2id)
    entity2id = getCSVData(dic_path2, entity2id)
    # 读取停用词词典
    stop_word = get_stop_dic()
    # 读取同义词集向量
    Id2synvec = readSynVec(synsetsVec_path)
    if 'P04797' in Id2synvec:
        print('P04797')
    # 读取训练预料的数据和golden ID
    train_data = get_train_out_data(train_out_path)
    train_id_list = getGoldenID(train_path)
    # 读取测试预料的数据和golden ID
    test_data = get_train_out_data(test_out_path)
    test_id_list = getGoldenID(test_path)

    with open(embeddingPath + '/emb.pkl', "rb") as f:
        embedding_matrix = pkl.load(f)
    with open(embeddingPath + '/length.pkl', "rb") as f:
        word_maxlen, sentence_maxlen = pkl.load(f)

    with codecs.open('../data/train.pkl', "rb") as f:
        train_x, train_y, train_char, train_cap, train_pos, train_chunk = pkl.load(f)
    train_label_list = []
    for y in train_y:
        y = np.array(y).argmax(axis=-1)
        train_label_list.append(y)

    with codecs.open('../data/test.pkl', "rb") as f:
        test_x, test_y, test_char, test_cap, test_pos, test_chunk = pkl.load(f)
    test_label_list = []
    for y in test_y:
        y = np.array(y).argmax(axis=-1)
        test_label_list.append(y)

    assert len(train_id_list) == len(train_data) == len(train_x)
    assert len(test_id_list) == len(test_data) == len(test_x)

    idx2pos = {}
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/pos2idx.txt') as f:
        for line in f:
            pos, idx = line.split('\t')
            idx2pos[idx.strip('\n')] = pos

    with open('data/LocalCollocations.pkl', 'rb') as f:
        features_dict = pkl.load(f)

    prex = 0
    # train_num_remove = [725, 749, 1410, 1412, 3273, 3973, 4364, 7550, 8326, 8695, 10079, 10592, 10773, 11844]
    train_num_remove = []
    test_num_remove = []
    x = []
    y = []
    synId2entity = {}

    print('\n开始获取SVM特征...')

    # 获取训练集的SVM特征，并收集{实体ID:实体同义词集}
    main(train_x, train_data, train_label_list, train_pos, train_id_list, train_num_remove)
    # 获取测试集的SVM特征，并收集{实体ID:实体同义词集}
    main(test_x, test_data, test_label_list, test_pos, test_id_list, test_num_remove)

    print('\n获取SVM特征 finish')

    # 存在不同id(同义词集)有相同的向量？？导致特征相同但标签不同
    for i in range(len(y[:20])):
        label = y[:20][i]
        if label == 0:
            print('0')
            print(x[i])
        else:
            print('1')
            print(x[i])

    with open('data/synId2entity.txt', "w") as f:
        for key,value in synId2entity.items():
            f.write('{}\t{}'.format(key, '::,'.join(value)))
            f.write('\n')

    with open('data/train_svm.pkl', "wb") as f:
        pkl.dump((x, y), f, -1)
