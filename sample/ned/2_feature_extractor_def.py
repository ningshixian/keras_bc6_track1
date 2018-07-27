'''
实体的上下文及其pos标记
'''
import re
import codecs
import os
import csv
import pickle as pkl
from urllib.error import URLError
import string
from tqdm import tqdm
import numpy as np
from xml.dom.minidom import parse
from sample.utils.write_test_result import readSynVec, get_stop_dic
from sample.utils.write_test_result import extract_id_from_res
from sample.utils.helpers import cos_sim, pos_surround, entityNormalize
import Levenshtein
from collections import OrderedDict
from bioservices import UniProt
from Bio import Entrez
u = UniProt()


exits = 0
not_exits = 0


def strippingAlgorithm(entity):
    '''
    lower-cased
    punctuation-removed
    remove common words like 'protein', 'proteins', 'gene', 'genes', 'RNA', 'organism' (stripping algorithm)
    id list 按频度排序×
    '''
    entity_variants1 = entity.lower().strip('\n').strip()
    entity_variants2 = entity.lower().strip('\n').strip()
    for punc in string.punctuation:
        if punc in entity_variants1:
            entity_variants1 = entity_variants1.replace(punc, ' ')
            entity_variants2 = entity_variants2.replace(punc, ' ' + punc + ' ')

    entity_variants1 = entity_variants1.replace('  ', ' ').strip()
    entity_variants2 = entity_variants2.replace('  ', ' ').strip()

    common = ['protein', 'proteins', 'gene', 'genes', 'rna', 'organism']
    for com in common:
        # if entity_variants1.startswith(com) or entity_variants1.endswith(com):
        entity_variants1 = entity_variants1.replace(com, '').strip()
        entity_variants2 = entity_variants2.replace(com, '').strip()

    entity_variants3 = re.findall(r'[0-9]+|[a-z]+', entity_variants2)
    entity_variants3 = ' '.join(entity_variants3)
    entity_variants3 = entity_variants3.replace('  ', ' ').strip()

    # start with the longest one
    return entity_variants2, entity_variants1, entity_variants3


def getCSVData(csv_path, entity2id):
    '''
    获取实体ID词典 {'entity':[id1, id2, ...]}
    实体区分大小写
    '''
    with open(csv_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            id = row['obj']
            if id.startswith('NCBI gene:') or id.startswith('Uniprot:') or \
                    id.startswith('gene:') or id.startswith('protein:'):
                #  对实体进行过滤
                entity = strippingAlgorithm(row['text'])[0]
                if entity not in entity2id:
                    entity2id[entity] = OrderedDict()
                if id not in entity2id[entity]:
                    entity2id[entity][id] = 1
                else:
                    entity2id[entity][id] += 1
        print('entity2id字典总长度：{}'.format(len(entity2id)))  # 5096

    # 按频度重新排序
    entity2id_1 = {}
    for key, value in entity2id.items():
        value_sorted = sorted(value.items(), key=lambda item: item[1], reverse=True)
        entity2id_1[key] = [item[0] for item in value_sorted]

    # 将protein排在前面
    entity2id_2 = {}
    for key, value in entity2id_1.items():
        protein_list = []
        gene_list = []
        for id in value:
            if id.startswith('Uniprot:') or id.startswith('protein:'):
                protein_list.append(id)
            elif id.startswith('NCBI gene:') or id.startswith('gene:'):
                gene_list.append(id)
            entity2id_2[key] = protein_list + gene_list

    for k,v in entity2id_2.items():
        print(k,v)

    print('F4/80: {}'.format(entity2id_2.get('f4 / 80')))  # ['Uniprot:Q61549']    ['NCBI gene:13733', 'Uniprot:Q61549']
    print('F480: {}'.format(entity2id_2.get('f480')))  # ['Uniprot:Q61549']    ['NCBI gene:13733', 'Uniprot:Q61549']
    return entity2id_2


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


def search_id_from_Uniprot(query_list, reviewed=True):
    # Uniprot 数据库API查询-reviewed
    for query in query_list:
        if reviewed:
            res_reviewed = u.search(query + '+reviewed:yes', frmt="tab", columns="id", limit=5)
        else:
            res_reviewed = u.search(query, frmt="tab", columns="id", limit=5)
        if isinstance(res_reviewed, int):
            print('请求无效\n')
        elif res_reviewed:  # 若是有返回结果
            Ids = extract_id_from_res(res_reviewed)
            return ['Uniprot:' + Ids[i] for i in range(len(Ids))]
    return []


def search_id_from_NCBI(query_list):
    # NCBI-gene数据库API查询
    for query in query_list:
        try:
            handle = Entrez.esearch(db="gene", idtype="acc", sort='relevance', term=query)
            record = Entrez.read(handle)
        except RuntimeError as e:
            print(e)
            continue
        except URLError as e:
            print(e)
            continue
        if record["IdList"]:
            return ['NCBI gene:' + record["IdList"][i] for i in
                               range(len(record["IdList"][:5]))]
    return []


def searchEntityId(entity, entity2id):
    '''
    通过词典匹配/知识库查询获取实体的IDs
    '''
    id_list = []
    entity_variants1, entity_variants2, entity_variants3 = strippingAlgorithm(entity)
    query_list = [entity, entity_variants1]
    # query_list = [entity, entity_variants1, entity_variants2, entity_variants3]

    # 词典精确匹配
    if entity_variants1 in entity2id:
        Ids = entity2id[entity_variants1]
        id_list.extend(Ids)
        return id_list
        # # 负例不够的随机填补
        # while len(id_list)<num_candidates:
        #     k = np.random.randint(0, len(test_id_list))
        #     if isinstance(test_id_list[k], list) and test_id_list[k][0]:
        #         id_list.append(test_id_list[k][0])
        # else:
        #     return id_list


    # 知识库精确匹配（先忽略类型 leixing）
    if entity_variants1 in gene2id:
        print('NCBI gene 知识库精确匹配')
        Ids = gene2id[entity.lower()]
        id_list.extend(['NCBI gene:' + Ids[i] for i in range(len(Ids[:num_candidates]))])
    elif entity_variants1 in protein2id:
        print('Uniprot 知识库精确匹配')
        Ids = protein2id[entity.lower()]
        id_list.extend(['Uniprot:' + Ids[i] for i in range(len(Ids[:num_candidates]))])
    else:
        # Uniprot 数据库API查询-reviewed
        Ids = search_id_from_Uniprot(query_list, reviewed=True)
        if Ids:
            id_list.extend(Ids)
            entity2id[entity_variants1] = Ids if entity_variants1 not in entity2id else list(set(entity2id[entity_variants1] + Ids))

        # Uniprot 数据库API查询-unreviewed
        Ids = search_id_from_Uniprot(query_list, reviewed=False)
        if Ids:
            id_list.extend(Ids)
            entity2id[entity_variants1] = Ids if entity_variants1 not in entity2id else list(set(entity2id[entity_variants1] + Ids))

        # NCBI-gene数据库API查询
        Ids = search_id_from_NCBI(query_list)
        id_list.extend(Ids)
        entity2id[entity_variants1] = Ids if entity_variants1 not in entity2id else list(
            set(entity2id[entity_variants1] + Ids))

    if not id_list:
        id_list = []
        print('未找到{}的ID，空'.format(entity))  # 152次出现

    return list(set(id_list))


# def getCandidateDef(def_file):
#     id2def = {}
#     with open(def_file) as f:
#         for line in f:
#             if not line=='\\':
#                 if not line=='\n' and not ' ' in line and not '\t' in line:
#                     # ID
#                     id2def[line]=[]
#                 else


def get_s_features(entity, goldenID, zhixin, entity2id, exits, not_exits):
    '''
    获取一个实体的正负例的 S_product 特征和标签
    '''
    method = 'S-product'
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
    x_id_one = []
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
        # print('未找到 {} 对应同义词集向量，随机初始化'.format(golden_id))
        syn_vec = np.random.uniform(-0.1, 0.1, 200)
    can_not = syn_vec

    x_one.append(list(func(zhixin, syn_vec)))
    x_id_one.append(golden_id)
    y_one.append(1)

    # 收集需要的实体ID
    if golden_id not in synId2entity:
        synId2entity[golden_id] = []
    if entity not in synId2entity[golden_id]:
        synId2entity[golden_id].append(entity)

    candidtaIds = searchEntityId(entity, entity2id)
    # while len(candidtaIds) < num_candidates:
    #     k = np.random.randint(0, len(train_id_list))
    #     if isinstance(train_id_list[k], list) and train_id_list[k][0]:
    #         candidtaIds.append(train_id_list[k][0])
    if goldenID in candidtaIds:
        candidtaIds.remove(goldenID)
        exits+=1
    else:
        not_exits+=1

    # print('len(candidtaIds):{}//{}'.format(len(candidtaIds), entity))

    for candidta in candidtaIds:
        if candidta.startswith('gene:') or candidta.startswith('protein:'):
            continue
        if '|' in candidta:
            candidta = candidta.split('|')
            candidta = '|'.join([item.split(':')[1] for item in candidta])
        else:
            # print(candidta)
            candidta = candidta.split(':')[1]
        if candidta in Id2synvec:
            syn_vec = Id2synvec.get(candidta)
        else:
            print('未找到候选ID-{}-的同义词集向量，随机初始化'.format(candidta))
            syn_vec = np.random.uniform(-0.1, 0.1, 200)

        # # 若不同ID，同义词集向量相同，跳过（more than 10,000）
        # if (syn_vec==can_not).all():
        #     print(goldenID, candidtaIds)
        #     print('same')
        #     continue

        x_one.append(list(func(zhixin, syn_vec)))
        x_id_one.append(candidta)
        y_one.append(0)

        # 收集需要的实体ID
        if candidta not in synId2entity:
            synId2entity[candidta] = []
        if entity not in synId2entity[candidta]:
            synId2entity[candidta].append(entity)

    return x_one, x_id_one, y_one


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


def get_x_y(entity, id, zhixin, x_sen, pos_sen, position, res, exits, not_exits):
    '''
    获取实体的上下文及其对应的pos标记
    获取标签
    '''
    x_left, x_pos_left, x_right, x_pos_right, x_id, y = res
    s_feas, x_id_one, labels = get_s_features(entity, id, zhixin, entity2id, exits, not_exits)

    num = 0
    end_l = position
    x_sen_left = []
    pos_sen_left = []
    while num<context_window_size:
        end_l -= 1
        if end_l>0:
            # 过滤停用词 stop_word
            if x_sen[end_l] not in stop_word:
                x_sen_left.append(x_sen[end_l])
                pos_sen_left.append(pos_sen[end_l])
                num+=1
        else:
            x_sen_left.append(0)
            pos_sen_left.append(0)
            num += 1
    x_sen_left = x_sen_left[::-1]
    pos_sen_left = pos_sen_left[::-1]

    num = 0
    start_r = position + len(entity.split())
    x_sen_right = []
    pos_sen_right = []
    while num < context_window_size:
        if start_r < len(x_sen):
            # 过滤停用词 stop_word
            if x_sen[start_r] not in stop_word:
                x_sen_right.append(x_sen[start_r])
                pos_sen_right.append(pos_sen[start_r])
                num += 1
        else:
            x_sen_right.append(0)
            pos_sen_right.append(0)
            num += 1
        start_r += 1

    assert len(x_sen_left)==len(x_sen_right)

    # start_l = position - context_window_size
    # end_l = position
    # start_r = position + len(entity.split())
    # end_r = start_r + context_window_size
    # x_sen_left = x_sen[start_l:end_l] if start_l > 0 else [0] * abs(start_l) + x_sen[:end_l]
    # x_sen_right = x_sen[start_r:end_r] if end_r < len(x_sen) else x_sen[start_r:] + [0] * (end_r - len(x_sen))
    # pos_sen_left = pos_sen[start_l:end_l] if start_l > 0 else [0] * abs(start_l) + pos_sen[:end_l]
    # pos_sen_right = pos_sen[start_r:end_r] if end_r < len(pos_sen) else pos_sen[start_r:] + [0] * (
    # end_r - len(pos_sen))

    for i in range(len(x_id_one)):
        fea = s_feas[i]
        id_one = x_id_one[i]
        label = labels[i]
        # print(str(label) + '\t' + str(id_one))
        x_left.append(x_sen_left)
        x_right.append(x_sen_right)
        x_pos_left.append(pos_sen_left)
        x_pos_right.append(pos_sen_right)
        x_id.append(id_one)
        y.append([label])


def main(data1, data2, label_list, pos, id_list):
    for senIdx in tqdm(range(len(data1))):
        entity_list = []
        x_sen = data1[senIdx]
        x_data = data2[senIdx][:sentence_maxlen]  # 原始数据截断
        y_sen = label_list[senIdx]
        pos_sen = pos[senIdx]
        IDs = id_list[senIdx]

        # 如果当前句子中没有实体，跳过
        if IDs==['']:
            # print(senIdx)
            continue

        assert len(x_sen) == len(x_data) == len(y_sen)

        # 计算质心向量
        zhixin = np.zeros(200)
        for i in range(len(x_sen)):
            wordId = x_sen[i]
            word = x_data[i]
            label = y_sen[i]
            if word not in stop_word_list and word not in string.punctuation:
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
            if label == 0 or label == 1 or label == 3:
                if entity:
                    entity_list.append(j)
                    if IDs[k].startswith('gene:') or IDs[k].startswith('protein:'):
                        pass
                    elif IDs[k].startswith('NCBI gene:') or IDs[k].startswith('Uniprot:'):
                        position = j - len(entity.split())
                        # print(position)
                        # entity = entityNormalize(entity, s, tokenIdx - len(entity.split()))
                        get_x_y(entity, IDs[k], zhixin, x_sen, pos_sen, position, res, exits, not_exits)
                    else:
                        print('IDs error!')
                    entity = ''
                    entityId = ''
                    k += 1
                if label == 1 or label == 3:
                    entity = str(word) + ' '
                    entityId += " " + str(wordId)
                prex = label
            elif label == 2:
                if prex == 1 or prex == 2:
                    entity += str(word) + ' '
                    entityId += " " + str(wordId)
                    prex = label
                else:
                    print('标签错误！跳过1: {}-->{}'.format(prex, label))
            else:
                if prex == 3 or prex == 4:
                    entity += str(word) + ' '
                    entityId += " " + str(wordId)
                    prex = label
                else:
                    print('标签错误！跳过2: {}-->{}'.format(prex, label))


        if not entity == '':
            entity_list.append(j)
            if IDs[k].startswith('gene:') or IDs[k].startswith('protein:'):
                pass
            elif IDs[k].startswith('NCBI gene:') or IDs[k].startswith('Uniprot:'):
                position = j - len(entity.split())
                get_x_y(entity, IDs[k], zhixin, x_sen, pos_sen, position, res, exits, not_exits)
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

    results = []
    # gap_pos = 0
    # gap_surround = 36
    # gap_s = 188950

    # 20 words in each side
    context_window_size = 10
    # 负采样个数
    num_candidates = 5
    # 停用词表/标点符号
    stop_word = [239, 153, 137, 300, 64, 947, 2309, 570, 10, 69, 238, 175, 852, 7017, 378, 136, 5022, 1116, 5194, 14048,
                 28, 217, 4759, 7359, 201, 671, 11, 603, 15, 1735, 2140, 390, 2366, 12, 649, 4, 1279, 3351, 3939, 5209,
                 16, 43, 2208, 8, 5702, 4976, 325, 891, 541, 1649, 17, 416, 2707, 108, 381, 678, 249, 5205, 914, 5180, 5, 20,
                 18695, 15593, 5597, 730, 1374, 18, 2901, 1440, 237, 150, 44, 10748, 549, 3707, 4325, 27, 331, 522, 10790, 297,
                 1060, 1976, 7803, 1150, 1189, 2566, 192, 5577, 703, 666, 315, 488, 89, 1103, 231, 16346, 9655, 6569, 605, 6, 294,
                 3932, 24965, 9, 775, 4593, 76, 21733, 140, 229, 16368, 21098, 181, 620, 134, 6032, 268, 2267, 22948, 88, 655, 24768,
                 6870, 25, 615, 4421, 99, 3, 375, 483, 7, 2661, 32, 2223, 42, 1612, 595, 22, 37, 432, 8439, 67, 15853, 6912,
                 459, 21441, 3811, 1538, 1644, 2834, 1192, 5197, 1734, 78, 647, 247, 491, 16228, 23, 578, 34, 47, 77, 1239,
                 846, 26, 24317, 785, 3601, 8504, 29, 9414, 520, 3399, 2035, 6778, 96, 2048, 1, 579, 1135, 173, 4089, 4980, 205,
                 63, 516, 169, 8413, 1980, 337, 19, 521, 13, 48, 551, 3927, 59, 10281, 11926, 3915]
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

    # 读取golden实体的ID词典(不包括测试集的实体ID)
    entity2id = {}
    entity2id = getCSVData(dic_path, entity2id)
    # entity2id = getCSVData(dic_path2, entity2id)

    # 读取停用词词典
    stop_word_list = get_stop_dic()

    # 读取训练预料的数据和golden ID
    train_data = get_train_out_data(train_out_path)
    train_id_list = getGoldenID(train_path)
    # 读取测试预料的数据和golden ID(作弊？)
    test_data = get_train_out_data(test_out_path)
    test_id_list = getGoldenID(test_path)

    # 知识库精确匹配(效果不好)
    protein2id, gene2id = {}, {}
    # with open('../pg2id.pkl', 'rb') as f:
    #     protein2id, gene2id = pkl.load(f)

    # 用于求质心
    with open(embeddingPath + '/emb.pkl', "rb") as f:
        embedding_matrix = pkl.load(f)
    with open(embeddingPath + '/length.pkl', "rb") as f:
        word_maxlen, sentence_maxlen = pkl.load(f)

    # 获取训练集和测试集的golden实体
    with codecs.open('../data/train.pkl', "rb") as f:
        train_x, train_y, train_char, train_cap, train_pos, train_chunk, train_dict = pkl.load(f)
    train_label_list = []
    for y in train_y:
        y = np.array(y).argmax(axis=-1)
        train_label_list.append(y)
    with codecs.open('../data/test.pkl', "rb") as f:
        test_x, test_y, test_char, test_cap, test_pos, test_chunk, test_dict = pkl.load(f)
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

    # # 读取同义词集向量
    # synsetsVec_path = '/home/administrator/PycharmProjects/embedding/data/synsetsVec.txt'
    # Id2synvec = readSynVec(synsetsVec_path)
    Id2synvec = {}

    prex = 0
    synId2entity = {}
    train_num_remove = []
    test_num_remove = []
    x_left = []
    x_pos_left = []
    x_right = []
    x_pos_right = []
    x_id = []
    y = []
    res = [x_left, x_pos_left, x_right, x_pos_right, x_id, y]

    print('\n开始获取特征...')
    # 获取训练集的SVM特征，并收集{实体ID:实体同义词集}
    main(train_x, train_data, train_label_list, train_pos, train_id_list)
    # 获取测试集的SVM特征，并收集{实体ID:实体同义词集}
    main(test_x, test_data, test_label_list, test_pos, test_id_list)
    print('\n获取特征 finish!')
    print('num_candidates:{}\texits:{}\tnot_exits:{}'.format(num_candidates, exits, not_exits))

    assert len(x_id)==len(x_left)==len(x_right)==len(y)

    # 保存训练数据
    with open('data/train_def.pkl', "wb") as f:
        res = (x_left, x_pos_left, x_right, x_pos_right, y)
        pkl.dump(res, f, -1)

    with open('data/synId2entity.txt', "w") as f:
        for key,value in synId2entity.items():
            f.write('{}\t{}'.format(key, '::,'.join(value)))
            f.write('\n')

    x_id_dict = {value:idx+1 for idx, value in enumerate(list(set(x_id)))}  # 只包含golden实体
    with open('data/x_id_dict.pkl', 'wb') as f:
        pkl.dump((x_id_dict), f, -1)
    with open('data/x_id.pkl', 'wb') as f:
        pkl.dump((x_id), f, -1)

    # # 第二次抽特征时使用！
    # with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/data/x_id_dict2.pkl', 'rb') as f:
    #     x_id_dict = pkl.load(f)

    x_id_new = []
    for id in x_id:
        index = x_id_dict[id]
        x_id_new.append([index])

    id_embedding = np.zeros((len(x_id_dict) + 1, 200))
    # for key, i in x_id_dict.items():
    #     vec = Id2synvec.get(key)
    #     if vec is None:
    #         print('未找到对应向量')
    #         vec = np.random.uniform(-0.1, 0.1, 200)  # 未登录词均统一表示
    #     id_embedding[i] = vec

    # 第二次抽特征时, 保存数据
    with open('data/train_def2.pkl', "wb") as f:
        res = (x_id_new, id_embedding)
        pkl.dump(res, f, -1)

    with open('data/train_def.txt', "w") as f:
        for i in range(len(x_id_new)):
            f.write('{}\t{}\t{}\t{}\t{}\t{}'.format(y[i], x_id_new[i],x_left[i],x_pos_left[i],x_right[i], x_pos_right[i]))
            f.write('\n')
