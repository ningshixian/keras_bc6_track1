import codecs
import csv
import datetime
import os
import pickle as pkl
import string
import xml.dom.minidom
import xml.dom.minidom
from xml.dom.minidom import parse
from collections import OrderedDict
import Levenshtein  # pip install python-Levenshtein
import numpy as np
from bioservices import UniProt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from sample.utils.helpers import get_stop_dic, pos_surround
from sample.utils.helpers import makeEasyTag, Indent, entityNormalize, cos_sim, extract_id_from_res

u = UniProt()

"""
将实体预测结果，以特定格式写入XML文件，用于scorer进行评估
"""


def post_process(sen_list, BioC_path, maxlen, predLabels):

    path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/idx_line2pmc_id_test.txt'
    idx_line2pmc_id = {}
    with open(path, 'r') as f:
        for line in f:
            idx_line, pmc_id = line.split('\t')
            idx_line2pmc_id[idx_line] = pmc_id

    idx_line = -1
    pmc_id_entity_list = {}

    files = os.listdir(BioC_path)
    files.sort()
    for j in tqdm(range(len(files))):  # 遍历文件夹
        file = files[j]
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = BioC_path + "/" + file
            try:
                DOMTree = parse(f)  # 使用minidom解析器打开 XML 文档
                collection = DOMTree.documentElement  # 得到了根元素对象
            except:
                print('异常情况：'.format(f))
                continue

            source = collection.getElementsByTagName("source")[0].childNodes[0].data
            date = collection.getElementsByTagName("date")[0].childNodes[0].data  # 时间
            key = collection.getElementsByTagName("key")[0].childNodes[0].data

            # 一、生成dom对象，根元素名collection
            impl = xml.dom.minidom.getDOMImplementation()
            dom = impl.createDocument(None, 'collection', None)  # 创建DOM文档对象
            root = dom.documentElement  # 创建根元素

            source = makeEasyTag(dom, 'source', source)
            date = makeEasyTag(dom, 'date', datetime.datetime.now().strftime('%Y-%m-%d'))
            key = makeEasyTag(dom, 'key', key)

            # 给根节点添加子节点
            root.appendChild(source)
            root.appendChild(date)
            root.appendChild(key)

            # 在集合中获取所有 document 的内容
            documents = collection.getElementsByTagName("document")
            for doc in documents:
                id = doc.getElementsByTagName("id")[0].childNodes[0].data
                sourcedata_document = doc.getElementsByTagName("infon")[0].childNodes[0].data
                doi = doc.getElementsByTagName("infon")[1].childNodes[0].data
                pmc_id = doc.getElementsByTagName("infon")[2].childNodes[0].data
                figure = doc.getElementsByTagName("infon")[3].childNodes[0].data
                sourcedata_figure_dir = doc.getElementsByTagName("infon")[4].childNodes[0].data

                document = dom.createElement('document')
                id_node = makeEasyTag(dom, 'id', str(id))
                s_d_node = makeEasyTag(dom, 'infon', str(sourcedata_document))
                doi_node = makeEasyTag(dom, 'infon', str(doi))
                pmc_id_node = makeEasyTag(dom, 'infon', str(pmc_id))
                figure_node = makeEasyTag(dom, 'infon', str(figure))
                s_f_d_node = makeEasyTag(dom, 'infon', str(sourcedata_figure_dir))
                s_d_node.setAttribute('key', 'sourcedata_document')  # 向元素中加入属性
                doi_node.setAttribute('key', 'doi')  # 向元素中加入属性
                pmc_id_node.setAttribute('key', 'pmc_id')  # 向元素中加入属性
                figure_node.setAttribute('key', 'figure')  # 向元素中加入属性
                s_f_d_node.setAttribute('key', 'sourcedata_figure_dir')  # 向元素中加入属性
                document.appendChild(id_node)
                document.appendChild(s_d_node)
                document.appendChild(doi_node)
                document.appendChild(pmc_id_node)
                document.appendChild(figure_node)
                document.appendChild(s_f_d_node)

                passages = doc.getElementsByTagName("passage")
                for passage in passages:
                    text = passage.getElementsByTagName('text')[0].childNodes[0].data
                    text_byte = text.encode('utf-8')
                    '''每读取一篇passage，在<annotation>结点记录识别实体'''
                    idx_line += 1
                    s = sen_list[idx_line][:maxlen]  # 单词列表形成的句子
                    prediction = predLabels[idx_line]

                    # 根据预测结果来抽取句子中的所有实体
                    entities = getEntityList(s, prediction)

                    pmc_id = idx_line2pmc_id[str(idx_line)]
                    if pmc_id not in pmc_id_entity_list:
                        pmc_id_entity_list[pmc_id] = []
                    for entity in entities:
                        if entity not in pmc_id_entity_list[pmc_id]:
                            pmc_id_entity_list[pmc_id].append(entity)

    return idx_line2pmc_id, pmc_id_entity_list


def readSynVec():
    '''
    读取AutoExtend训练得到的同义词集向量，和停用词词典
    '''
    # synsetsVec_path1 = '/home/administrator/PycharmProjects/embedding/AutoExtend_Gene/synsetsVec.txt'
    # synsetsVec_path2 = '/home/administrator/PycharmProjects/embedding/AutoExtend_Protein/synsetsVec.txt'
    #
    # geneId2vec = {}
    # with open(synsetsVec_path1, 'r') as f:
    #     for line in f:
    #         splited = line.strip().split(' ')
    #         geneId = splited[0].lower()
    #         vec = np.asarray(splited[1:], dtype=np.float32)
    #         geneId2vec[geneId] = vec
    # # print(geneId2vec['31459'])
    #
    #
    # proteinId2vec = {}
    # with open(synsetsVec_path2, 'r') as f:
    #     for line in f:
    #         splited = line.strip().split(' ')
    #         proteinId = splited[0].lower()
    #         vec = np.asarray(splited[1:], dtype=np.float32)
    #         proteinId2vec[proteinId] = vec

    proteinId2vec = {}
    geneId2vec = {}
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/synsets.txt', 'r') as f:
        for line in f:
            id, entities = line.split('\t')
            proteinId2vec[id] = np.random.uniform(-1, 1, 200)

    return geneId2vec, proteinId2vec


def readBinEmbedFile(embFile, word_size):
    """
    读取二进制格式保存的词向量文件，引入外部知识
    """
    print("\nProcessing Embedding File...")
    from collections import OrderedDict
    import word2vec
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.25, 0.25, word_size)

    model = word2vec.load(embFile)
    print('加载词向量文件完成')
    for i in tqdm(range(len(model.vectors))):
        vector = model.vectors[i]
        word = model.vocab[i].lower()   # convert all characters to lowercase
        embeddings[word] = vector
    return embeddings


def get_w2v():
    if os.path.exists('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/word2vec.pkl'):
        with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/word2vec.pkl', "rb") as f:
            word2vec = pkl.load(f)
    else:
        embedPath = r'/home/administrator/PycharmProjects/embedding'
        embedFile = r'wikipedia-pubmed-and-PMC-w2v.bin'
        word2vec = readBinEmbedFile(embedPath+'/'+embedFile, 200)
        with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/word2vec.pkl', "wb") as f:
            pkl.dump(word2vec, f, -1)
    return word2vec


# def getXlsxData(path):
#     from openpyxl import load_workbook
#
#     wb = load_workbook(path)  # 加载一个工作簿
#     sheets = wb.get_sheet_names()  # 获取各个sheet的名字
#     sheet0 = sheets[0]  # 第一个表格的名称
#     ws = wb.get_sheet_by_name(sheet0)  # 获取特定的 worksheet
#
#     # 获取表格所有行和列，两者都是可迭代的
#     rows = ws.rows
#     # columns = ws.columns
#
#     # 行迭代
#     content = []
#     for row in rows:
#         line = [col.value for col in row]
#         content.append(line)


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
        # f_csv = csv.reader(f)
        # headers = next(f_csv)
        # print(headers)
        f_csv = csv.DictReader(f)
        for row in f_csv:
            if row['obj'].startswith('NCBI gene:') or \
                    row['obj'].startswith('Uniprot:') or \
                    row['obj'].startswith('gene:') or \
                    row['obj'].startswith('protein:'):

                text = row['text'].lower()

                if text not in entity2id:
                    entity2id[text] = []
                if row['obj'] not in entity2id[text]:
                    entity2id[text].append(row['obj'])
                # entity2id[row['text']] = list(set(entity2id[row['text']]))
        print('entity2id字典总长度：{}'.format(len(entity2id)))   # 4221

    # 拓展实体词典(去掉实体的标点符号)
    for key, value in entity2id.items():
        if len(value)>1:
            num_word_multiID+=1
        entity2id_new[key] = value
        for char in string.punctuation:
            if char in key:
                key = key.replace(char, '')
        if key not in entity2id_new:
            entity2id_new[key] = value
    entity2id = {}
    print('其中，多ID实体的个数：{}'.format(num_word_multiID))    # 1562
    return entity2id_new


def getEntityList(s, predLabels):
    '''
    抽取句子中的所有实体
    '''
    entity_dict = OrderedDict()
    entity = ''
    prex = 0
    for tokenIdx in range(len(s)):
        label = predLabels[tokenIdx]
        word = s[tokenIdx]
        if label == 1 or label == 0:
            if entity:
                position = tokenIdx - len(entity.split())
                entity_dict[entityNormalize(entity, s, tokenIdx - len(entity.split()))] = position
                entity = ''
            if label==1:
                entity = str(word) + ' '
        else:
            if prex == 1 or prex == 2:
                entity += word + ' '
            else:
                print('标签错误！跳过')
        prex = label
    if not entity == '':
        position = tokenIdx - len(entity.split())
        entity_dict[entityNormalize(entity, s, tokenIdx - len(entity.split()))] = position
        entity = ''

    # l2 = list(set(entity_list))  # 去除相同元素
    # entities = sorted(l2, key=entity_list.index)  # 不改变原list顺序

    # # 多个词组成的实体中，单个组成词也可能是实体
    # temp_entities = entities.copy()     # 字典的直接赋值和copy的区别（浅拷贝引用，深拷贝）
    # for entity in temp_entities:
    #     splited = entity.split(' ')
    #     if len(splited)>1:
    #         for e in splited:
    #             if e in entity2id and e not in entities:
    #                 entities.append(e)

    return entity_dict


def searchEntityId(s, predLabels, entity_tag_consisteny, entity2id):
    ''' 
    对识别的实体进行ID链接：
    
    先是词典精确匹配
    然后是知识库API匹配
    最后是模糊匹配
    '''
    entities_dict = getEntityList(s, predLabels)
    # entities = entity_tag_consisteny
    entities_new = entities_dict

    id_list = {}
    for entity in entities_dict:

        temp = entity
        for char in string.punctuation:
            if char in temp:
                temp = temp.replace(char, '')

        # 词典精确匹配
        if entity.lower() in entity2id:
            Ids = entity2id[entity.lower()]
            # Ids = idFilter(type, Ids)     # 不进行筛选，否则ID几乎全被干掉了
            id_list[entity] = Ids
            continue
        elif temp.lower() in entity2id:
            Ids = entity2id[temp.lower()]
            # Ids = idFilter(type, Ids)
            id_list[entity] = Ids
            continue

        # 数据库API查询1-reviewed
        res_reviewed = u.search(entity + '+reviewed:yes', frmt="tab", columns="id", limit=5)
        if res_reviewed==400:
            print('请求无效\n')
            print(entity)   # ab′)2
            entities_new.pop(entity)
            # entities_new.remove(entity)
            continue
        if res_reviewed:  # 若是有返回结果
            Ids = extract_id_from_res(res_reviewed)
            id_list[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
            entity2id[entity.lower()] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 将未登录实体添加到实体ID词典中
            continue

        # 数据库API查询1-unreviewed
        unres_reviewed = u.search(entity, frmt="tab", columns="id", limit=5)
        if unres_reviewed==400:
            print('请求无效\n')
            print(entity)
            entities_new.pop(entity)
            # entities_new.remove(entity)
            continue
        if unres_reviewed:  # 若是有返回结果
            Ids = extract_id_from_res(unres_reviewed)
            id_list[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
            entity2id[entity.lower()] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 将未登录实体添加到实体ID词典中
            continue

        # 数据库API查询2-reviewed
        res_reviewed = u.search(temp + '+reviewed:yes', frmt="tab", columns="id", limit=5)
        if res_reviewed==400:
            print('请求无效\n')
            print(entity)
            entities_new.pop(entity)
            # entities_new.remove(entity)
            continue
        if res_reviewed:
            Ids = extract_id_from_res(res_reviewed)
            id_list[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
            entity2id[entity.lower()] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 将未登录实体添加到实体ID词典中
            continue

        # 数据库API查询2-unreviewed
        unres_reviewed = u.search(temp, frmt="tab", columns="id", limit=5)
        if unres_reviewed==400:
            print('请求无效\n')
            print(entity)
            entities_new.pop(entity)
            # entities_new.remove(entity)
            continue
        if unres_reviewed:
            Ids = extract_id_from_res(res_reviewed)
            id_list[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
            entity2id[entity.lower()] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 将未登录实体添加到实体ID词典中
            continue

        # 模糊匹配--计算 Jaro–Winkler 距离
        max_score = -1
        max_score_key = ''
        for key in entity2id.keys():
            score = Levenshtein.jaro_winkler(key, entity.lower())
            if score > max_score:
                max_score = score
                max_score_key = key
        Ids = entity2id.get(max_score_key)
        id_list[entity] = Ids

    # if entity == 'Ubc9':
    #     print(entity2id.get(entity))
    return entities_new, id_list


def entity_disambiguation(entity_id, geneId2vec, proteinId2vec, zhixin):
    score = 0.7   # cos相似度的阙值

    for id in entity_id:    # 计算质心与实体每个歧义（同义词集）的cos相似度
        id = id.split('|')[0] if '|' in id else id
        entityType = id.split(':')[0]
        splited_id = id.split(':')[1]
        if entityType.startswith('NCBI') or entityType.startswith('gene'):
            synset_vec = geneId2vec.get(splited_id)
        elif entityType.startswith('Uniprot') or entityType.startswith('protein'):
            synset_vec = proteinId2vec.get(splited_id)
        else:
            print('类型错误!')
            continue
        if synset_vec is not None:
            # print('{}对应的同义词集向量 ok'.format(id))
            cos_sim_score = cos_sim(zhixin, synset_vec)
            if cos_sim_score > score:
                score = cos_sim_score
                type_id = id
        else:
            # print('{}对应的同义词集向量 failed'.format(id))
            pass
    return type_id


def writeOutputToFile(path, predLabels, maxlen):
    '''
    按顺序读取原文件夹中的xml格式文件
    同时，对应每个text生成annotation标签：
        getElementsByTagName方法：获取孩子标签
        getAttribute方法：可以获得元素的属性所对应的值。
        firstChild.data≈childNodes[0].data：返回被选节点的第一个子标签对之间的数据
    将实体预测结果写入XML文件

    :param path: 测试数据路径
    :param predLabels: 测试数据的实体预测结果
    :param maxlen: 句子截断长度
    :param split_pos: 划分训练集和验证集的位置
    :return:
    '''
    idx_line = -1
    num_entity_no_id = 0
    words_with_multiId = []
    base = r'/home/administrator/桌面/BC6_Track1'
    BioC_path = base + '/' + 'test_corpus_20170804/caption_bioc'    # 测试数据文件夹
    dic_path = base + '/' + 'BioIDtraining_2/annotations.csv'   # 实体ID查找词典文件
    result_path = base + '/' + 'test_corpus_20170804/prediction'

    # 获取测试集所有句子list的集合
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

    idx_line2pmc_id, pmc_id2entity_list = post_process(sen_list, BioC_path, maxlen, predLabels)


    entity2id = getCSVData(dic_path)    # 读取实体ID查找词典
    # geneId2vec, proteinId2vec = readSynVec() # 读取AutoExtend训练获得的同义词集向量
    # stop_word = get_stop_dic()
    # word2vec = get_w2v()    # 读取词向量词典

    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/clf.pkl', 'rb') as f:
        svm = pkl.load(f)
    with open(r'/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test.pkl', "rb") as f:
        test_x, test_y, test_char, test_cap, test_pos, test_chunk = pkl.load(f)
    synId2entity = {}
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/synId2entity.txt') as f:
        for line in f:
            s1, s2 = line.split('\t')
            entities = s2.strip('\n').split('::,')
            synId2entity[s1] = entities

    files = os.listdir(BioC_path)
    files.sort()
    for j in tqdm(range(len(files))):  # 遍历文件夹
        file = files[j]
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = BioC_path + "/" + file
            try:
                DOMTree = parse(f)  # 使用minidom解析器打开 XML 文档
                collection = DOMTree.documentElement  # 得到了根元素对象
            except:
                print('异常情况：'.format(f))
                continue

            source = collection.getElementsByTagName("source")[0].childNodes[0].data
            date = collection.getElementsByTagName("date")[0].childNodes[0].data    # 时间
            key = collection.getElementsByTagName("key")[0].childNodes[0].data

            # 一、生成dom对象，根元素名collection
            impl = xml.dom.minidom.getDOMImplementation()
            dom = impl.createDocument(None, 'collection', None)   # 创建DOM文档对象
            root = dom.documentElement       # 创建根元素

            source = makeEasyTag(dom, 'source', source)
            date = makeEasyTag(dom, 'date', datetime.datetime.now().strftime('%Y-%m-%d'))
            key = makeEasyTag(dom, 'key', key)

            # 给根节点添加子节点
            root.appendChild(source)
            root.appendChild(date)
            root.appendChild(key)

            # 在集合中获取所有 document 的内容
            documents = collection.getElementsByTagName("document")
            for doc in documents:
                id = doc.getElementsByTagName("id")[0].childNodes[0].data
                sourcedata_document = doc.getElementsByTagName("infon")[0].childNodes[0].data
                doi = doc.getElementsByTagName("infon")[1].childNodes[0].data
                pmc_id = doc.getElementsByTagName("infon")[2].childNodes[0].data
                figure = doc.getElementsByTagName("infon")[3].childNodes[0].data
                sourcedata_figure_dir = doc.getElementsByTagName("infon")[4].childNodes[0].data

                document = dom.createElement('document')
                id_node = makeEasyTag(dom, 'id', str(id))
                s_d_node = makeEasyTag(dom, 'infon', str(sourcedata_document))
                doi_node = makeEasyTag(dom, 'infon', str(doi))
                pmc_id_node = makeEasyTag(dom, 'infon', str(pmc_id))
                figure_node = makeEasyTag(dom, 'infon', str(figure))
                s_f_d_node = makeEasyTag(dom, 'infon', str(sourcedata_figure_dir))
                s_d_node.setAttribute('key', 'sourcedata_document')  # 向元素中加入属性
                doi_node.setAttribute('key', 'doi')  # 向元素中加入属性
                pmc_id_node.setAttribute('key', 'pmc_id')  # 向元素中加入属性
                figure_node.setAttribute('key', 'figure')  # 向元素中加入属性
                s_f_d_node.setAttribute('key', 'sourcedata_figure_dir')  # 向元素中加入属性
                document.appendChild(id_node)
                document.appendChild(s_d_node)
                document.appendChild(doi_node)
                document.appendChild(pmc_id_node)
                document.appendChild(figure_node)
                document.appendChild(s_f_d_node)

                passages = doc.getElementsByTagName("passage")
                for passage in passages:
                    text = passage.getElementsByTagName('text')[0].childNodes[0].data
                    text_byte = text.encode('utf-8')
                    '''每读取一篇passage，在<annotation>结点记录识别实体'''
                    idx_line += 1
                    pmc_id = idx_line2pmc_id[str(idx_line)]
                    annotation_list = []
                    s = sen_list[idx_line][:maxlen]  # 单词列表形成的句子
                    prediction = predLabels[idx_line]

                    # 根据预测结果来抽取句子中的所有实体，并进行实体链接
                    # start_time = time.time()
                    entity_tag_consisteny = []
                    entity_list = pmc_id2entity_list[pmc_id]
                    for entity in entity_list:
                        if not text.find(entity)==-1:
                            entity_tag_consisteny.append(entity)
                    entities, entity_ids = searchEntityId(s, prediction, entity_tag_consisteny, entity2id)
                    # time_diff = time.time() - start_time
                    # print("%.2f sec" % time_diff)

                    for entity, tokenIdx in entities.items():
                        entity_id = entity_ids[entity]
                        for id in entity_id:
                            if id.startswith('gene:') or id.startswith('protein:'):
                                continue
                            id = id.split('|')[0] if '|' in id else id
                            id = id.split(':')[1]
                            if id not in synId2entity:
                                synId2entity[id] = []
                            if entity not in synId2entity[id]:
                                synId2entity[id].append(entity)

                    ''' 
                    多ID的实体需要进行实体消岐，AutoExtend 实体消歧方法：
                        1、计算实体所在句子的质心
                        2、计算质心与实体每个歧义（同义词集）的相似度
                        3、取最大得分作为ID
                    '''
                    annotation_id = 0
                    for entity, tokenIdx in entities.items():
                        entity_id = entity_ids[entity]
                        type_id = entity_id[0]
                        if len(entity_id)>1:
                            if entity not in words_with_multiId:
                                words_with_multiId.append(entity)

                            # zhixin = np.zeros(200)  # 计算质心向量
                            # for word in sen_list[idx_line]:
                            #     if word not in stop_word and word not in string.punctuation + '-':
                            #         vector = word2vec.get(word.lower())
                            #         if vector is None:
                            #             vector = np.random.uniform(-0.1, 0.1, 200)
                            #         zhixin += vector
                            # # type_id = entity_disambiguation(entity_id, geneId2vec, proteinId2vec, zhixin)
                            #
                            # id_list = []
                            # for id in entity_id:
                            #     pos, surroundding_word = pos_surround(test_x[idx_line], test_pos[idx_line], tokenIdx, entity)
                            #     # pos, surround = get_x_y(idx_line, tokenIdx, entity)
                            #     id = id.split('|')[0] if '|' in id else id
                            #     id = id.split(':')[1]
                            #     if id in geneId2vec:
                            #         syn_vec = np.round(geneId2vec.get(id),6)
                            #     elif id in proteinId2vec:
                            #         syn_vec = np.round(proteinId2vec.get(id),6)
                            #     else:
                            #         print('未找到对应同义词集向量，随机初始化')
                            #         syn_vec = np.round(np.random.uniform(-0.1, 0.1, 200),6)
                            #     s_profuction = list(np.multiply(zhixin, syn_vec))
                            #
                            #     id_list.append(pos + s_profuction)
                            #     # id_list.append(pos + surroundding_word + s_profuction)
                            # id_list = np.array(id_list)
                            # # print(id_list.shape)  # (?, 212)
                            # scaler = StandardScaler()
                            # scaler.fit(id_list)   # Expected 2D array
                            # id_list = scaler.transform(id_list)
                            # cls = list(svm.predict_proba(id_list))
                            # max_proba = -1
                            # for i in range(len(cls)):
                            #     proba = cls[i][1]
                            #     if proba > max_proba:
                            #         max_proba = proba
                            #         type_id = entity_id[i]

                        elif len(entity_id)==1:  # 说明实体对应了唯一ID
                            type_id = entity_id[0]
                        else:     # 未找到对应的ID
                            print('未找到对应的ID')
                            type_id = 'protein:' + entity
                            num_entity_no_id += 1

                        # 标记句子中所有相同的实体

                        if entity.encode('utf-8') in text_byte:
                            entity_byte = entity.encode('utf-8')
                        else:
                            print(entity)
                            print(text)
                            entity_byte = entity.replace(' ', '').replace('.', '. ').encode('utf-8')
                        offset = -1
                        while 1:
                            offset = text_byte.find(entity_byte, offset+1)   # 二进制编码查找offset
                            if not offset == -1:
                                annotation_id += 1
                                annotation = dom.createElement('annotation')
                                annotation.setAttribute('id', str(annotation_id))
                                infon1 = makeEasyTag(dom, 'infon', type_id)
                                infon1.setAttribute('key', 'type')
                                infon2 = makeEasyTag(dom, 'infon', str(annotation_id))
                                infon2.setAttribute('key', 'sourcedata_figure_annot_id')
                                infon3 = makeEasyTag(dom, 'infon', str(annotation_id))
                                infon3.setAttribute('key', 'sourcedata_article_annot_id')
                                location = dom.createElement('location')
                                location.setAttribute('offset', str(offset))
                                location.setAttribute('length', str(len(entity)))
                                text_node = makeEasyTag(dom, 'text', entity)
                                annotation.appendChild(infon1)
                                annotation.appendChild(infon2)
                                annotation.appendChild(infon3)
                                annotation.appendChild(location)
                                annotation.appendChild(text_node)
                                annotation_list.append(annotation)
                            else:
                                break


                    # 最后串到根结点上，形成一棵树
                    passage1 = dom.createElement('passage')
                    offset1 = makeEasyTag(dom, 'offset', '0')
                    text1 = makeEasyTag(dom, 'text', text)
                    passage1.appendChild(offset1)
                    passage1.appendChild(text1)
                    for annotation in annotation_list:
                        passage1.appendChild(annotation)

                    # 给根节点添加子节点
                    document.appendChild(passage1)
                root.appendChild(document)

            '''
            将DOM对象doc写入文件
            每读完一个file后，将结果写入同名的XML文件
            '''
            Indent(dom, dom.documentElement) # 美化
            outputName = result_path + '/' + file
            f = open(outputName, 'w')
            writer = codecs.lookup('utf-8')[3](f)
            dom.writexml(f, indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')
            writer.close()
            f.close()

    print('测试集预测结果写入成功！')
    print('{}个词未找到对应的ID'.format(num_entity_no_id))    # 0
    print('{}个词有歧义'.format(len(words_with_multiId)))    # 701
    print('完结撒花')

    with open('synsets.txt', "w") as f:
        for key, value in synId2entity.items():
            f.write('{}\t{}'.format(key, '::,'.join(value)))
            f.write('\n')
