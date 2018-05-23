"""
将实体预测结果，以特定格式写入XML文件，用于scorer进行评估
"""
import codecs
import csv
import datetime
import os
import pickle as pkl
import string
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xml.dom.minidom
import xml.dom.minidom
from xml.dom.minidom import parse
import Levenshtein  # pip install python-Levenshtein
from bioservices import UniProt
from Bio import Entrez
from sample.utils.helpers import get_stop_dic, pos_surround
from sample.utils.helpers import makeEasyTag, Indent, entityNormalize, cos_sim, extract_id_from_res
u = UniProt()


def post_process(sen_list, BioC_path, maxlen, predLabels):
    '''
    # 实体标注一致性
    :param sen_list:
    :param BioC_path:
    :param maxlen:
    :param predLabels:
    :return:
    '''

    path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/idx_line2pmc_id_test.txt'
    idx_line2pmc_id = {}
    with open(path, 'r') as f:
        for line in f:
            idx_line, pmc_id = line.split('\t')
            idx_line2pmc_id[idx_line] = pmc_id

    idx_line = -1
    pmc_id2entity_list = {}

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
                    if pmc_id not in pmc_id2entity_list:
                        pmc_id2entity_list[pmc_id] = []
                    for entity in entities:
                        if entity not in pmc_id2entity_list[pmc_id]:
                            pmc_id2entity_list[pmc_id].append(entity)

    return idx_line2pmc_id, pmc_id2entity_list


def readSynVec(synsetsVec_path):
    '''
    读取AutoExtend训练得到的同义词集向量
    '''
    proteinId2vec = {}
    with open(synsetsVec_path, 'r') as f:
        for line in f:
            splited = line.strip('\n').split(' ')
            proteinId = splited[0]
            vec = np.asarray(splited[1:], dtype=np.float32)
            proteinId2vec[proteinId] = vec

    return proteinId2vec


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
    # if os.path.exists('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/word2vec.pkl'):
    #     with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/word2vec.pkl', "rb") as f:
    #         word2vec = pkl.load(f)
    # else:
    #     embedPath = r'/home/administrator/PycharmProjects/embedding'
    #     embedFile = r'wikipedia-pubmed-and-PMC-w2v.bin'
    #     word2vec = readBinEmbedFile(embedPath+'/'+embedFile, 200)
    #     with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/word2vec.pkl', "wb") as f:
    #         pkl.dump(word2vec, f, -1)
    # return word2vec
    with open(r'/home/administrator/PycharmProjects/embedding/emb.pkl', "rb") as f:
        embedding_matrix = pkl.load(f)
    with open(r'/home/administrator/PycharmProjects/embedding/length.pkl', "rb") as f:
        word_maxlen, sentence_maxlen = pkl.load(f)
    return embedding_matrix, sentence_maxlen


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


def getCSVData(csv_path, entity2id):
    '''
    获取实体ID词典 {'entity':[id1, id2, ...]}
    实体区分大小写
    id list 按频度排序×
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
        print('entity2id字典总长度：{}'.format(len(entity2id)))  # 5096   1950

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
    print('F4/80: {}'.format(entity2id_new['F4/80']))   # ['Uniprot:Q61549']    ['NCBI gene:13733', 'Uniprot:Q61549']
    print('其中，多ID实体的个数：{}'.format(num_word_multiID))    # 1538  494
    return entity2id_new


def getEntityList(s, predLabels):
    '''
    抽取句子中的所有实体
    '''
    entity_list = []
    position_list = []
    leixing_list = []
    entity = ''
    prex = 0
    for tokenIdx in range(len(s)):
        label = predLabels[tokenIdx]
        word = s[tokenIdx]
        if label == 0 or label == 1 or label==3:
            if entity:
                splited = entity.split('/')
                if len(splited)==2:
                    if not splited[1].strip().isdigit():
                        entity = [splited[0].strip(), splited[1].strip()]
                        if entity[0] not in entity_list:
                            entity_list.append(entity[0])
                            position = tokenIdx - 3
                            position_list.append(position)
                            leixing_list.append('protein' if prex == 1 or prex==2 else 'gene')
                        if entity[1] not in entity_list:
                            entity_list.append(entity[1])
                            position = tokenIdx - 1
                            position_list.append(position)
                            leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
                    else:
                        position = tokenIdx - len(entity.split())
                        entity = entityNormalize(entity, s, tokenIdx - len(entity.split()))
                        if entity not in entity_list:
                            entity_list.append(entity)
                            position_list.append(position)
                            leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
                else:
                    position = tokenIdx - len(entity.split())
                    entity = entityNormalize(entity, s, tokenIdx - len(entity.split()))
                    if entity not in entity_list:
                        entity_list.append(entity)
                        position_list.append(position)
                        leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
                entity = ''
            if label==1 or label==3:
                entity = str(word) + ' '
            prex = label
        elif label == 2:
            if prex == 1 or prex == 2:
                entity += word + ' '
                prex = label
            else:
                # [1 2 2 0 0 0 0 0 2 2 0]
                # print(s)
                # print(predLabels)
                print('标签错误！跳过')    # 154次出现
        else:
            if prex == 3 or prex == 4:
                entity += word + ' '
                prex = label
            else:
                print('标签错误！跳过2')    #

    if not entity == '':
        print('!!!!!!!!!!!!!!')
        splited = entity.split('/')
        if len(splited) == 2:
            if not splited[1].strip().isdigit():
                entity = [splited[0].strip(), splited[1].strip()]
                if entity[0] not in entity_list:
                    entity_list.append(entity[0])
                    position = tokenIdx - 3
                    position_list.append(position)
                    leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
                if entity[1] not in entity_list:
                    entity_list.append(entity[1])
                    position = tokenIdx - 1
                    position_list.append(position)
                    leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
            else:
                position = tokenIdx - len(entity.split())
                entity = entityNormalize(entity, s, tokenIdx - len(entity.split()))
                if entity not in entity_list:
                    entity_list.append(entity)
                    position_list.append(position)
                    leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
        else:
            position = tokenIdx - len(entity.split())
            entity = entityNormalize(entity, s, tokenIdx - len(entity.split()))
            if entity not in entity_list:
                entity_list.append(entity)
                position_list.append(position)
                leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
        entity = ''
        prex = label

    # l2 = list(set(entity_list))  # 去除相同元素
    # entities = sorted(l2, key=entity_list.index)  # 不改变原list顺序

    # # 多个词组成的实体中，单个组成词也可能是实体
    # temp_entities = entity_list.copy()     # 字典的直接赋值和copy的区别（浅拷贝引用，深拷贝）
    # for entity in temp_entities:
    #     splited = entity.split(' ')
    #     if len(splited)>1:
    #         for e in splited:
    #             if e in entity2id and e not in entity_list:
    #                 entity_list.append(e)

    return entity_list, position_list, leixing_list


def searchEntityId(s, predLabels, entity_tag_consisteny, entity2id, protein2id, gene2id):
    ''' 
    对识别的实体进行ID链接：
    
    先是词典精确匹配
    然后是知识库API匹配
    最后是模糊匹配
    '''
    entity_list, position_list, leixing_list = getEntityList(s, predLabels)
    assert len(entity_list)==len(position_list)==len(leixing_list)
    entities_new = entity_list.copy()

    id_dict = {}
    for i in range(len(entity_list)):
        entity = entity_list[i]
        leixing = leixing_list[i]

        temp = entity
        for char in string.punctuation:
            if char in temp:
                temp = temp.replace(char, ' ' + char + ' ')
        temp = temp.strip()

        # 词典精确匹配
        if entity in entity2id:
            Ids = entity2id[entity]
            id_dict[entity] = Ids
            continue
        if entity.lower() in entity2id:
            Ids = entity2id[entity.lower()]
            id_dict[entity] = Ids
            continue
        if temp in entity2id:
            Ids = entity2id[temp]
            id_dict[entity] = Ids
            continue
        if entity.replace(' ', '') in entity2id:
            Ids = entity2id[entity.replace(' ', '')]
            id_dict[entity] = Ids
            continue

        if leixing == 'gene' and entity in gene2id:
            id_dict[entity] = gene2id[entity]
        if leixing == 'gene' and entity.lower() in gene2id:
            id_dict[entity] = gene2id[entity.lower()]
        elif leixing == 'protein' and entity in protein2id:
            id_dict[entity] = protein2id[entity]
        elif leixing == 'protein' and entity.lower() in protein2id:
            id_dict[entity] = protein2id[entity.lower()]
        else:

            # NCBI-gene数据库API查询
            handle = Entrez.esearch(db="gene", idtype="acc", sort='relevance', term=entity)
            record = Entrez.read(handle)
            if record["IdList"]:
                id_dict[entity] = ['NCBI gene:' + record["IdList"][i] for i in range(len(record["IdList"][:3]))]
                entity2id[entity] = ['NCBI gene:' + record["IdList"][i] for i in range(len(record["IdList"][:3]))]
                # print(record["IdList"][:3])
                continue

            # 数据库API查询1-reviewed
            res_reviewed = u.search(entity + '+reviewed:yes', frmt="tab", columns="id", limit=3)
            if res_reviewed.isdigit():
                print('请求无效\n')
            elif res_reviewed:  # 若是有返回结果
                Ids = extract_id_from_res(res_reviewed)
                id_dict[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
                entity2id[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 将未登录实体添加到实体ID词典中
                continue

            # 数据库API查询1-unreviewed
            unres_reviewed = u.search(entity, frmt="tab", columns="id", limit=3)
            if unres_reviewed.isdigit():
                print('请求无效\n')
                # entities_new.remove(entity)
            elif unres_reviewed:  # 若是有返回结果
                Ids = extract_id_from_res(unres_reviewed)
                id_dict[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
                entity2id[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 将未登录实体添加到实体ID词典中
                continue

            # 数据库API查询2-reviewed
            res_reviewed = u.search(temp + '+reviewed:yes', frmt="tab", columns="id", limit=3)
            if res_reviewed.isdigit():
                print('请求无效\n')
                # entities_new.remove(entity)
            elif res_reviewed:
                print('# 数据库API查询2-reviewed')
                Ids = extract_id_from_res(res_reviewed)
                id_dict[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
                entity2id[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 将未登录实体添加到实体ID词典中
                continue

            # 数据库API查询2-unreviewed
            unres_reviewed = u.search(temp, frmt="tab", columns="id", limit=3)
            if unres_reviewed.isdigit():
                print('请求无效\n')
                # entities_new.remove(entity)
            elif unres_reviewed:
                print('# # 数据库API查询2-unreviewed')
                Ids = extract_id_from_res(unres_reviewed)
                id_dict[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
                entity2id[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 将未登录实体添加到实体ID词典中
                continue


        print('未找到{}的ID，空'.format(entity))  # 152次出现
        id_dict[entity] = []

        # # 模糊匹配--计算 Jaro–Winkler 距离
        # max_score = -1
        # max_score_key = ''
        # for key in entity2id.keys():
        #     score = Levenshtein.jaro_winkler(key, entity.lower())
        #     if score > max_score:
        #         max_score = score
        #         max_score_key = key
        # Ids = entity2id.get(max_score_key)
        # id_dict[entity] = Ids

    return entity_list, position_list, id_dict


def entity_disambiguation(entity_id, Id2synvec, zhixin):
    '''
    计算质心与实体向量（同义词集）的cos相似度进行消歧
    :param entity_id:
    :param Id2synvec:
    :param zhixin:
    :return:
    '''
    score = 0.65   # cos相似度的阙值
    type_id = ''
    for i in range(len(entity_id)):    # 计算质心与实体每个歧义（同义词集）的cos相似度
        id = entity_id[i]
        if '|' in id:
            id = id.split('|')
            id = '|'.join([item.split(':')[1] for item in id])
        else:
            id = id.split(':')[1]
        synset_vec = Id2synvec.get(id)
        if synset_vec is None:
            synset_vec = np.round(np.random.uniform(-0.1, 0.1, 200), 6)
            Id2synvec[id] = synset_vec

        cos_sim_score = cos_sim(zhixin, synset_vec)
        if cos_sim_score > score:
            score = cos_sim_score
            type_id = entity_id[i]
    return type_id


def get_test_out_data(path):
    '''
    获取测试集所有句子list的集合
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


def writeOutputToFile(path, predLabels):
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
    test_dic_path = base + '/' + 'test_corpus_20170804/annotations.csv'
    result_path = base + '/' + 'test_corpus_20170804/prediction'
    synsetsVec_path = '/home/administrator/PycharmProjects/embedding/data/synsetsVec.txt'

    # 读取golden实体的ID词典
    entity2id = {}
    entity2id = getCSVData(dic_path, entity2id)
    # entity2id = getCSVData(test_dic_path, entity2id)

    # 读取同义词集向量
    id2synvec = readSynVec(synsetsVec_path)
    # 读取停用词词典
    stop_word = get_stop_dic()
    # 读取测试预料的数据和golden ID
    sen_list = get_test_out_data(path)
    # 读取词向量词典
    embedding_matrix, maxlen = get_w2v()
    # 实体标注一致性
    idx_line2pmc_id, pmc_id2entity_list = post_process(sen_list, BioC_path, maxlen, predLabels)

    with open(r'/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test.pkl', "rb") as f:
        test_x, test_y, test_char, test_cap, test_pos, test_chunk = pkl.load(f)
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/data/clf.pkl', 'rb') as f:
        svm = pkl.load(f)
    with open('data/LocalCollocations.pkl', 'rb') as f:
        features_dict = pkl.load(f)

    synId2entity = {}
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/data/synId2entity.txt') as f:
        for line in f:
            s1, s2 = line.split('\t')
            entities = s2.strip('\n').split('::,')
            synId2entity[s1] = entities
    # print(synId2entity['Q8BIP0'])

    idx2pos = {}
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/pos2idx.txt') as f:
        for line in f:
            pos, idx = line.split('\t')
            idx2pos[idx.strip('\n')] = pos

    protein2id = {}
    gene2id = {}
    protein_path = '/home/administrator/PycharmProjects/embedding/uniprot_sprot.dat2'
    gene_path = '/home/administrator/PycharmProjects/embedding/gene_info2'
    with open(gene_path) as f:
        for line in tqdm(f):
            splited = line.split('\t')
            id_list = splited[0].split(';')
            e_list = splited[1].split(';')
            for e in e_list:
                if e not in gene2id:
                    gene2id[e] = []
                for id in id_list:
                    if id not in gene2id[e]:
                        gene2id[e].append(id)
    with open(protein_path) as f:
        for line in tqdm(f):
            splited = line.split('\t')
            id_list = splited[0].split(';')
            e_list = splited[1].split(';')
            for e in e_list:
                if e not in protein2id:
                    protein2id[e] = []
                for id in id_list:
                    if id not in protein2id[e]:
                        protein2id[e].append(id)

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
                    entity_tag_consisteny = []
                    # entity_list = pmc_id2entity_list[pmc_id]
                    # for entity in entity_list:
                    #     if not text.find(entity)==-1:
                    #         entity_tag_consisteny.append(entity)

                    # 根据预测结果获取实体
                    entity_list, position_list, id_dict = searchEntityId(s, prediction, entity_tag_consisteny, entity2id, protein2id, gene2id)
                    # entity_list, position_list, id_dict = searchEntityId(s, prediction, entity_tag_consisteny, entity2id)

                    # 收集需要的实体ID
                    for i in range(len(entity_list)):
                        entity = entity_list[i]
                        entity_id = id_dict[entity]
                        for id in entity_id:
                            if id.startswith('gene:') or id.startswith('protein:'):
                                continue
                            if '|' in id:
                                id = id.split('|')
                                id = '|'.join([item.split(':')[1] for item in id])
                            else:
                                id = id.split(':')[1]
                            if id not in synId2entity:
                                synId2entity[id] = []
                            if entity not in synId2entity[id]:
                                synId2entity[id].append(entity)

                    # 计算质心向量
                    zhixin = np.zeros(200)
                    for word_id in test_x[idx_line]:
                        vector = embedding_matrix[word_id]
                        zhixin += vector
                        # if word not in stop_word and word not in string.punctuation:
                        # vector = word2vec.get(word.lower())
                        # if vector is None:
                        #     vector = np.random.uniform(-0.1, 0.1, 200)
                        # zhixin += vector

                    ''' 针对多ID的实体进行实体消岐，AutoExtend '''
                    annotation_id = 0
                    # for entity, tokenIdx in entities.items():
                    for i in range(len(entity_list)):
                        entity = entity_list[i]
                        position = position_list[i]
                        entity_id = id_dict[entity]
                        if len(entity_id)>1:
                            if entity not in words_with_multiId:
                                words_with_multiId.append(entity)

                            type_id = entity_id[np.random.randint(0, len(entity_id))]
                            # type_id = entity_disambiguation(entity_id, id2synvec, zhixin)

                            # # 若实体的id中存在entity=id的情况，则仅标注其实体类型
                            # if 'protein:'+entity in entity_id :
                            #     print('仅标注其实体类型')
                            #     type_id = 'protein:' + entity
                            # elif 'gene:'+entity in entity_id:
                            #     print('仅标注其实体类型')
                            #     type_id = 'gene:' + entity
                            # else:
                            #     fea_list = []
                            #     for id in entity_id:
                            #         pos, local_collocations = pos_surround(test_x[idx_line], test_pos[idx_line], position, entity, idx2pos, features_dict)
                            #         if '|' in id:
                            #             id = id.split('|')
                            #             id = '|'.join([item.split(':')[1] for item in id])
                            #         else:
                            #             id = id.split(':')[1]
                            #         if id in id2synvec:
                            #             syn_vec = id2synvec.get(id)
                            #         else:
                            #             print('未找到{}的实体向量，{}随机初始化.'.format(entity,id))
                            #             syn_vec = np.round(np.random.uniform(-0.1, 0.1, 200),6)
                            #
                            #         s_profuction = list(np.multiply(zhixin, syn_vec))
                            #         fea_list.append(pos + local_collocations + s_profuction)
                            #
                            #     fea_list = np.array(fea_list)
                            #     # 标准化
                            #     scaler = StandardScaler()
                            #     scaler.fit(fea_list)   # Expected 2D array
                            #     fea_list = scaler.transform(fea_list)
                            #     # 样本距离超平面的距离
                            #     result = list(svm.decision_function(fea_list))
                            #     max_dis = -100
                            #     for i in range(len(result)):
                            #         if result[i] > max_dis:
                            #             max_dis = result[i]
                            #             type_id = entity_id[i]

                        elif len(entity_id)==1:  # 说明实体对应了唯一ID
                            type_id = entity_id[0]
                        else:
                            # print('未找到{}的ID，仅标注其实体类型'.format(entity))
                            type_id = 'protein:' + entity
                            num_entity_no_id += 1

                        '''给句子中所有相同的实体标记上找到的ID'''

                        if entity.encode('utf-8') in text_byte:
                            entity_byte = entity.encode('utf-8')
                        else:
                            entity = entity.replace(' ', '')
                            if entity.encode('utf-8') in text_byte:
                                entity_byte = entity.encode('utf-8')
                            else:
                                entity_byte = entity.replace(',', ', ').replace('.', '. ')
                                if entity.encode('utf-8') in text_byte:
                                    entity_byte = entity.encode('utf-8')
                                else:
                                    entity_byte = entity.encode('utf-8')
                                    print('未在句子中找到{}的offset索引？'.format(entity))
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
    print('{}个词未找到对应的ID'.format(num_entity_no_id))    # 152
    print('{}个词有歧义'.format(len(words_with_multiId)))    # 1757
    print('完结撒花')

    with open('synsets.txt', "w") as f:
        for key, value in synId2entity.items():
            f.write('{}\t{}'.format(key, '::,'.join(value)))
            f.write('\n')
