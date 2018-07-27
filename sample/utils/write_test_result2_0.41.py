"""
将实体预测结果，以特定格式写入XML文件，用于scorer进行评估
"""
import re
import codecs
import csv
import datetime
import os
import pickle as pkl
import string
from urllib.error import URLError
from collections import OrderedDict
import numpy as np
import word2vec
from tqdm import tqdm
import xml.dom.minidom
import xml.dom.minidom
from xml.dom.minidom import parse
from bioservices import UniProt
from Bio import Entrez
from keras.models import load_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.sequence import pad_sequences
import Levenshtein  # pip install python-Levenshtein
from sklearn.preprocessing import StandardScaler
from sample.utils.helpers import get_stop_dic, pos_surround
from sample.utils.helpers import makeEasyTag, Indent, entityNormalize, cos_sim, extract_id_from_res
u = UniProt()

# GPU内存分配
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 按比例
# config.gpu_options.allow_growth = True  # 自适应分配
set_session(tf.Session(config=config))


def post_process(sen_list, BioC_path, maxlen, predLabels):
    '''
    实体标注一致性
    按文档收集相应的实体mention
    '''
    idx_line = -1
    entity2count = {}
    pmc_id2entity_list = {}
    pmc_id2lx_list = {}
    idx_line2pmc_id = {}

    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/idx_line2pmc_id_test.txt', 'r') as f:
        for line in f:
            n_line, pmc_id = line.split('\t')
            idx_line2pmc_id[n_line] = pmc_id.replace('\n', '')

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
                    entity_list, position_list, leixing_list = getEntityList(s, prediction)
                    # 当前句子对应的文档ID
                    pmc_id = idx_line2pmc_id[str(idx_line)]
                    if pmc_id not in pmc_id2entity_list:
                        pmc_id2entity_list[pmc_id] = []
                        pmc_id2lx_list[pmc_id] = []
                    for i in range(len(entity_list)):
                        entity = entity_list[i]
                        lx = leixing_list[i]
                        # 统计实体出现次数
                        if entity not in entity2count:
                            entity2count[entity] = 1
                        else:
                            entity2count[entity] += 1
                        # 仅保留出现2次以上的实体
                        if entity2count[entity]>=2 and entity not in pmc_id2entity_list[pmc_id]:
                            pmc_id2entity_list[pmc_id].append(entity)
                            pmc_id2lx_list[pmc_id].append(lx)

    return idx_line2pmc_id, pmc_id2entity_list, pmc_id2lx_list


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
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.25, 0.25, word_size)

    model = word2vec.load(embFile)
    print('加载词向量文件完成')
    for i in tqdm(range(len(model.vectors))):
        vector = model.vectors[i]
        word = model.vocab[i].lower()  # convert all characters to lowercase
        embeddings[word] = vector
    return embeddings


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


def getCSVData(csv_path):
    '''
    获取实体ID词典 {'entity':[id1, id2, ...]}
    '''
    entity2id = {}
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
                    # if row['text']=='F4/80':
                    #     print(row['text'])
                    #     print(entity)     # f480
        print('entity2id字典总长度：{}'.format(len(entity2id)))  # 5096   1950

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

    print('F4/80: {}'.format(entity2id_2.get('f4 / 80')))  # ['Uniprot:Q61549']    ['NCBI gene:13733', 'Uniprot:Q61549']
    print('F480: {}'.format(entity2id_2.get('f480')))  # ['Uniprot:Q61549']    ['NCBI gene:13733', 'Uniprot:Q61549']
    return entity2id_2

# base = r'/home/administrator/桌面/BC6_Track1'
# test_dic_path = base + '/' + 'test_corpus_20170804/annotations.csv'
# entity2id_test = getCSVData(test_dic_path)

def search_id_from_Uniprot(query_list, reviewed=True):
    # Uniprot 数据库API查询-reviewed
    for query in query_list:
        if reviewed:
            res_reviewed = u.search(query + '+reviewed:yes', frmt="tab", columns="id", limit=5)
        else:
            res_reviewed = u.search(query, frmt="tab", columns="id", limit=5)
        if isinstance(res_reviewed, int):
            print('请求无效+{}'.format(query))
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
        if label == 0 or label == 1 or label == 3:
            if entity:
                splited = entity.split('/')
                if len(splited) == 2:
                    if not splited[1].strip().isdigit():
                        entity = [splited[0].strip(), splited[1].strip()]
                        if len(entity[0]) > 2 and entity[0] not in entity_list:
                            entity_list.append(entity[0])
                            position = tokenIdx - 3
                            position_list.append(position)
                            leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
                        if len(entity[1]) > 2 and entity[1] not in entity_list:
                            entity_list.append(entity[1])
                            position = tokenIdx - 1
                            position_list.append(position)
                            leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
                    else:
                        position = tokenIdx - len(entity.split())
                        entity = entityNormalize(entity, s, tokenIdx - len(entity.split()))
                        if len(entity) > 2 and entity not in entity_list:
                            entity_list.append(entity)
                            position_list.append(position)
                            leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
                else:
                    position = tokenIdx - len(entity.split())
                    entity = entityNormalize(entity, s, tokenIdx - len(entity.split()))
                    # 仅保留第一次出现且长度大于2的实体
                    # position 是否应该与每个实体真实位置对应?
                    if len(entity) > 2 and entity not in entity_list:
                        entity_list.append(entity)
                        position_list.append(position)
                        leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
                entity = ''
            if label == 1 or label == 3:
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
                print('标签错误！跳过')  # 154次出现
        else:
            if prex == 3 or prex == 4:
                entity += word + ' '
                prex = label
            else:
                print('标签错误！跳过2')  #

    if not entity == '':
        print('!!!!!!!!!!!!!!')
        splited = entity.split('/')
        if len(splited) == 2:
            if not splited[1].strip().isdigit():
                entity = [splited[0].strip(), splited[1].strip()]
                if len(entity[0]) > 2 and entity[0] not in entity_list:
                    entity_list.append(entity[0])
                    position = tokenIdx - 3
                    position_list.append(position)
                    leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
                if len(entity[1]) > 2 and entity[1] not in entity_list:
                    entity_list.append(entity[1])
                    position = tokenIdx - 1
                    position_list.append(position)
                    leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
            else:
                position = tokenIdx - len(entity.split())
                entity = entityNormalize(entity, s, tokenIdx - len(entity.split()))
                if len(entity) > 2 and entity not in entity_list:
                    entity_list.append(entity)
                    position_list.append(position)
                    leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
        else:
            position = tokenIdx - len(entity.split())
            entity = entityNormalize(entity, s, tokenIdx - len(entity.split()))
            if len(entity) > 2 and entity not in entity_list:
                entity_list.append(entity)
                position_list.append(position)
                leixing_list.append('protein' if prex == 1 or prex == 2 else 'gene')
        entity = ''
        prex = label

    # 多个词组成的实体中，单个组成词也可能是实体(F值差别不大)
    return entity_list, position_list, leixing_list


def searchEntityId(s, predLabels, entity_tag_consisteny, lx_tag_consisteny, entity2id, protein2id, gene2id, text_byte):
    '''
    对识别的实体进行ID链接：

    先是词典精确匹配
    然后是知识库API匹配
    最后是模糊匹配
    '''
    entity_list, position_list, leixing_list = getEntityList(s, predLabels)
    assert len(entity_list) == len(position_list) == len(leixing_list)

    # # 标签一致性
    # entity_list2, position_list2, leixing_list2 = entity_tag_consisteny, [], lx_tag_consisteny
    # assert len(entity_list2) == len(leixing_list2)
    #
    # # 列表合并
    # for i in range(len(entity_list2)):
    #     if len(entity_list2[i])>=5 and entity_list2[i] not in entity_list and entity_list2[i].encode('utf-8') in text_byte:
    #         entity_list.append(entity_list2[i])
    #         leixing_list.append(leixing_list2[i])

    # entity_list2, position_list2, leixing_list2 = entity_list.copy(), position_list.copy(), leixing_list.copy()

    id_dict = {}
    for i in range(len(entity_list)):
        leixing = leixing_list[i]
        entity = entity_list[i]

        # 修正实体格式
        entity1 = entity
        if entity.encode('utf-8') not in text_byte:
            entity1 = entity.replace(' ', '')
            if entity1.encode('utf-8') not in text_byte:
                entity1 = entity
                for punc in string.punctuation:
                    if punc in entity1:
                        entity1 = entity1.replace(punc, punc + ' ')
                entity1 = entity1.replace('  ', ' ')
                if entity1.encode('utf-8') not in text_byte:
                    entity1 = entity
                    for punc in string.punctuation:
                        if punc in entity1:
                            entity1 = entity1.replace(punc, ' ' + punc + ' ')
                    entity1 = entity1.replace('  ', ' ')
        entity = entity1
        entity_list[i] = entity1

        # 实体拓展
        entity_variants1, entity_variants2, entity_variants3 = strippingAlgorithm(entity)
        query_list = [entity, entity_variants1]
        # query_list = [entity, entity_variants1, entity_variants2]

        # 词典精确匹配
        if entity_variants1 in entity2id:
            Ids = entity2id[entity_variants1]
            id_dict[entity] = Ids
            # if leixing == 'gene':
            #     id_dict[entity] = [Ids[k] for k in range(len(Ids)) if
            #                        Ids[k].startswith('gene') or Ids[k].startswith('NCBI')]
            # elif leixing == 'protein':
            #     id_dict[entity] = [Ids[k] for k in range(len(Ids)) if
            #                        Ids[k].startswith('protein') or Ids[k].startswith('Uniprot')]
            # if not id_dict[entity]:
            #     id_dict[entity] = Ids

            # # 词典匹配的结果中可能未包含正确答案
            # if not len(Ids) >= 2:
            #     if leixing == 'protein':
            #         # Uniprot 数据库API查询-reviewed
            #         Ids = search_id_from_Uniprot(query_list, reviewed=True)
            #         if Ids:
            #             id_dict[entity].extend(Ids)
            #             entity2id[entity_variants1].extend(Ids)
            #         else:
            #             Ids = search_id_from_Uniprot(query_list, reviewed=False)
            #             id_dict[entity].extend(Ids)
            #             entity2id[entity_variants1].extend(Ids)
            #     else:
            #         Ids = search_id_from_NCBI(query_list)
            #         if Ids:
            #             id_dict[entity].extend(Ids)
            #             entity2id[entity_variants1].extend(Ids)
            continue

        # # 知识库精确匹配（先忽略类型 leixing）
        # if leixing == 'protein':
        #     if entity_variants1 in protein2id or entity in protein2id:
        #         print('Uniprot 知识库精确匹配')
        #         Ids = protein2id[entity_variants1]
        #         id_dict[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]
        #         entity2id[entity_variants1] = id_dict[entity]
        #         # Ids2 = search_id_from_Uniprot(query_list, reviewed=True)
        #         # print(entity)
        #         # print(id_dict[entity])
        #         # print(Ids2)
        #     if entity not in id_dict:
        #         print('Uniprot 数据库API查询')
        #         Ids = search_id_from_Uniprot(query_list, reviewed=False)
        #         if Ids:
        #             id_dict[entity] = Ids
        #             entity2id[entity_variants1] = Ids
        # elif leixing == 'gene':
        #     if entity_variants1 in gene2id or entity in gene2id:
        #         print('NCBI gene 知识库精确匹配')
        #         Ids = gene2id[entity_variants1]
        #         id_dict[entity] = ['NCBI gene:' + Ids[i] for i in range(len(Ids))]
        #         entity2id[entity_variants1] = id_dict[entity]
        #         # Ids2 = search_id_from_NCBI(query_list)
        #         # print(entity)
        #         # print(id_dict[entity])
        #         # print(Ids2)


        if entity not in id_dict:
            if leixing == 'protein':
                # Uniprot 数据库API查询-reviewed
                Ids = search_id_from_Uniprot(query_list, reviewed=True)
                if Ids:
                    id_dict[entity] = Ids
                    entity2id[entity_variants1] = Ids
                else:
                    Ids = search_id_from_Uniprot(query_list, reviewed=False)
                    if Ids:
                        id_dict[entity] = Ids
                        entity2id[entity_variants1] = Ids
                    else:
                        Ids = search_id_from_NCBI(query_list)
                        id_dict[entity] = Ids
                        entity2id[entity_variants1] = Ids

                    # else:
                    #     # 模糊匹配--计算 Jaro–Winkler 距离
                    #     max_score = -1
                    #     max_score_key = ''
                    #     for key, value in entity2id.items():
                    #         score = Levenshtein.jaro_winkler(key, entity.lower())
                    #         if score > max_score:
                    #             max_score = score
                    #             max_score_key = key
                    #     Ids = entity2id.get(max_score_key)
                    #     id_dict[entity] = [Ids[k] for k in range(len(Ids)) if
                    #                        Ids[k].startswith('protein') or Ids[k].startswith('Uniprot')]
                    #     entity2id[entity_variants1] = id_dict[entity]

                # # Uniprot 数据库API查询-reviewed
                # for query in [entity, entity_variants1, entity_variants2, entity_variants3]:
                #     res_reviewed = u.search(query + '+reviewed:yes', frmt="tab", columns="id", limit=5)
                #     if isinstance(res_reviewed, int):
                #         print('请求无效\n')
                #     elif res_reviewed:  # 若是有返回结果
                #         Ids = extract_id_from_res(res_reviewed)
                #         id_dict[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
                #         entity2id[entity_variants1] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 将未登录实体添加到实体ID词典中
                #         break
                #
                # if entity not in id_dict:
                #     # Uniprot 数据库API查询-unreviewed
                #     for query in [entity, entity_variants1, entity_variants2, entity_variants3]:
                #         unres_reviewed = u.search(query, frmt="tab", columns="id", limit=5)
                #         if isinstance(unres_reviewed, int):
                #             print('请求无效\n')
                #         elif unres_reviewed:  # 若是有返回结果
                #             Ids = extract_id_from_res(unres_reviewed)
                #             id_dict[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
                #             entity2id[entity_variants1] = ['Uniprot:' + Ids[i] for i in
                #                                            range(len(Ids))]  # 将未登录实体添加到实体ID词典中
                #             break
                #
                # # 若在protein数据库中未找到，换gene数据库
                # if entity not in id_dict:
                #     # NCBI-gene数据库API查询
                #     for query in [entity, entity_variants1, entity_variants2, entity_variants3]:
                #         try:
                #             handle = Entrez.esearch(db="gene", idtype="acc", sort='relevance', term=query)
                #             record = Entrez.read(handle)
                #         except RuntimeError as e:
                #             print(e)
                #             continue
                #         except URLError as e:
                #             print(e)
                #             continue
                #         if record["IdList"]:
                #             id_dict[entity] = ['NCBI gene:' + record["IdList"][i] for i in
                #                                range(len(record["IdList"][:5]))]
                #             entity2id[entity_variants1] = id_dict[entity]
                #             break
            else:
                # NCBI-gene数据库API查询
                Ids = search_id_from_NCBI(query_list)
                if Ids:
                    id_dict[entity] = Ids
                    entity2id[entity_variants1] = Ids
                else:
                    Ids = search_id_from_Uniprot(query_list, reviewed=True)
                    if Ids:
                        id_dict[entity] = Ids
                        entity2id[entity_variants1] = Ids
                    else:
                        Ids = search_id_from_Uniprot(query_list, reviewed=False)
                        id_dict[entity] = Ids
                        entity2id[entity_variants1] = Ids

                    # else:
                    #     # 模糊匹配--计算 Jaro–Winkler 距离
                    #     max_score = -1
                    #     max_score_key = ''
                    #     for key in entity2id.keys():
                    #         score = Levenshtein.jaro_winkler(key, entity.lower())
                    #         if score > max_score:
                    #             max_score = score
                    #             max_score_key = key
                    #     Ids = entity2id.get(max_score_key)
                    #     id_dict[entity] = [Ids[k] for k in range(len(Ids)) if
                    #                        Ids[k].startswith('gene') or Ids[k].startswith('NCBI')]
                    #     entity2id[entity_variants1] = id_dict[entity]

                # # NCBI-gene数据库API查询
                # # for query in [entity, entity_variants1, entity_variants2, entity_variants3]:
                # for query in [entity, entity_variants1]:
                #     try:
                #         handle = Entrez.esearch(db="gene", idtype="acc", sort='relevance', term=query)
                #         record = Entrez.read(handle)
                #     except RuntimeError as e:
                #         print(e)
                #         continue
                #     except URLError as e:
                #         print(e)
                #         continue
                #     if record["IdList"]:
                #         id_dict[entity] = ['NCBI gene:' + record["IdList"][i] for i in
                #                            range(len(record["IdList"][:5]))]
                #         entity2id[entity_variants1] = id_dict[entity]
                #         break
                #
                # if entity not in id_dict:
                #     # Uniprot 数据库API查询-reviewed
                #     for query in [entity, entity_variants1]:
                #         res_reviewed = u.search(query + '+reviewed:yes', frmt="tab", columns="id", limit=5)
                #         if isinstance(res_reviewed, int):
                #             print('请求无效\n')
                #         elif res_reviewed:  # 若是有返回结果
                #             Ids = extract_id_from_res(res_reviewed)
                #             id_dict[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
                #             entity2id[entity_variants1] = ['Uniprot:' + Ids[i] for i in
                #                                            range(len(Ids))]  # 将未登录实体添加到实体ID词典中
                #             break
                #
                #     if entity not in id_dict:
                #         # Uniprot 数据库API查询-unreviewed
                #         for query in [entity, entity_variants1]:
                #             unres_reviewed = u.search(query, frmt="tab", columns="id", limit=5)
                #             if isinstance(unres_reviewed, int):
                #                 print('请求无效\n')
                #             elif unres_reviewed:  # 若是有返回结果
                #                 Ids = extract_id_from_res(unres_reviewed)
                #                 id_dict[entity] = ['Uniprot:' + Ids[i] for i in range(len(Ids))]  # 取第一个结果作为ID
                #                 entity2id[entity_variants1] = ['Uniprot:' + Ids[i] for i in
                #                                                range(len(Ids))]  # 将未登录实体添加到实体ID词典中
                #                 break

        if entity not in id_dict:
            print('未找到{}的ID，空'.format(entity))  # 152次出现
            id_dict[entity] = []

            # # 是否应该丢弃没找到ID的实体？？
            # index = entity_list2.index(entity)
            # entity_list2.pop(index)
            # position_list2.pop(index)
            # leixing_list2.pop(index)

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

    return entity_list, id_dict, position_list, leixing_list


def entity_disambiguation(entity_id, Id2synvec, zhixin):
    '''
    计算质心与实体向量（同义词集）的cos相似度进行消歧
    :param entity_id:
    :param Id2synvec:
    :param zhixin:
    :return:
    '''
    score = 0.65  # cos相似度的阙值
    type_id = ''
    for i in range(len(entity_id)):  # 计算质心与实体每个歧义（同义词集）的cos相似度
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


def entity_disambiguation_svm(entity, entity_id, id2synvec, zhixin, svm):
    type_id = None
    # 若实体的id中存在entity=id的情况，则仅标注其实体类型
    if 'protein:' + entity in entity_id:
        print('仅标注其实体类型')
        type_id = 'protein:' + entity
    elif 'gene:' + entity in entity_id:
        print('仅标注其实体类型')
        type_id = 'gene:' + entity
    else:
        fea_list = []
        for id in entity_id:
            # pos, local_collocations = pos_surround(test_x[idx_line], test_pos[idx_line], position, entity, idx2pos, features_dict)
            if '|' in id:
                id = id.split('|')
                id = '|'.join([item.split(':')[1] for item in id])
            else:
                id = id.split(':')[1]
            if id in id2synvec:
                syn_vec = id2synvec.get(id)
            else:
                print('未找到{}的实体向量，{}随机初始化.'.format(entity, id))
                syn_vec = np.round(np.random.uniform(-0.1, 0.1, 200), 6)

            s_profuction = list(np.multiply(zhixin, syn_vec))
            fea_list.append(s_profuction)
            # fea_list.append(pos + local_collocations + s_profuction)

        fea_list = np.array(fea_list)
        # 标准化
        # scaler = StandardScaler()
        # scaler.fit(fea_list)   # Expected 2D array
        # fea_list = scaler.transform(fea_list)

        # 样本距离超平面的距离
        result = list(svm.decision_function(fea_list))
        assert len(result) == len(entity_id)
        max_dis = -1000
        for i in range(len(result)):
            if result[i] > max_dis:
                max_dis = result[i]
                type_id = entity_id[i]
    return type_id


def entity_disambiguation_cnn(entity, entity_id, cnn, x_sen, s, pos_sen, x_id_dict, position, stop_word, entity2id_one, leixing):
    type_id = ''
    context_window_size = 10
    # 若实体的id中存在entity=id的情况，则仅标注其实体类型
    if leixing == 'protein':
        if entity_id.count('protein:' + entity) > 1:
            print('仅标注实体类型-protein')
            type_id = 'protein:' + entity
            return type_id
    else:
        if entity_id.count('gene:' + entity) > 1:
            print('仅标注实体类型-gene')
            type_id = 'gene:' + entity
            return type_id

    if entity in entity2id_one:
        print('entity in entity2id_one')
        type_id = entity2id_one[entity]
    else:
        fea_list = {'x_left': [], 'x_right': [], 'x_left_pos': [], 'x_right_pos': [], 'x_id': []}

        num = 0
        end_l = position
        x_sen_left = []
        pos_sen_left = []
        while num < context_window_size:
            end_l -= 1
            if end_l > 0:
                # 过滤停用词 stop_word
                if x_sen[end_l] not in stop_word:
                    x_sen_left.append(x_sen[end_l])
                    pos_sen_left.append(pos_sen[end_l])
                    num += 1
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

        assert len(x_sen_left) == len(x_sen_right)

        for id in entity_id:
            if '|' in id:
                id = id.split('|')
                id = '|'.join([item.split(':')[1] for item in id])
            else:
                id = id.split(':')[1]

            if id in x_id_dict:
                x_id = x_id_dict[id]
            else:   # 丢弃此候选ID
                x_id = 0
                # x_id_dict[id] = len(x_id_dict) + 1
                # print('{} not in x_id_dict'.format(id))
                continue

            fea_list['x_left'].append(x_sen_left)
            fea_list['x_right'].append(x_sen_right)
            fea_list['x_left_pos'].append(pos_sen_left)
            fea_list['x_right_pos'].append(pos_sen_right)
            fea_list['x_id'].append([x_id])

        if not fea_list['x_id']:
            # 说明候选ID集合仅包含实体类型√
            print(entity_id)
            if leixing == 'protein':
                type_id = 'protein:' + entity
            else:
                type_id = 'gene:' + entity
            return type_id

        for key, value in fea_list.items():
            fea_list[key] = np.array(fea_list[key])

        dataSet = [fea_list['x_id'], fea_list['x_left'], fea_list['x_right'], fea_list['x_left_pos'],
                   fea_list['x_right_pos']]

        predictions = cnn.predict(dataSet)
        max_dis = 0.5     # 是否需要设定一个阙值
        for i in range(len(predictions)):
            # if i==0:
            #     predictions[i][1] = predictions[i][1]*5
            if predictions[i][1] > max_dis:
                max_dis = predictions[i][1]
                type_id = entity_id[i]
        if type_id:
            entity2id_one[entity] = type_id
            return type_id
        else:
            # 若候选id的预测概率均小于阙值
            # 1 返回实体类型√；2 取第一个候选作为结果
            if leixing == 'protein':
                type_id = 'protein:' + entity
            else:
                type_id = 'gene:' + entity
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
    exit = 0
    not_exit = 0
    not_find = 0
    num_match = 0
    idx_line = -1
    num_entity_no_id = 0
    words_with_multiId = []
    base = r'/home/administrator/桌面/BC6_Track1'
    BioC_path = base + '/' + 'test_corpus_20170804/caption_bioc'  # 测试数据文件夹
    dic_path = base + '/' + 'BioIDtraining_2/annotations.csv'  # 实体ID查找词典文件
    test_dic_path = base + '/' + 'test_corpus_20170804/annotations.csv'
    result_path = base + '/' + 'test_corpus_20170804/prediction'
    synsetsVec_path = '/home/administrator/PycharmProjects/embedding/data/synsetsVec.txt'

    # 读取golden实体的ID词典
    entity2id = getCSVData(dic_path)
    # entity2id_test = getCSVData(test_dic_path)
    # for key, value in entity2id_test.items():
    #     if key not in entity2id:
    #         entity2id[key]=value

    # 读取同义词集向量
    id2synvec = readSynVec(synsetsVec_path)
    # 停用词表/标点符号
    stop_word = [239, 153, 137, 300, 64, 947, 2309, 570, 10, 69, 238, 175, 852, 7017, 378, 136, 5022, 1116, 5194, 14048,
                 28, 217, 4759, 7359, 201, 671, 11, 603, 15, 1735, 2140, 390, 2366, 12, 649, 4, 1279, 3351, 3939, 5209, 16, 43,
                 2208, 8, 5702, 4976, 325, 891, 541, 1649, 17, 416, 2707, 108, 381, 678, 249, 5205, 914, 5180, 5, 20, 18695,
                 15593, 5597, 730, 1374, 18, 2901, 1440, 237, 150, 44, 10748, 549, 3707, 4325, 27, 331, 522, 10790, 297, 1060, 1976,
                 7803, 1150, 1189, 2566, 192, 5577, 703, 666, 315, 488, 89, 1103, 231, 16346, 9655, 6569, 605, 6, 294, 3932, 24965,
                 9, 775, 4593, 76, 21733, 140, 229, 16368, 21098, 181, 620, 134, 6032, 268, 2267, 22948, 88, 655, 24768, 6870,
                 25, 615, 4421, 99, 3, 375, 483, 7, 2661, 32, 2223, 42, 1612, 595, 22, 37, 432, 8439, 67, 15853, 6912, 459,
                 21441, 3811, 1538, 1644, 2834, 1192, 5197, 1734, 78, 647, 247, 491, 16228, 23, 578, 34, 47, 77, 1239, 846, 26,
                 24317, 785, 3601, 8504, 29, 9414, 520, 3399, 2035, 6778, 96, 2048, 1, 579, 1135, 173, 4089, 4980, 205, 63, 516, 169,
                 8413, 1980, 337, 19, 521, 13, 48, 551, 3927, 59, 10281, 11926, 3915]
    # 读取测试预料的数据和golden ID
    sen_list = get_test_out_data(path)

    # 读取词向量词典
    # with open(r'/home/administrator/PycharmProjects/embedding/emb.pkl', "rb") as f:
    #     embedding_matrix = pkl.load(f)

    with open(r'/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test.pkl', "rb") as f:
        test_x, test_y, test_char, test_cap, test_pos, test_chunk, test_dict = pkl.load(f)
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/data/x_id_dict.pkl', 'rb') as f:
        x_id_dict = pkl.load(f)

    with open('/home/administrator/PycharmProjects/embedding/length.pkl', "rb") as f:
        word_maxlen, sentence_maxlen = pkl.load(f)

    # # 实体标注一致性
    # idx_line2pmc_id, pmc_id2entity_list, pmc_id2lx_list = post_process(sen_list, BioC_path, sentence_maxlen, predLabels)

    # 加载实体消歧模型
    cnn = load_model('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/data/weights2.hdf5')

    synId2entity = {}
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/data/synId2entity.txt') as f:
        for line in f:
            s1, s2 = line.split('\t')
            entities = s2.replace('\n', '').split('::,')
            synId2entity[s1] = entities

    idx2pos = {}
    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/pos2idx.txt') as f:
        for line in f:
            pos, idx = line.split('\t')
            idx2pos[idx.strip('\n')] = pos


    # 获取知识库中的实体及其ID，用于精确匹配
    protein2id, gene2id = {}, {}
    # with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/pg2id.pkl', 'rb') as f:
    #     protein2id, gene2id = pkl.load(f)


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

                # 一篇文档中的相同实体理应具有县相同的ID?
                entity2id_one = {}

                passages = doc.getElementsByTagName("passage")
                for passage in passages:
                    text = passage.getElementsByTagName('text')[0].childNodes[0].data
                    text_byte = text.encode('utf-8')
                    annotations = passage.getElementsByTagName('annotation')
                    entity2golden = {}
                    for annotation in annotations:
                        info = annotation.getElementsByTagName("infon")[0]
                        ID = info.childNodes[0].data
                        txt = annotation.getElementsByTagName("text")[0]
                        entity = txt.childNodes[0].data
                        if ID.startswith('gene') or ID.startswith('protein') or ID.startswith(
                                'Uniprot') or ID.startswith('NCBI'):
                            a = strippingAlgorithm(entity)[0]
                            entity2golden[a] = ID

                    '''每读取一篇passage，在<annotation>结点记录识别实体'''
                    idx_line += 1
                    annotation_list = []
                    s = sen_list[idx_line][:sentence_maxlen]  # 单词列表形成的句子
                    prediction = predLabels[idx_line]

                    '''根据预测结果来抽取句子中的所有实体，并进行实体链接
                    修正实体格式, 获得实体标注一致性后的所有实体集'''
                    entity_tag_consisteny = []
                    lx_tag_consisteny = []
                    # pmc_id = idx_line2pmc_id[str(idx_line)]
                    # entity_mul = pmc_id2entity_list[pmc_id]
                    # lx_mul = pmc_id2lx_list[pmc_id]
                    # assert len(entity_mul)==len(lx_mul)
                    # for i in range(len(entity_mul)):
                    #     entity = entity_mul[i]
                    #     lx = lx_mul[i]
                    #     entity1 = entity
                    #     entity_tag_consisteny.append(entity1)
                    #     lx_tag_consisteny.append(lx)

                    # 根据预测结果来抽取句子中的所有实体，并进行实体链接
                    entity_list, id_dict, post_list, lx_list = searchEntityId(s, prediction, entity_tag_consisteny, lx_tag_consisteny,
                                                                              entity2id, protein2id, gene2id, text_byte)

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

                            if id not in x_id_dict:
                                x_id_dict[id] = len(x_id_dict) + 1


                    ''' 针对多ID的实体进行实体消岐'''
                    annotation_id = 0
                    # for entity, tokenIdx in entities.items():
                    for i in range(len(entity_list)):
                        entity = entity_list[i]
                        position = post_list[i]
                        leixing = lx_list[i]
                        entity_id = id_dict[entity]

                        if len(entity_id) > 1:
                            if entity not in words_with_multiId:
                                words_with_multiId.append(entity)

                            # First
                            # type_id = entity_id[0]

                            # cos/svm/arnn
                            # type_id = entity_disambiguation(entity_id, id2synvec, zhixin)
                            # type_id = entity_disambiguation_svm(entity, entity_id, id2synvec, zhixin, svm)
                            type_id = entity_disambiguation_cnn(entity, entity_id, cnn, test_x[idx_line], s, test_pos[idx_line], x_id_dict, position, stop_word, entity2id_one, leixing)

                        elif len(entity_id) == 1:  # 说明实体对应了唯一ID
                            type_id = entity_id[0]
                        else:
                            # print('未找到{}的ID，仅标注其实体类型'.format(entity))
                            if leixing == 'protein':
                                type_id = 'protein:' + entity
                            else:
                                type_id = 'gene:' + entity
                            num_entity_no_id += 1


                        # 统计覆盖率
                        a = strippingAlgorithm(entity)[0]
                        goldenID = entity2golden.get(a)
                        if goldenID:
                            if goldenID in entity_id:
                                exit += 1
                                if goldenID == type_id:
                                    num_match += 1
                            else:
                                not_exit += 1
                        else:
                            not_find += 1  # 实体未找到


                        '''给句子中所有相同的实体标记上找到的ID'''

                        if entity.encode('utf-8') in text_byte:
                            entity_byte = entity.encode('utf-8')
                        else:
                            entity1 = entity.replace(' ', '')
                            if entity1.encode('utf-8') in text_byte:
                                entity_byte = entity1.encode('utf-8')
                            else:
                                entity2 = entity.replace(',', ', ').replace('.', '. ')
                                if entity2.encode('utf-8') in text_byte:
                                    entity_byte = entity2.encode('utf-8')
                                else:
                                    for punc in string.punctuation:
                                        if punc in entity:
                                            entity = entity.replace(punc, ' ' + punc + ' ')
                                    entity = entity.replace('  ', ' ')
                                    entity_byte = entity.encode('utf-8')
                                    print('未在句子中找到{}的offset索引？'.format(entity))
                        offset = -1
                        while 1:
                            offset = text_byte.find(entity_byte, offset + 1)  # 二进制编码查找offset
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
            Indent(dom, dom.documentElement)  # 美化
            outputName = result_path + '/' + file
            f = open(outputName, 'w')
            writer = codecs.lookup('utf-8')[3](f)
            dom.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
            writer.close()
            f.close()

    print('exit:{}, not_exit:{}'.format(exit, not_exit))  # exit:5809, not_exit:2763
    print('num_match:{}, not_find:{}'.format(num_match, not_find))  # num_match:3811, not_find:1744
    print('测试集预测结果写入成功！')
    print('{}个词未找到对应的ID'.format(num_entity_no_id))  # 214
    print('{}个词有歧义'.format(len(words_with_multiId)))  # 1673
    print('完结撒花')

    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/ned/data/x_id_dict2.pkl', 'wb') as f:
        pkl.dump((x_id_dict), f, -1)

    with open('synsets.txt', "w") as f:
        for key, value in synId2entity.items():
            f.write('{}\t{}'.format(key, '::,'.join(value)))
            f.write('\n')
