import datetime
import xml.dom.minidom
import Levenshtein  # pip install python-Levenshtein
from tqdm import tqdm
from helpers import makeEasyTag, Indent, convert_2_BIO, entityNormalize, cos_sim, idFilter, idFilter2
import numpy as np
import os
import csv
import pickle as pkl
from xml.dom.minidom import parse
import xml.dom.minidom
import codecs
import string
from bioservices import UniProt
u = UniProt()


"""
将实体预测结果，以特定格式写入XML文件，用于scorer进行评估
<collection>
    <source>SourceData</source>
    <date>00000000</date>
    <key>sourcedata.key</key>
    <document>
        <id>1-A</id>
        <infon key="sourcedata_document">?</infon>
        <infon key="doi">?</infon>
        <infon key="pmc_id">?</infon>
        <infon key="figure">?</infon>
        <infon key="sourcedata_figure_dir">?</infon>
        <passage>
            <offset>0</offset>
            <text>XXX</text>
            <annotation id="1">
                <infon key="type">GO:0005764</infon>
                <infon key="sourcedata_figure_annot_id">1</infon>
                <infon key="sourcedata_article_annot_id">1</infon>
                <location offset="16" length="9"/>
                <text>lysosomes</text>
            </annotation>
        </passage>
    </document>
</collection>
"""


synsets_path = '/home/administrator/PycharmProjects/embedding/AutoExtend/synsets.txt'
synsets_code_path = '/home/administrator/PycharmProjects/embedding/AutoExtend/synsets_code.txt'
lexemes_path = '/home/administrator/PycharmProjects/embedding/AutoExtend/lexemes.txt'

synset2id = {}
with open(synsets_code_path, 'r') as f:
    for line in f:
        splited = line.strip().split('\t')
        synset_ncbi_id = splited[0]
        words = splited[1]
        words = words.replace('@@', ' ')
        synset2id[words] = synset_ncbi_id
# print(synset2id['pMF1.22::,pMX1.22::,'])


id2vec = {}
with open(synsets_path, 'r') as f:
    for line in f:
        splited = line.strip().split(' ')
        synset = splited[0].replace('@@', ' ')
        id = synset2id[synset]
        vec = np.asarray(splited[1:], dtype=np.float32)
        id2vec[id] = vec
# print(id2vec['22196'])
# print(id2vec['7329'])


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


# embedPath = r'/home/administrator/PycharmProjects/embedding'
# embedFile = r'wikipedia-pubmed-and-PMC-w2v.bin'
# word2vec = readBinEmbedFile(embedPath+'/'+embedFile, 200)
# with open('data/word2vec.pkl', "wb") as f:
#     pkl.dump(word2vec, f, -1)
with open('data/word2vec.pkl', "rb") as f:
    word2vec = pkl.load(f)


def getXlsxData(path):
    from openpyxl import load_workbook

    wb = load_workbook(path)  # 加载一个工作簿
    sheets = wb.get_sheet_names()  # 获取各个sheet的名字
    sheet0 = sheets[0]  # 第一个表格的名称
    ws = wb.get_sheet_by_name(sheet0)  # 获取特定的 worksheet

    # 获取表格所有行和列，两者都是可迭代的
    rows = ws.rows
    # columns = ws.columns

    # 行迭代
    content = []
    for row in rows:
        line = [col.value for col in row]
        content.append(line)


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

    for key, value in entity2id.items():
        if len(value)>1:
            num_word_multiID+=1
        entity2id_new[key] = value
        # 去掉实体中的标点符号
        for char in string.punctuation:
            if char in key:
                key = key.replace(char, '')
        if key not in entity2id_new:
            entity2id_new[key] = value
    entity2id = {}
    print('其中，多ID实体的个数：{}'.format(num_word_multiID))    # 1562
    return entity2id_new


def searchEntityId(s, predLabels, entity2id):
    '''
    抽取句子中的所有实体及其对应ID
    :param s: 单词列表形成的句子
    :param idx: 当前句子的预测结果id
    :param predLabels: 预测结果
    :param entity2id: 词典
    :return:
    '''
    entity_list = {}
    id_list = {}
    entity = ''
    prex = 0
    for tokenIdx in range(len(s)):
        label = predLabels[tokenIdx]
        word = s[tokenIdx]
        if label == 1:
            if entity:
                if prex==1 or prex==2:
                    entity_list[entityNormalize(entity.strip())]='gene'  # 实体标准化
                elif prex == 3 or prex == 4:
                    entity_list[entityNormalize(entity.strip())] = 'protein'  # 实体标准化
                entity = ''
            prex = label
            entity = word + ' '
        elif label == 2:
            if prex == 1:
                entity += word + ' '
            else:
                print('标签错误！')
        elif label == 3:
            if entity:
                if prex==1 or prex==2:
                    entity_list[entityNormalize(entity.strip())]='gene'  # 实体标准化
                elif prex == 3 or prex == 4:
                    entity_list[entityNormalize(entity.strip())] = 'protein'  # 实体标准化
                entity = ''
            prex = label
            entity = word + ' '
        elif label == 4:
            if prex == 3:
                entity += word + ' '
            else:
                print('标签错误！')
        else:
            if entity:
                if prex==1 or prex==2:
                    entity_list[entityNormalize(entity.strip())]='gene'  # 实体标准化
                elif prex == 3 or prex == 4:
                    entity_list[entityNormalize(entity.strip())] = 'protein'  # 实体标准化
                entity = ''
            else:
                entity = ''
            prex = 0
    if not entity == '':
        if prex == 1 or prex == 2:
            entity_list[entityNormalize(entity.strip())] = 'gene'  # 实体标准化
        elif prex == 3 or prex == 4:
            entity_list[entityNormalize(entity.strip())] = 'protein'  # 实体标准化
        entity = ''
    entities = entity_list
    # l2 = list(set(entity_list))  # 去除相同元素
    # entities = sorted(l2, key=entity_list.index)  # 不改变原list顺序
    # print(entities)

    # 多个词组成的实体中，单个组成词也可能是实体
    temp_entities = entities.copy()     # 字典的直接赋值和copy的区别（浅拷贝引用，深拷贝）
    for entity in temp_entities.keys():
        splited = entity.split(' ')
        if len(splited)>1:
            for e in splited:
                if e in entity2id and e not in entities:
                    entities[e]=entities[entity]
    temp_entities = None

    ''' 对识别的实体进行ID链接 '''
    for entity, type in entities.items():
        # 词典精确匹配1
        if entity.lower() in entity2id:
            Ids = entity2id[entity.lower()]
            # Ids = idFilter(type, Ids)     # ID几乎全被干掉了
            id_list[entity] = Ids
            continue

        temp = entity
        for char in string.punctuation:
            if char in temp:
                temp = temp.replace(char, '')

        # 词典精确匹配2
        if temp.lower() in entity2id:
            Ids = entity2id[temp.lower()]
            # Ids = idFilter(type, Ids)
            id_list[entity] = Ids
            continue

        # 数据库API查询
        res = u.search(entity + '+reviewed:yes', frmt="tab", columns="genes, id", limit=3)
        if res:     # 若是有返回结果
            Ids = idFilter2(res, type)
            id_list[entity] = Ids
            entity2id[entity.lower()] = Ids   # 将未登录实体添加到实体ID词典中
            continue

        # 数据库API查询2
        res = u.search(temp + '+reviewed:yes', frmt="tab", columns="genes, id", limit=3)
        if res:
            Ids = idFilter2(res, type)
            id_list[entity] = Ids
            entity2id[entity.lower()] = Ids  # 将未登录实体添加到实体ID词典中
            continue

        # 模糊匹配--计算 Jaro–Winkler 距离
        print('模糊匹配--计算 Jaro–Winkler 距离')
        max_score = -1
        max_score_key = ''
        for key in entity2id.keys():
            score = Levenshtein.jaro_winkler(key, entity.lower())
            if score > max_score:
                max_score = score
                max_score_key = key
        Ids = entity2id.get(max_score_key)
        # Ids = idFilter(type, Ids)
        id_list[entity] = Ids

    # if entity == 'Ubc9':
    #     print(entity2id.get(entity))
    return entities, id_list


def writeOutputToFile(path, predLabels, maxlen):
    '''
    将实体预测结果写入XML文件
    :param path: 测试数据路径
    :param predLabels: 测试数据的实体预测结果
    :param maxlen: 句子截断长度
    :param split_pos: 划分训练集和验证集的位置
    :return:
    '''

    # 读取实体 ID 词典
    csv_path = r'/home/administrator/桌面/BC6_Track1/BioIDtraining_2/annotations.csv'
    entity2id = getCSVData(csv_path)
    print('tau 的ID集合：{}'.format(entity2id['tau']))

    # 获取 train.out 所有句子的集合
    s = []
    sen_list = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                sen_list.append(s)  # ' '.join()
                s = []
            else:
                token = line.replace('\n', '').split('\t')
                word = token[0]
                s.append(word)

    '''
    按顺序读取文件夹中的xml格式文件
    同时，对应每个text生成annotation标签：
        getElementsByTagName方法：获取孩子标签
        getAttribute方法：可以获得元素的属性所对应的值。
        firstChild.data≈childNodes[0].data：返回被选节点的第一个子标签对之间的数据
    '''
    idx_line = -1
    num_entity_no_id = 0
    words_with_multiId = []
    BioC_PATH = r'/home/administrator/桌面/BC6_Track1/BioIDtraining_2/devel_115/gold'
    files = os.listdir(BioC_PATH)  # 得到文件夹下的所有文件名称
    files.sort()

    for j in tqdm(range(len(files))):  # 遍历文件夹
        file = files[j]
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = BioC_PATH + "/" + file
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
                    '''每读取一篇passage，在<annotation>结点记录识别实体'''
                    idx_line += 1
                    # print(idx_line)
                    annotation_list = []
                    s = sen_list[idx_line][:maxlen]  # 单词列表形成的句子
                    # sen = ' '.join(sen_list[idx_line][:maxlen])  # 字符串句子
                    prediction = predLabels[idx_line]

                    # 根据预测结果来抽取句子中的所有实体，并进行实体链接
                    entities, entity_ids = searchEntityId(s, prediction, entity2id)
                    # print((entities))
                    # print((entity_ids))

                    ''' 
                    多ID的实体需要进行实体消岐，AutoExtend 实体消歧方法：
                        1、计算实体所在句子的质心
                        2、计算质心与实体每个歧义（同义词集）的相似度
                        3、取最大得分作为ID
                    '''
                    annotation_id = 0
                    for entity, e_type in entities.items():
                        type_id = None
                        entity_id = entity_ids[entity]
                        if entity=='SUMO1':
                            print(entity_id)
                        if len(entity_id)>1:
                            if entity not in words_with_multiId:
                                words_with_multiId.append(entity)
                            if e_type == 'gene':
                                score = 0
                                zhixin = np.zeros(200)
                                for word in sen_list[idx_line]:
                                    vector = word2vec.get(word)
                                    if vector is None:
                                        vector = np.zeros(200)
                                    zhixin += vector

                                type_id = entity_id[0]
                                for id in entity_id:
                                    splited_id = id.split(':')[1]
                                    synset_vec = id2vec.get(splited_id)
                                    if synset_vec is not None:
                                        cos_sim_score = cos_sim(zhixin, synset_vec)
                                        if cos_sim_score > score:
                                            score = cos_sim_score
                                            type_id = id
                            else:
                                type_id = entity_id[0]
                        elif len(entity_id)==1:
                            type_id = entity_id[0]  # 说明实体对应了唯一ID
                        else:
                            type_id = e_type + ':' + entity     # 未找到对应的ID
                            num_entity_no_id += 1

                        # 标记句子中所有相同的实体
                        offset = -1
                        while 1:
                            offset = text.find(entity, offset+1)
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
                                location.setAttribute('length', str(len(entityNormalize(entity))))
                                text_node = makeEasyTag(dom, 'text', entityNormalize(entity))
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
            result_path = r'/home/administrator/桌面/BC6_Track1/BioIDtraining_2/devel_115/result'
            outputName = result_path + '/' + file
            f = open(outputName, 'w')
            writer = codecs.lookup('utf-8')[3](f)
            dom.writexml(f, indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')
            writer.close()
            f.close()

    print('{}个词未找到对应的ID'.format(num_entity_no_id))
    print('{}个词有歧义'.format(len(words_with_multiId)))    # 262
    print('完结撒花')


'''
python bioid_score.py --verbose 1 --force \
/home/administrator/桌面/BC6_Track1/BioID_scorer_1_0_3/data/bioid_scores \
/home/administrator/桌面/BC6_Track1/BioIDtraining_2/devel_115/gold \
/home/administrator/桌面/BC6_Track1/BioIDtraining_2/devel_115/result
'''