'''
抽取实体特征训练SVM

'''
import codecs
import pickle as pkl
import string

import numpy as np
from bioservices import UniProt
from sample.writeXMLResult import readSynVec, get_w2v, getCSVData

from sample.utils.helpers import cos_sim


def searchEntityId(entity, entity2id):
    ''' 对识别的实体进行ID链接 '''
    u = UniProt()
    Ids = []
    id_list = {}

    # 词典精确匹配1
    if entity.lower() in entity2id:
        Ids = entity2id[entity.lower()]
        # Ids = idFilter(type, Ids)     # 不进行筛选，否则ID几乎全被干掉了
        id_list[entity] = Ids
        return id_list

    temp = entity
    for char in string.punctuation:
        if char in temp:
            temp = temp.replace(char, '')

    # 词典精确匹配2
    if temp.lower() in entity2id:
        Ids = entity2id[temp.lower()]
        # Ids = idFilter(type, Ids)
        id_list[entity] = Ids
        return id_list

    # 数据库API查询
    res = u.search(entity + '+reviewed:yes', frmt="tab", columns="genes, id", limit=3)
    if res:  # 若是有返回结果
        results = res.split('\n')[1:-1]  # 去除开头一行和最后的''
        for line in results:
            Id = line.split('\t')[-1]
            Ids.append(Id)
            break
        id_list[entity] = Ids
        entity2id[entity.lower()] = Ids  # 将未登录实体添加到实体ID词典中
        return id_list

    # 数据库API查询2
    res = u.search(temp + '+reviewed:yes', frmt="tab", columns="genes, id", limit=3)
    if res:
        results = res.split('\n')[1:-1]  # 去除开头一行和最后的''
        for line in results:
            Id = line.split('\t')[-1]
            Ids.append(Id)
            break
        id_list[entity] = Ids
        entity2id[entity.lower()] = Ids  # 将未登录实体添加到实体ID词典中
        return id_list

    return []


geneId2vec, proteinId2vec, stop_word = readSynVec()
base = r'/home/administrator/桌面/BC6_Track1'
dic_path = base + '/' + 'BioIDtraining_2/annotations.csv'   # 实体ID查找词典文件
entity2id = getCSVData(dic_path)    # 读取实体ID查找词典


def getFeatures(entity, goldenID, zhixin, entity2id, method='S-product'):
    '''
    # Elementwise product; both produce the array
    print(np.multiply(x, y))    # x * y
    # Inner product of vectors; both produce 219
    print(np.dot(v, w))
    '''
    if method=='S-cosine':
        func = lambda x1,x2: cos_sim(x1, x2)
    elif method=='S-product':
        func = lambda x1,x2: np.multiply(x1, x2)
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
        print(temp)
        syn_vec = geneId2vec.get(goldenID1)
    elif goldenID1 in proteinId2vec:
        print(temp)
        syn_vec = proteinId2vec.get(goldenID1)
    else:
        print('未找到golden对应同义词集向量，随机初始化')
        syn_vec = np.random.uniform(-0.1, 0.1, 200)
    x.append(list(func(zhixin, syn_vec)))
    y.append(1)

    candidtaIds = searchEntityId(entity, entity2id)
    candidtaIds.remove(temp) if temp in candidtaIds else candidtaIds
    for candidta in candidtaIds:
        candidta = candidta.split('|')[0] if '|' in candidta else candidta
        if ':' in candidta:
            print(candidta)
            candidta = candidta.split(':')[1]
        if candidta in geneId2vec:
            syn_vec = geneId2vec.get(candidta)
        elif candidta in proteinId2vec:
            syn_vec = proteinId2vec.get(candidta)
        else:
            print('未找到对应同义词集向量，随机初始化')
            syn_vec = np.random.uniform(-0.1, 0.1, 200)
        x.append(list(func(zhixin, syn_vec)))
        y.append(0)
    return x, y


def getSen(path):
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
    return sen_list


def getData():
    from xml.dom.minidom import parse
    import os
    from tqdm import tqdm

    x = []
    y = []
    idx_line = -1
    with codecs.open('../data/train.pkl', "rb") as f:
        train_x, train_y, train_char, train_cap, train_pos, train_chunk = pkl.load(f)
    embeddingPath = r'/home/administrator/PycharmProjects/embedding'
    with open(embeddingPath + '/emb.pkl', "rb") as f:
        embedding_matrix, word_maxlen, sentence_maxlen = pkl.load(f)
    sen_list = getSen('../data/train.out.txt')
    w2v = get_w2v()

    BioC_path = '/home/administrator/桌面/BC6_Track1/BioIDtraining_2/caption_bioc'
    files = os.listdir(BioC_path)
    files.sort()
    for j in tqdm(range(len(files))):  # 遍历文件夹
        file = files[j]
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = BioC_path + "/" + file
            DOMTree = parse(f)  # 使用minidom解析器打开 XML 文档
            collection = DOMTree.documentElement  # 得到了根元素对象

            source = collection.getElementsByTagName("source")[0].childNodes[0].data
            date = collection.getElementsByTagName("date")[0].childNodes[0].data    # 时间
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
                if not len(passages)==1:
                    print('not 1')
                for passage in passages:
                    idx_line += 1
                    text = passage.getElementsByTagName('text')[0].childNodes[0].data
                    zhixin = np.zeros(200)  # 计算质心向量
                    for wordId in train_x[idx_line]:
                        # if word not in stop_word:
                        vector = embedding_matrix[wordId]
                        if vector is None:
                            vector = np.random.uniform(-0.1, 0.1, 200)
                        zhixin += vector

                    annotations = passage.getElementsByTagName('annotation')
                    for annotation in annotations:
                        info = annotation.getElementsByTagName("infon")[0]
                        ID = info.childNodes[0].data
                        location = annotation.getElementsByTagName("location")[0]
                        offset = location.getAttribute("offset")
                        length = location.getAttribute("length")
                        annotation_txt = annotation.getElementsByTagName("text")[0]
                        entity = annotation_txt.childNodes[0].data
                        if len(entity) == 1:
                            continue
                        if ID.startswith('Uniprot:') or ID.startswith('protein:') or \
                            ID.startswith('NCBI gene:') or ID.startswith('gene:'):
                            # for char in string.punctuation:
                            #     if char in entity:
                            #         entity = entity.replace(char, ' ' + char + ' ')
                            # print(text)

                            # idx = text.split().index(entity.split()[0])
                            # tmp = text.split()[idx - 5 if idx >= 5 else 0:idx + 5 + len(entity.split())]
                            # surroundding_word = []
                            # for w in tmp:
                            #     surroundding_word.append(w2v.get(w) if w2v.get(w) is not None
                            #                              else np.random.uniform(-0.1, 0.1, 200))
                            try:
                                idx = sen_list[idx_line].index(entity.split()[0])
                                surroundding_word = train_x[idx_line][
                                                    idx - 5 if idx >= 5 else 0:idx + 5 + len(entity.split())]
                            except:
                                idx = text.split().index(entity.split()[0])
                                surroundding_word = train_x[idx_line][
                                                    idx - 5 if idx >= 5 else 0:idx + 5 + len(entity.split())]
                            # pos = train_pos[idx_line][idx]

                            if 15 - len(surroundding_word) > 0:
                                surroundding_word.extend([0]*(10-len(surroundding_word)))
                            if ID.startswith('Uniprot:') or ID.startswith('protein:'):
                                fea, label = getFeatures(entity, ID, zhixin, entity2id)
                                x.extend([surroundding_word] + fea)
                                # x.extend([pos, surroundding_word] + fea)
                                y.extend(label)
                            elif ID.startswith('NCBI gene:') or ID.startswith('gene:'):
                                fea, label = getFeatures(entity, ID, zhixin, entity2id)
                                x.extend([surroundding_word] + fea)
                                # x.extend([pos, surroundding_word] + fea)
                                y.extend(label)
                            elif ID.startswith('Rfam:'):
                                print('过，不要')
                                pass

    print(len(x), len(y))
    return x, y


from sklearn import svm
x, y = getData()
clf = svm.SVC(kernel='linear')  # polynomial
clf.fit(x, y)

#保存Model(注:save文件夹要预先建立，否则会报错)
with open('save/clf.pickle', 'wb') as f:
    pkl.dump(clf, f)

# #读取Model
# with open('save/clf.pickle', 'rb') as f:
#     clf2 = pkl.load(f)
#     #测试读取后的Model
#     print(clf2.predict([[2., 2.]]))