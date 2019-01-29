import re
import string
import sys
# from sample.ned.LocalCollocationsExtractor import LocalCollocationsExtractor
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

print(sys.getdefaultencoding())

SYMBOLS = {'}': '{', ']': '[', ')': '('}  # 符号表
SYMBOLS_L, SYMBOLS_R = SYMBOLS.values(), SYMBOLS.keys()


def get_stop_dic(stopWord_path):
    """获取停用词词典."""
    stop_word = []
    with open(stopWord_path, 'r') as f:
        for line in f:
            stop_word.append(line.strip('\n'))

    from nltk.corpus import stopwords
    stop_word.extend(stopwords.words('english'))
    stop_word = list(set(stop_word))
    return stop_word


def wordNormalize(word):
    '''
    对单词进行清洗,特殊符号归一化
    :param word:
    :return:
    '''
    word = word.strip().lower()
    word = re.sub(u'\s+', '', word, flags=re.U)  # 匹配任何空白字符
    word = word.replace("--", "-")
    word = re.sub("\"+", '"', word)

    if word.isdigit():
        word = '1'
    else:
        temp = word
        for char in word:
            if char not in string.printable:
                temp = temp.replace(char, '*')
        word = temp
    return word


def postprocess(entity, s, tokenIdx):
    '''
    实体-括号平衡
    实体中的标点符号周围的空格去掉
    '''
    entity = entity.strip()
    result = check(entity)
    idx = tokenIdx
    if result:
        if result == 'R':
            print('R括号多')
            while idx-1>=0 and s[idx - 1] not in SYMBOLS_L:
                entity = s[idx - 1] + ' ' + entity
                idx-=1
            if s[idx - 1] in SYMBOLS_L:
                entity = s[idx - 1] + ' ' + entity
                idx = idx - 1
        elif result == 'L':
            print('L括号多')
            while idx+1<len(s) and s[idx + 1] not in SYMBOLS_R:
                entity = entity + ' ' + s[idx + 1]
                idx+=1
            if s[idx + 1] in SYMBOLS_R:
                entity = entity + ' ' + s[idx + 1]
                idx = idx+1

    entity = entity.strip()
    for char in string.punctuation:
        if char in entity:
            entity = entity.replace(' '+char+' ', char)
            entity = entity.replace(char+' ', char)
            entity = entity.replace(' '+char, char)
    return entity, idx


def idFilter(type, Ids):
    '''
    将字典匹配或API匹配得到的Ids集合中，与type不符的部分去掉
    '''
    temp = []
    for Id in Ids:
        if type == 'protein':
            if 'uniprot' in Id or 'protein' in Id:
                temp.append(Id)
        elif type == 'gene':
            if 'NCBI' in Id or 'gene' in Id:
                temp.append(Id)
        else:
            print('??')
    return temp


def extract_id_from_res(res):
    '''
    从数据库API匹配的结果中抽取实体id
    :param res:
    :return:
    '''

    Ids = []
    results = res.split('\n')[1:-1]  # 去除开头一行和最后的''
    for line in results:
        id = line.split('\t')[0]
        Ids.append(id)
    return Ids

def extract_id_from_res2(res):
    '''
    从数据库API匹配的结果中抽取实体id和描述信息
    :param res:
    :return:
    '''

    Ids = []
    descriptions = []
    results = res.split('\n')[1:-1]  # 去除开头一行和最后的''
    for line in results:
        results = line.split('\t')
        Ids.append(results[0])
        descriptions.append(results[-1])
    return Ids, descriptions


def pos_surround(test_x, test_pos, tokenIdx, entity, idx2pos, features_dict):
    '''
    获取实体周围的窗口为3的上下文及其pos标记
    # 获取 Local Collocations 特征
    '''
    pos_list = ['null', 'n', 'v', 'a', 'r', 'other']
    MAP = [
        "n N NOUN NN NNP NNPS NE NNS NN|NNS NN|SYM NN|VBG NP N",
        "v V VERB MD VB VBD VBD|VBN VBG VBG|NN VBN VBP VBP|TO VBZ VP VVD VVZ VVN VVB VVG VV V",
        "a A ADJ JJ JJR JJRJR JJS JJ|RB JJ|VBG",
        "r R ADV RB RBR RBS RB|RP RB|VBG WRB R IN IN|RP"
        ]
    map_dict = {}
    for line in MAP:
        splited = line.split(' ')
        token = splited[0]
        for i in range(1, len(splited)):
            map_dict[splited[i]] = token

    index1 = tokenIdx - 3
    index2 = tokenIdx
    index3 = tokenIdx + len(entity.split())
    index4 = tokenIdx + 3 + len(entity.split())

    test_pos2 = []
    for item in test_pos:
        pos = idx2pos[str(item)]
        new_idx = pos_list.index(map_dict.get(pos)) if map_dict.get(pos) else pos_list.index('other')
        test_pos2.append(new_idx)

    left_pos = test_pos2[index1:index2] if index1 >= 0 else [0] * abs(index1) + test_pos2[0:index2]
    right_pos = test_pos2[index3:index4] if index4 <= len(test_pos2) else test_pos2[index3:] + [0] * (index4-len(test_pos2))
    pos = left_pos + [1] + right_pos    # 1:'NN'

    # local_collocations_fea = []
    # DEFAULT_COLLOCATIONS = ["-2,-2", "-1,-1", "1,1", "2,2", "-2,-1", "-1,1", "1,2", "-3,-1", "-2,1", "-1,2", "1,3"]
    # local_collocations = LocalCollocationsExtractor(tokenIdx, len(entity.split()), len(test_x), test_x)
    # features = features_dict.get(entity)
    # if features:
    #     for i in range(len(local_collocations)):
    #         if local_collocations[i] in features[DEFAULT_COLLOCATIONS[i]]:
    #             x = list(features[DEFAULT_COLLOCATIONS[i]]).index(local_collocations[i])
    #             local_collocations_fea.append(x)
    #         else:
    #             local_collocations_fea.append(-1)
    # else:
    #     local_collocations_fea = [-1]*11

    left_x = test_x[index1:index2] if index1 >= 0 else [0] * abs(index1) + test_x[0:index2]
    right_x = test_x[index3:index4] if index4 <= len(test_x) else test_x[index3:] + [0] * (index4 - len(test_x))
    surroundding_word = left_x + [test_x[tokenIdx]] + right_x

    assert len(pos) == 7
    assert len(surroundding_word) == 7
    # assert len(local_collocations_fea) == 11

    return pos, surroundding_word


def createCharDict():
    '''
    创建字符字典
    '''
    # charSet = set()
    # with open(trainCorpus + '/' + 'train.out', encoding='utf-8') as f:
    #     for line in f:
    #         if not line == '\n':
    #             a = line.strip().split('\t')
    #             charSet.update(a[0])  # 获取字符集合

    char2idx = {}
    char2idx['None'] = len(char2idx)  # 0索引用于填充
    for char in string.printable:
        char2idx[char] = len(char2idx)
    char2idx['**'] = len(char2idx)  # 用于那些未收录的字符
    # print(char2idx)
    return char2idx


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)  # np.dot(vector1,vector2)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos   # ?
    return sim


def check(s):
    '''
    python 括号检测是否匹配？
    '''
    arr = []
    for c in s:
        if c in SYMBOLS_L:
            # 左符号入栈
            arr.append(c)
        elif c in SYMBOLS_R:
            # 右符号要么出栈，要么匹配失败
            if arr and arr[-1] == SYMBOLS[c]:
                arr.pop()
            else:
                return 'R'
    if arr:
        return 'L'
    else:
        return not arr


def Indent(dom, node, indent=0):
    # Copy child list because it will change soon
    children = node.childNodes[:]
    # Main node doesn't need to be indented
    if indent:
        text = dom.createTextNode('\n' + '\t' * indent)
        node.parentNode.insertBefore(text, node)
    if children:
        # Append newline after last child, except for text nodes
        if children[-1].nodeType == node.ELEMENT_NODE:
            text = dom.createTextNode('\n' + '\t' * indent)
            node.appendChild(text)
        # Indent children which are elements
        for n in children:
            if n.nodeType == node.ELEMENT_NODE:
                Indent(dom, n, indent + 1)


def makeEasyTag(dom, tagname, value, type='text'):
    '''
    :param dom: DOM文档对象
    :param tagname: 标签名
    :param value:   文本结点值
    :param type:
    :return:    标签对+值
    '''
    tag = dom.createElement(tagname)    # 二、元素结点的生成 <tagname></tagname>
    if value.find(']]>') > -1:
        type = 'text'
    if type == 'text':
        value = value.replace('&', '&amp;')
        value = value.replace('<', '&lt;')
        text = dom.createTextNode(value)    # 三、文本结点text的生成
    elif type == 'cdata':
        text = dom.createCDATASection(value)
    tag.appendChild(text)       # 将子结点加就到元素结点中,<tagname>value</tagname>
    return tag


def convert_2_BIO(label):
    """ Convert inplace IOBES encoding to BIO encoding """
    tag = []
    i = 0
    while i < len(label):
        char = label[i]
        i += 1
        if char == 'S':
            tag.append('B')
        elif char == 'E':
            tag.append('I')
        elif char == 'I':
            tag.append('I')
            if i < len(label) and label[i] == 'B':
                tag.append('I')
                i = i + 1
        else:
            tag.append(char)
    return tag


def testLabel2Word():
    word_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    label_list = [0,1,2,2,3,4,0,1,2,0,1,3]
    result = ''
    prex = 0
    entities = []
    for i in range(len(word_list)):
        word = word_list[i]
        label = label_list[i]
        if label == 1:
            if result:
                entities.append(result.strip())
                result = ''
            prex = label
            result = word + ' '
        elif label == 2:
            if prex == 1:
                result += word + ' '
        elif label == 3:
            if result:
                entities.append(result.strip())
                result = ''
            prex = label
            result = word + ' '
        elif label == 4:
            if prex == 3:
                result += word + ' '
        else:
            if not result == '':
                entities.append(result.strip())
                result = ''
            else:
                result = ''
    if not result == '':
        entities.append(result.strip())
    print(entities)


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
        if entity_variants1.startswith(com) or entity_variants1.endswith(com):
            entity_variants1 = entity_variants1.replace(com, '').strip()
            entity_variants2 = entity_variants2.replace(com, '').strip()

    entity_variants3 = re.findall(r'[0-9]+|[a-z]+', entity_variants1)
    entity_variants3 = ' '.join(entity_variants3)
    entity_variants3 = entity_variants3.replace('  ', ' ').strip()

    # start with the longest one
    return entity_variants1, entity_variants2, entity_variants3


def extractID(path1,path2):
    protein2id = {}
    with open(path1) as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        line = line.replace('/', ';')
        splited = line.split('\t')
        ids = splited[0].split('; ')
        word_list = splited[1].strip('\n').strip().split(';')
        for id in ids:
            if id:
                for word in word_list:
                    #  对实体进行过滤
                    entity = strippingAlgorithm(word)[0]
                    if entity not in protein2id:
                        protein2id[entity] = OrderedDict()
                    if id not in protein2id[entity]:
                        protein2id[entity][id] = 1
                    else:
                        protein2id[entity][id] += 1

    # 按频度重新排序
    protein2id_1 = {}
    for key, value in protein2id.items():
        value_sorted = sorted(value.items(), key=lambda item: item[1], reverse=True)
        protein2id_1[key] = [item[0] for item in value_sorted]


    gene2id = {}
    with open(path2) as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        line = line.replace('/', ';')
        splited = line.split('\t')
        id = splited[0]
        word_list = splited[1].strip('\n').strip().split(';')
        for word in word_list:
            #  对实体进行过滤
            entity = strippingAlgorithm(word)[0]
            if entity not in gene2id:
                gene2id[entity] = OrderedDict()
            if id and id not in gene2id[entity]:
                gene2id[entity][id] = 1
            else:
                gene2id[entity][id] += 1

    # 按频度重新排序
    gene2id_1 = {}
    for key, value in gene2id.items():
        value_sorted = sorted(value.items(), key=lambda item: item[1], reverse=True)
        gene2id_1[key] = [item[0] for item in value_sorted]

    import pickle as pkl
    with open('pg2id.pkl', "wb") as f:
        res = (protein2id_1, gene2id_1)
        pkl.dump(res, f, -1)
    return protein2id_1, gene2id_1


if __name__ == '__main__':

    # from sample.utils.helpers import extractID

    path1 = '/home/administrator/PycharmProjects/embedding/uniprot_sprot2.txt'
    path2 = '/home/administrator/PycharmProjects/embedding/gene_info_processed2.txt'
    protein2id, gene2id = extractID(path1, path2)

    nn = 0
    for key, value in protein2id.items():
        if nn == 100:
            break
        nn += 1
        print(key, value)
    nn = 0
    for key, value in gene2id.items():
        if nn == 100:
            break
        nn += 1
        print(key, value)