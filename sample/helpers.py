import re
import string
import sys
import numpy as np
print(sys.getdefaultencoding())


def get_answer():
    """Get an answer."""
    return True


# 对单词进行清洗
def wordNormalize(word):
    word = word.strip().lower()
    word = re.sub(u'\s+', '', word, flags=re.U)  # 匹配任何空白字符
    word = word.replace("--", "-")
    word = re.sub("\"+", '"', word)

    # 特殊符号归一化，替换为'-'
    temp = word
    for char in word:
        if char not in string.printable:
            temp = temp.replace(char, '-')
    word = temp
    return word


def entityNormalize(entity):
    for char in string.punctuation:
        if char in entity:
            entity = entity.replace(' '+char+' ', char)
            entity = entity.replace(char+' ', char)
            entity = entity.replace(' '+char, char)
    return entity



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


def idFilter2(res, type):
    '''
        将字典匹配或API匹配得到的Ids集合中，与type不符的部分去掉
        取第一个结果作为实体ID
    '''
    temp = []
    if res == 400:
        print('请求无效\n')
    results = res.split('\n')[1:-1]  # 去除开头一行和最后的''
    for line in results:
        Id = line.split('\t')[-1]
        temp.append(Id)
        break
        # if type == 'protein':
        #     if 'uniprot' in Id or 'protein' in Id:
        #         temp.append(Id)
        #         break
        # elif type == 'gene':
        #     if 'NCBI' in Id or 'gene' in Id:
        #         temp.append(Id)
        #         break
        # else:
        #     print(res)
    return temp


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
    print(char2idx)
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