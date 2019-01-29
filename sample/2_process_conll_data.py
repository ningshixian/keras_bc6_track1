'''
语料预处理，加入词典特征
'''
import pickle as pkl
from collections import OrderedDict
import numpy as np
np.random.seed(1337)
from tqdm import tqdm
import word2vec
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sample.utils.helpers import wordNormalize, createCharDict
from sample.utils.helpers import get_stop_dic
# import nltk
# nltk.download()
from nltk.corpus import stopwords

import string


corpusPath = r'data'
embeddingPath = r'/home/administrator/PycharmProjects/embedding'
embeddingFile = r'wikipedia-pubmed-and-PMC-w2v.bin'
dict2idx = {'O': 0, 'B': 1, 'I': 2}  # 字典特征
label2idx = {'O': 0, 'B-protein': 1, 'I-protein': 2, 'B-gene': 3, 'I-gene': 4}

maxlen_s = 455  # 句子截断长度
maxlen_w = 21  # 单词截断长度
word_size = 200  # 词向量维度
MAX_NB_WORDS = None  # 不设置最大词数
word_len_list = [0]  # 用于统计单词长度
sen_len_list = [0]  # 用于统计句子长度



def readBinEmbedFile(embFile):
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
        word = model.vocab[i].lower()   # convert all characters to lowercase
        embeddings[word] = vector
    return embeddings


def readTxtEmbedFile(embFile):
    """
    读取预训练的词向量文件，引入外部知识
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.1, 0.1, word_size)

    with open(embFile) as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        if len(line.split())<=2:
            continue
        values = line.strip().split()
        word = values[0].lower()
        vector = np.asarray(values[1:], dtype=np.float32)
        embeddings[word] = vector
    return embeddings


def readGensimFile(embFile):
    print("\nProcessing Embedding File...")
    import gensim
    model = gensim.models.Word2Vec.load(embFile)  # 'word2vec_words.model'
    word_vectors = model.wv
    return word_vectors


def produce_matrix(word_index, embedFile):
    '''
    生成词向量矩阵 embedding_matrix
    '''

    """获取停用词词典+标点符号"""
    stopWord_path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/stopwords_gene'
    stop_word = []
    with open(stopWord_path, 'r') as f:
        for line in f:
            stop_word.append(line.strip('\n'))
    stop_word.extend(stopwords.words('english'))
    stop_word.extend(list(string.punctuation))
    stop_word = list(set(stop_word))

    word_id_filter = []
    miss_num=0
    num=0
    # embeddings_index = readGensimFile(embedFile)
    embeddings_index = readBinEmbedFile(embedFile)
    print('Found %s word vectors.' % len(embeddings_index))  # 4706287

    num_words = len(word_index)+1
    embedding_matrix = np.zeros((num_words, word_size))
    for word, i in word_index.items():
        if word in stop_word:
            word_id_filter.append(i)

        if word.lower() in embeddings_index:
            vec = embeddings_index.get(word.lower())
        else:
            for punc in string.punctuation:
                word = word.replace(punc, '')
            vec = embeddings_index.get(word.lower())
        if vec is not None:
            num=num+1
        else:
            miss_num=miss_num+1
            vec = embeddings_index["UNKNOWN_TOKEN"] # 未登录词均统一表示
        embedding_matrix[i] = vec
    print('missnum',miss_num)    # 8381
    print('num',num)    # 20431
    print('word_id_filter:{}'.format(word_id_filter))

    '''
    [239, 153, 137, 300, 64, 947, 2309, 570, 10, 69, 238, 175, 852, 7017, 378, 136, 5022, 1116, 5194, 14048, 28, 217,
     4759, 7359, 201, 671, 11, 603, 15, 1735, 2140, 390, 2366, 12, 649, 4, 1279, 3351, 3939, 5209, 16, 43, 2208, 8,
     5702, 4976, 325, 891, 541, 1649, 17, 416, 2707, 108, 381, 678, 249, 5205, 914, 5180, 5, 20, 18695, 15593, 5597,
     730, 1374, 18, 2901, 1440, 237, 150, 44, 10748, 549, 3707, 4325, 27, 331, 522, 10790, 297, 1060, 1976, 7803, 1150,
     1189, 2566, 192, 5577, 703, 666, 315, 488, 89, 1103, 231, 16346, 9655, 6569, 605, 6, 294, 3932, 24965, 9, 775,
     4593, 76, 21733, 140, 229, 16368, 21098, 181, 620, 134, 6032, 268, 2267, 22948, 88, 655, 24768, 6870, 25, 615,
     4421, 99, 3, 375, 483, 7, 2661, 32, 2223, 42, 1612, 595, 22, 37, 432, 8439, 67, 15853, 6912, 459, 21441, 3811,
     1538, 1644, 2834, 1192, 5197, 1734, 78, 647, 247, 491, 16228, 23, 578, 34, 47, 77, 1239, 846, 26, 24317, 785, 3601,
     8504, 29, 9414, 520, 3399, 2035, 6778, 96, 2048, 1, 579, 1135, 173, 4089, 4980, 205, 63, 516, 169, 8413, 1980, 337,
     19, 521, 13, 48, 551, 3927, 59, 10281, 11926, 3915]
    '''

    return embedding_matrix


def padCharacters(chars_dic, max_char):
    print('\n补齐字符向量长度')
    for senIdx in tqdm(range(len(chars_dic))):
        for tokenIdx in range(len(chars_dic[senIdx])):
            token = chars_dic[senIdx][tokenIdx]
            lenth = max_char - len(token)
            if lenth >= 0:
                chars_dic[senIdx][tokenIdx] = np.pad(token, (0, lenth), 'constant')
            else:
                chars_dic[senIdx][tokenIdx] = token[:max_char]
            assert len(chars_dic[senIdx][tokenIdx])==max_char
    return chars_dic


def getCasting(word):
    casing = 'other'

    if word.isdigit():
        casing = 'numeric'
    elif word.islower():
        casing='allLower'
    elif word.isupper():    # DF43 也属于
        casing = 'allUpper'
    elif word[0].isupper():
        casing = 'initialUpper'

    return casing

def getCastingVocab():
    entries = ['other', 'numeric','allLower', 'allUpper', 'initialUpper']
    return {entries[idx]:idx for idx in range(len(entries))}


def getData(trainCorpus, sen_len_list):
    '''
    获取训练和验证数据
    '''
    nb_word = 0     # 用于统计句子的单词个数
    chars_not_exit = set()     # 统计未登录字符
    char2idx = createCharDict()
    casing_vocab = getCastingVocab()    # 大小写字典
    pos2idx = OrderedDict()
    pos2idx['None'] = 0
    chunk2idx = {'None': 0}

    datasDic = {'train':[], 'devel':[], 'test':[]}
    charsDic = {'train':[], 'devel':[], 'test':[]}
    capDic = {'train':[], 'devel':[], 'test':[]}
    posDic = {'train': [], 'devel': [], 'test':[]}
    chunkDic = {'train': [], 'devel': [], 'test':[]}
    labelsDic = {'train':[], 'devel':[], 'test':[]}
    dictDic = {'train':[], 'devel':[], 'test':[]}
    ngramDic = {'train':[], 'devel':[], 'test':[]}

    len_list={'train':[], 'test':[]}
    for name in ['train', 'test']:
        with open(trainCorpus + '/' + name + '.final.txt', encoding='utf-8') as f:
            data_sen = []
            char_sen = []
            cap_sen = []
            pos_sen = []
            chunk_sen = []
            labels_sen = []
            dict_sen = []
            # ngram_sen = []
            num = -1
            line_number = 0
            for line in f:
                if line == '\n':
                    len_list[name].append(nb_word)
                    line_number +=1
                    num += 1
                    if nb_word > sen_len_list[-1]:
                        sen_len_list.append(nb_word)
                    assert len(data_sen) == len(labels_sen)
                    if nb_word <= maxlen_s:
                        datasDic[name].append(' '.join(data_sen))  # .join()
                        charsDic[name].append(char_sen)
                        capDic[name].append(cap_sen)
                        posDic[name].append(pos_sen)
                        chunkDic[name].append(chunk_sen)
                        labelsDic[name].append(labels_sen)
                        dictDic[name].append(dict_sen)
                        # ngramDic[name].append(ngram_sen)
                    else:
                        datasDic[name].append(' '.join(data_sen[:maxlen_s]))  # .join()
                        charsDic[name].append(char_sen[:maxlen_s])
                        capDic[name].append(cap_sen[:maxlen_s])
                        posDic[name].append(pos_sen[:maxlen_s])
                        chunkDic[name].append(chunk_sen[:maxlen_s])
                        labelsDic[name].append(labels_sen[:maxlen_s])
                        dictDic[name].append(dict_sen[:maxlen_s])
                        # ngramDic[name].append(ngram_sen[:maxlen_s])

                    # if name=='train' and num==3725:
                    #     print(len(data_sen[:maxlen_s]), len(labels_sen[:maxlen_s]))
                    #     print(data_sen[:maxlen_s])
                    # elif name=='train' and num==7587:
                    #     print(len(data_sen[:maxlen_s]), len(labels_sen[:maxlen_s]))
                    #     print(data_sen[:maxlen_s])
                    assert len(data_sen[:maxlen_s])==len(labels_sen[:maxlen_s])

                    data_sen = []
                    char_sen = []
                    cap_sen = []
                    pos_sen = []
                    chunk_sen = []
                    labels_sen = []
                    dict_sen = []
                    nb_word = 0
                else:
                    # print(line_number)
                    nb_word+=1
                    line_number += 1
                    token = line.replace('\n', '').split('\t')
                    word = token[0]
                    pos = token[1]
                    chunk = token[2]
                    dict = token[3]
                    label = token[-1]

                    labelIdx = label2idx.get(label) if label in label2idx else label2idx.get('O')
                    labelIdx = np.eye(len(label2idx))[labelIdx]
                    labelIdx = list(labelIdx)

                    # 大小写特征
                    cap = casing_vocab[getCasting(word)]
                    # 对单词进行清洗
                    word = wordNormalize(word)
                    # 获取pos和chunk字典
                    if not pos in pos2idx:
                        pos2idx[pos] = len(pos2idx)
                    if not chunk in chunk2idx:
                        chunk2idx[chunk] = len(chunk2idx)
                    # 字符特征
                    nb_char = 0
                    char_w = []
                    for char in word:
                        nb_char+=1
                        charIdx = char2idx.get(char)
                        if not charIdx:
                            chars_not_exit.add(char)
                            char_w.append(char2idx['**'])
                        else:
                            char_w.append(charIdx)
                    if nb_char > word_len_list[-1]:
                        word_len_list.append(nb_char)
                    # 字典特征
                    dict_fea = dict2idx[dict]

                    data_sen.append(word)
                    char_sen.append(char_w)
                    cap_sen.append(cap)
                    pos_sen.append(pos)
                    chunk_sen.append(chunk)
                    labels_sen.append(labelIdx)
                    dict_sen.append(dict_fea)


    print('chars not exits in the char2idx:{}'.format(chars_not_exit))
    print('longest char is', word_len_list[-5:])  # [557, 628, 752, 760, 902]
    print('longest word is', sen_len_list[-5:])  # [391, 399, 427, 451, 470]
    print('len(pos2idx):{}'.format(len(pos2idx)))     # 50
    print('len(chunk2idx):{}'.format(len(chunk2idx)))     # 22

    a = sorted(len_list['train'])
    b = sorted(len_list['test'])
    print('len_list: {}, {}'.format(a[-5:], b[-5:]))

    return datasDic, charsDic, capDic, posDic, chunkDic, labelsDic, dictDic, pos2idx, chunk2idx


def main():

    # stop_word_dic = get_stop_dic()
    datasDic, charsDic, capDic, posDic, chunkDic, labelsDic, dictDic, pos2idx, chunk2idx = getData(corpusPath, sen_len_list)

    with open('pos2idx.txt', 'w') as f:
        for key, value in pos2idx.items():
            if key:
                f.write('{}\t{}\n'.format(key, value))
    with open('chunk2idx.txt', 'w') as f:
        for key, value in chunk2idx.items():
            if key:
                f.write('{}\t{}\n'.format(key, value))

    # # 将验证集并入训练集(无关)
    # for item in [datasDic, charsDic, capDic, posDic, chunkDic, labelsDic]:
    #     item['train'].extend(item['devel'])
    #     item['devel'] = None
    # print('合并训练集data大小：', len(datasDic['train']))  # 13697

    elmo_input = {}
    for name in ['train', 'test']:
        elmo_input[name] = []
        for i in range(len(datasDic[name])):
            line = datasDic[name][i]
            line = line.split()
            # line = text_to_word_sequence(line)
            elmo_input[name].append(line)
    # print(elmo_input['train'][:2])

    data = []
    for name in ['train', 'test']:
        print('The size of {} is {}'.format(name, len(datasDic[name])))  # 13697   4528
        data.extend(datasDic[name])
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          filters='',   # 需要过滤掉的字符列表（或连接）
                          split=' ')    # 词的分隔符
    tokenizer.fit_on_texts(data)
    data = []

    word_index = tokenizer.word_index   # 将词（字符串）映射到索引（整型）的字典
    word_counts = tokenizer.word_counts # 在训练时将词（字符串）映射到其出现次数的字典
    print('Found %s unique tokens.' % len(word_index))  # 26987

    with open('word_index.pkl', "wb") as f:
        pkl.dump(word_index, f, -1)


    # 将训练数据序列化
    datasDic['train'] = tokenizer.texts_to_sequences(datasDic['train'])
    datasDic['test'] = tokenizer.texts_to_sequences(datasDic['test'])

    # 保证训练数据与标签长度一致
    for name in ['train', 'test']:
        for i in range(len(datasDic[name])):
            assert len(datasDic[name][i]) == len(labelsDic[name][i])== len(elmo_input[name][i])

    # # 保证训练数据与标签长度一致
    # for i in range(len(datasDic['train'])):
    #     line = datasDic['train'][i]
    #     if not len(line) == len(labelsDic['train'][i]):
    #         print(i)
    # for i in range(len(datasDic['test'])):
    #     line = datasDic['test'][i]
    #     if not len(line) == len(labelsDic['test'][i]):
    #         print(i)

    # 将pos特征序列化
    for name in ['train', 'test']:
        for i in range(len(posDic[name])):
            sent = posDic[name][i]
            posDic[name][i] = [pos2idx[item] for item in sent]
    # 将chunk特征序列化
    for name in ['train', 'test']:
        for i in range(len(chunkDic[name])):
            sent = chunkDic[name][i]
            chunkDic[name][i] = [chunk2idx[item] for item in sent]
    # 补齐字符向量长度 word_maxlen
    charsDic['train'] = padCharacters(charsDic['train'], maxlen_w)
    charsDic['test'] = padCharacters(charsDic['test'], maxlen_w)
    # 获取词向量矩阵
    embedding_matrix = produce_matrix(word_index, embeddingPath+'/'+embeddingFile)

    # 保存文件
    with open(corpusPath+'/train.pkl', "wb") as f:
        pkl.dump((datasDic['train'], elmo_input['train'], labelsDic['train'], charsDic['train'],
                  capDic['train'], posDic['train'], chunkDic['train'], dictDic['train']), f, -1)
    with open(corpusPath+'/test.pkl', "wb") as f:
        pkl.dump((datasDic['test'], elmo_input['test'], labelsDic['test'], charsDic['test'],
                  capDic['test'], posDic['test'], chunkDic['test'], dictDic['test']), f, -1)

    with open(embeddingPath + '/emb.pkl', "wb") as f:
        pkl.dump((embedding_matrix), f, -1)
    with open(embeddingPath + '/length.pkl', "wb") as f:
        pkl.dump((maxlen_w, maxlen_s), f, -1)
    embedding_matrix = {}

    print('\n保存成功')


if __name__ == '__main__':
    main()
