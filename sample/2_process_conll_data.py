'''
语料预处理，加入词典特征
'''
import pickle as pkl
from collections import OrderedDict
import numpy as np
np.random.seed(1337)
import string
from tqdm import tqdm
import word2vec
from keras.preprocessing.text import Tokenizer
from sample.utils.helpers import wordNormalize, createCharDict
# from sample.utils.helpers import get_stop_dic


corpusPath = r'F:/B910/PythonCode/keras_bc6_track1/sample/data'
embeddingPath = r'F:/B910\PythonCode/keras_bc6_track1/sample/embedding'
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

    # """获取停用词词典+标点符号"""
    stop_word = []
    # from nltk.corpus import stopwords
    # import string
    # stopWord_path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/stopwords_gene'
    # stop_word = []
    # with open(stopWord_path, 'r') as f:
    #     for line in f:
    #         stop_word.append(line.strip('\n'))
    # stop_word.extend(stopwords.words('english'))
    # stop_word.extend(list(string.punctuation))
    # stop_word = list(set(stop_word))

    word_id_filter = []
    miss_num=0
    num=0
    # embeddings_index = readGensimFile(embedFile)
    embeddings_index = readBinEmbedFile(embedFile)
    print('Found %s word vectors.' % len(embeddings_index))  # 4706287

    num_words = len(word_index)+1
    embedding_matrix = np.zeros((num_words, word_size))
    for word, i in word_index.items():
        # if word in stop_word:
        #     word_id_filter.append(i)
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
    # [5180, 703, 6778, 21098, 10790, 88, 2661, 150, 551, 2035, 5209, 2834, 5197, 541, 26, 15, 1135, 47, 914, 9655, 4976, 1060, 7017, 647, 205, 268, 483, 1734, 2707, 8, 16228, 18, 390, 516, 4325, 5194, 7359, 5022, 76, 22, 605, 8413, 2309, 4421, 27, 42, 947, 24317, 5597, 2566, 24965, 169, 217, 2901, 595, 891, 325, 615, 603, 775, 153, 137, 852, 10748, 22948, 7, 192, 6032, 3399, 28, 10281, 15853, 520, 522, 89, 16368, 32, 6912, 2140, 1374, 579, 4980, 3927, 20, 10, 2223, 488, 3351, 459, 649, 99, 173, 1116, 7803, 1644, 96, 3811, 2366, 432, 1239, 666, 785, 521, 1279, 13, 1538, 16346, 16, 1612, 337, 378, 12, 381, 300, 18695, 1976, 64, 25, 8504, 3915, 655, 140, 231, 181, 1735, 8439, 6870, 5205, 69, 3932, 247, 846, 491, 678, 77, 63, 1980, 578, 1189, 21733, 249, 134, 1440, 671, 59, 3939, 21441, 229, 4593, 2208, 6569, 237, 238, 2267, 4089, 78, 15593, 136, 416, 549, 48, 1192, 1649, 5702, 375, 3601, 620, 331, 9, 239, 3707, 11926, 1150, 43, 9414, 37, 315]
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

    for name in ['train', 'test']:
        with open(trainCorpus + '/' + name + '.final.txt', encoding='utf-8') as f:
            data_sen = []
            char_sen = []
            cap_sen = []
            pos_sen = []
            chunk_sen = []
            labels_sen = []
            dict_sen = []
            num = -1
            line_number = 0
            for line in f:
                if line == '\n':
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
                    else:
                        datasDic[name].append(' '.join(data_sen[:maxlen_s]))  # .join()
                        charsDic[name].append(char_sen[:maxlen_s])
                        capDic[name].append(cap_sen[:maxlen_s])
                        posDic[name].append(pos_sen[:maxlen_s])
                        chunkDic[name].append(chunk_sen[:maxlen_s])
                        labelsDic[name].append(labels_sen[:maxlen_s])
                        dictDic[name].append(dict_sen[:maxlen_s])

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

    # # 加BC2GM预料效果不太好
    # name = 'train'
    # with open('F:\B910\PythonCode\keras_bc6_track1\sample\data\data.final.txt', encoding='utf-8') as f:
    #     data_sen = []
    #     char_sen = []
    #     cap_sen = []
    #     pos_sen = []
    #     chunk_sen = []
    #     labels_sen = []
    #     dict_sen = []
    #     num = -1
    #     line_number = 0
    #     for line in f:
    #         if line == '\n':
    #             line_number += 1
    #             num += 1
    #             if nb_word > sen_len_list[-1]:
    #                 sen_len_list.append(nb_word)
    #             assert len(data_sen) == len(labels_sen)
    #             if nb_word <= maxlen_s:
    #                 datasDic[name].append(' '.join(data_sen))  # .join()
    #                 charsDic[name].append(char_sen)
    #                 capDic[name].append(cap_sen)
    #                 posDic[name].append(pos_sen)
    #                 chunkDic[name].append(chunk_sen)
    #                 labelsDic[name].append(labels_sen)
    #                 dictDic[name].append(dict_sen)
    #             else:
    #                 datasDic[name].append(' '.join(data_sen[:maxlen_s]))  # .join()
    #                 charsDic[name].append(char_sen[:maxlen_s])
    #                 capDic[name].append(cap_sen[:maxlen_s])
    #                 posDic[name].append(pos_sen[:maxlen_s])
    #                 chunkDic[name].append(chunk_sen[:maxlen_s])
    #                 labelsDic[name].append(labels_sen[:maxlen_s])
    #                 dictDic[name].append(dict_sen[:maxlen_s])
    #
    #             # if name=='train' and num==3725:
    #             #     print(len(data_sen[:maxlen_s]), len(labels_sen[:maxlen_s]))
    #             #     print(data_sen[:maxlen_s])
    #             # elif name=='train' and num==7587:
    #             #     print(len(data_sen[:maxlen_s]), len(labels_sen[:maxlen_s]))
    #             #     print(data_sen[:maxlen_s])
    #             assert len(data_sen[:maxlen_s]) == len(labels_sen[:maxlen_s])
    #
    #             data_sen = []
    #             char_sen = []
    #             cap_sen = []
    #             pos_sen = []
    #             chunk_sen = []
    #             labels_sen = []
    #             nb_word = 0
    #         else:
    #             # print(line_number)
    #             nb_word += 1
    #             line_number += 1
    #             token = line.replace('\n', '').split('\t')
    #             word = token[0]
    #             pos = token[1]
    #             chunk = token[2]
    #             dict = token[3]
    #             label = token[-1]
    #
    #             labelIdx = label2idx.get(label) if label in label2idx else label2idx.get('O')
    #             labelIdx = np.eye(len(label2idx))[labelIdx]
    #             labelIdx = list(labelIdx)
    #
    #             # 大小写特征
    #             try:
    #                 cap = casing_vocab[getCasting(word)]
    #             except:
    #                 continue
    #             # 对单词进行清洗
    #             word = wordNormalize(word)
    #             # 获取pos和chunk字典
    #             if not pos in pos2idx:
    #                 pos2idx[pos] = len(pos2idx)
    #             if not chunk in chunk2idx:
    #                 chunk2idx[chunk] = len(chunk2idx)
    #             # 字符特征
    #             nb_char = 0
    #             char_w = []
    #             for char in word:
    #                 nb_char += 1
    #                 charIdx = char2idx.get(char)
    #                 if not charIdx:
    #                     chars_not_exit.add(char)
    #                     char_w.append(char2idx['**'])
    #                 else:
    #                     char_w.append(charIdx)
    #             if nb_char > word_len_list[-1]:
    #                 word_len_list.append(nb_char)
    #             # 字典特征
    #             dict_fea = dict2idx[dict]
    #
    #             data_sen.append(word)
    #             char_sen.append(char_w)
    #             cap_sen.append(cap)
    #             pos_sen.append(pos)
    #             chunk_sen.append(chunk)
    #             labels_sen.append(labelIdx)
    #             dict_sen.append(dict_fea)

    print('chars not exits in the char2idx:{}'.format(chars_not_exit))
    print('longest char is', word_len_list[-5:])  # [12, 14, 17, 21, 34]
    print('longest word is', sen_len_list[-5:])  # [370, 422, 455, 752, 902]
    print('len(pos2idx):{}'.format(len(pos2idx)))     # 50
    print('len(chunk2idx):{}'.format(len(chunk2idx)))     # 22

    return datasDic, charsDic, capDic, posDic, chunkDic, labelsDic, dictDic, pos2idx, chunk2idx


def main():

    # stop_word_dic = get_stop_dic()
    datasDic, charsDic, capDic, posDic, chunkDic, labelsDic, dictDic, pos2idx, chunk2idx = getData(corpusPath, sen_len_list)

    with open('pos2idx.txt', 'w') as f:
        for key, value in pos2idx.items():
            if key:
                f.write('{}\t{}\n'.format(key, value))

    # # 将验证集并入训练集(无关)
    # for item in [datasDic, charsDic, capDic, posDic, chunkDic, labelsDic]:
    #     item['train'].extend(item['devel'])
    #     item['devel'] = None
    # print('合并训练集data大小：', len(datasDic['train']))  # 13697

    data = []
    for name in ['train', 'test']:
        print('训练集data大小：', len(datasDic[name]))  # 13697   4528
        data.extend(datasDic[name])
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          filters='',   # 需要过滤掉的字符列表（或连接）
                          split=' ')    # 词的分隔符
    tokenizer.fit_on_texts(data)
    data = []

    word_index = tokenizer.word_index   # 将词（字符串）映射到索引（整型）的字典
    word_counts = tokenizer.word_counts # 在训练时将词（字符串）映射到其出现次数的字典
    print('Found %s unique tokens.' % len(word_index))  # 26987

    # 将训练数据序列化
    datasDic['train'] = tokenizer.texts_to_sequences(datasDic['train'])
    datasDic['test'] = tokenizer.texts_to_sequences(datasDic['test'])
    # 保证训练数据与标签长度一致
    for i in range(len(datasDic['train'])):
        line = datasDic['train'][i]
        if not len(line) == len(labelsDic['train'][i]):
            print(i)
    for i in range(len(datasDic['test'])):
        line = datasDic['test'][i]
        if not len(line) == len(labelsDic['test'][i]):
            print(i)

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
        pkl.dump((datasDic['train'], labelsDic['train'], charsDic['train'],
                  capDic['train'], posDic['train'], chunkDic['train'], dictDic['train']), f, -1)
    with open(corpusPath+'/test.pkl', "wb") as f:
        pkl.dump((datasDic['test'], labelsDic['test'], charsDic['test'],
                  capDic['test'], posDic['test'], chunkDic['test'], dictDic['test']), f, -1)

    with open(embeddingPath + '/emb.pkl', "wb") as f:
        pkl.dump((embedding_matrix), f, -1)
    with open(embeddingPath + '/length.pkl', "wb") as f:
        pkl.dump((maxlen_w, maxlen_s), f, -1)
    embedding_matrix = {}

    print('\n保存成功')


if __name__ == '__main__':
    main()
