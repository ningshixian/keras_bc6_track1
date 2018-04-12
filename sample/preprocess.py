import pickle as pkl
from collections import OrderedDict
import numpy as np
np.random.seed(1337)
from tqdm import tqdm
import word2vec
from keras.preprocessing.text import Tokenizer
from helpers import wordNormalize, createCharDict
import string

'''
对CONLL格式的训练数据进行预处理
'''

trainCorpus = r'data'
embeddingPath = r'/home/administrator/PycharmProjects/embedding'
embeddingFile = r'/wikipedia-pubmed-and-PMC-w2v.bin'

label2idx = {'O': 0, 'B-GENE': 1, 'I-GENE': 2, 'B-PROTEIN': 3, 'I-PROTEIN': 4}

maxlen_s = 473  # 句子截断长度(太短？)
maxlen_w = 48   # 单词截断长度
word_size = 200  # 词向量维度
MAX_NB_WORDS = None # 不设置最大词数
word_len_list = [0]
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
    embeddings["NUMBER"] = np.random.uniform(-0.25, 0.25, word_size)

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


def produce_matrix(word_index, embedFile):
    '''
    生成词向量矩阵 embedding_matrix
    '''
    miss_num=0
    num=0
    embeddings_index = readBinEmbedFile(embedFile)
    print('Found %s word vectors.' % len(embeddings_index))  # 356224

    num_words = len(word_index)+1
    embedding_matrix = np.zeros((num_words, word_size))
    for word, i in word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            num=num+1
        else:
            miss_num=miss_num+1
            vec = embeddings_index["UNKNOWN_TOKEN"] # 未登录词均统一表示
        embedding_matrix[i] = vec
    print('missnum',miss_num)    # 5393
    print('num',num)    # 25784
    return embedding_matrix


def padCharacters(chars_dic, max_char):
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


def getData(trainCorpus, sen_len_list):
    '''
    获取训练和测试数据
    '''
    nb_word = 0 # 用于统计句子的单词个数
    char2idx = createCharDict()

    datasDic = []
    charsDic = []
    capDic = []
    labelsDic = []
    chars_not_exit = set()

    with open(trainCorpus + '/' + 'train.out', encoding='utf-8') as f:
        data_sen = []
        char_sen = []
        cap_sen = []
        labels_sen = []
        num = 0
        line_number = 0
        for line in f:
            if line == '\n':
                line_number +=1
                num += 1
                if nb_word > sen_len_list[-1]:
                    sen_len_list.append(nb_word)
                assert len(data_sen) == len(labels_sen)
                if nb_word <= maxlen_s:
                    datasDic.append(' '.join(data_sen))  # .join()
                    charsDic.append(char_sen)
                    capDic.append(cap_sen)
                    labelsDic.append(labels_sen)
                else:
                    datasDic.append(' '.join(data_sen[:maxlen_s]))  # .join()
                    charsDic.append(char_sen[:maxlen_s])
                    capDic.append(cap_sen[:maxlen_s])
                    labelsDic.append(labels_sen[:maxlen_s])
                data_sen = []
                char_sen = []
                cap_sen = []
                labels_sen = []
                nb_word = 0
            else:
                nb_word+=1
                line_number += 1
                token = line.replace('\n', '').split('\t')
                word = token[0]
                pos = token[1]
                chunk = token[2]
                label = token[-1]
                try:
                    cap = 1 if word[0].isupper() else 0  # 大小写特征
                except:
                    print(line_number)
                    continue

                if label in label2idx:
                    labelIdx = label2idx.get(label)
                else:
                    labelIdx = label2idx.get('O')
                labelIdx = np.eye(len(label2idx))[labelIdx]
                labelIdx = list(labelIdx)

                # 是先对单词进行清洗，还是先获取字符组成？
                word = wordNormalize(word)
                nb_char = 0
                temp_character = []
                for char in word:
                    nb_char+=1
                    charIdx = char2idx.get(char)
                    if not charIdx:
                        chars_not_exit.add(char)
                        temp_character.append(char2idx['**'])
                    else:
                        temp_character.append(charIdx)
                # temp_character = [char2idx[char] if char2idx.get(char) else char2idx['**'] for char in word]
                if nb_char > word_len_list[-1]:
                    word_len_list.append(nb_char)

                char_sen.append(temp_character)
                data_sen.append(word)
                cap_sen.append(cap)
                labels_sen.append(labelIdx)

    print('chars not exits in the char2idx:{}'.format(chars_not_exit))
    print('longest char is', word_len_list[-3:])  # [41, 48, 52]
    print('longest word is', sen_len_list[-3:])  # [419, 473, 693]
    return datasDic, charsDic, capDic, labelsDic


if __name__ == '__main__':
    datasDic, charsDic, capDic, labelsDic = getData(trainCorpus, sen_len_list)
    print('训练集data大小：', len(datasDic))  # 13697

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          filters='',   # 需要过滤掉的字符列表（或连接）
                          split=' ')    # 词的分隔符
    tokenizer.fit_on_texts(datasDic)

    word_index = tokenizer.word_index   # 将词（字符串）映射到索引（整型）的字典
    word_counts = tokenizer.word_counts # 在训练时将词（字符串）映射到其出现次数的字典

    print('Found %s unique tokens.\n' % len(word_index))  # 31553

    # 将训练数据序列化
    datasDic = tokenizer.texts_to_sequences(datasDic)
    # 补齐字符向量长度 word_maxlen
    charsDic = padCharacters(charsDic, maxlen_w)
    # 获取词向量矩阵
    embedding_matrix = produce_matrix(word_index, embeddingPath+embeddingFile)

    # 保存文件
    with open(trainCorpus+'/train.pkl', "wb") as f:
        pkl.dump((datasDic, labelsDic, charsDic, capDic), f, -1)

    with open(embeddingPath + '/emb.pkl', "wb") as f:
        pkl.dump((embedding_matrix, maxlen_w, maxlen_s), f, -1)
    embedding_matrix = {}

    print('\n保存成功')
