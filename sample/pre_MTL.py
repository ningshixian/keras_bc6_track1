import pickle as pkl
from collections import OrderedDict
import numpy as np
np.random.seed(1337)
from tqdm import tqdm
import re
import word2vec
import string
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
stop_word = stopwords.words('english')


trainCorpus = r'/Users/ningshixian/Desktop/BC6_Track1/BioIDtraining_2/train'

embeddingFile = 'embedding/wikipedia-pubmed-and-PMC-w2v.bin'

label2idx = {'O': 0, 'B-GENE': 1, 'I-GENE': 2, 'B-PROTEIN': 3, 'I-PROTEIN': 4}

maxlen_s = 180  # 句子截断长度
maxlen_w = 25   # 单词截断长度
word_size = 200  # 词向量维度
MAX_NB_WORDS = 100000
word_lenlist = []
sentence_lenlist = [0]  # 用于保存最长句子的长度（不用）

def wordNormalize(word):
    word = word.lower()
    word = re.sub(u'\s+', '', word, flags=re.U)  # 匹配任何空白字符
    # word = word.replace("--", "-")
    # word = re.sub("\"+", '"', word)

    # # 特殊符号归一化为@表示
    # temp = word
    # for i in range(len(word)):
    #     char = word[i]
    #     if char not in str_num:
    #         temp = temp.replace(char, '@')
    # word = temp

    # # 数字/字母的预处理
    # rNUM = u'(-|\+)?\d+((\.|·)\d+)?%?'
    # rENG = u'[A-Za-z_.]+'
    # word = re.sub(rNUM, u'0', word, flags=re.U)
    # word = re.sub(rENG, u'X', word)
    # word = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}", 'DATE_TOKEN', word)
    # word = re.sub("[0-9]{2}:[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
    # word = re.sub("[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
    # word = re.sub("[0-9.,]+", 'NUMBER', word)
    return word


def readEmbedFile(embFile):
    """
    读取预训练的词向量文件，引入外部知识
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.25, 0.25, word_size)

    # 针对二进制格式保存的词向量文件
    model = word2vec.load(embeddingFile)
    print('加载词向量文件完成')
    for i in tqdm(range(len(model.vectors))):
        vector = model.vectors[i]
        word = model.vocab[i].lower()   # convert all characters to lowercase
        embeddings[word] = vector

    # with open(embFile) as f:
    #     lines = f.readlines()
    # for i in tqdm(range(len(lines))):
    #     line = lines[i]
    #     if len(line.split())<=2:
    #         continue
    #     values = line.strip().split()
    #     word = values[0].lower()
    #     vector = np.asarray(values[1:], dtype=np.float32)
    #     embeddings[word] = vector

    return embeddings


def produce_matrix(word_index):
    miss_num=0
    num=0
    embeddings_index = readEmbedFile(embeddingFile)
    print('Found %s word vectors.' % len(embeddings_index))  # 356224

    num_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((num_words, word_size))
    for word, i in word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            num=num+1
        else:
            miss_num=miss_num+1
            vec = embeddings_index["UNKNOWN_TOKEN"]
        embedding_matrix[i] = vec
    print('missnum',miss_num)    # 9929
    print('num',num)    # 58858
    return embedding_matrix


def create_char_dict():
    char2idx = {}

    # for char in string.printable:
    #     charSet.add(char)

    # with open(trainCorpus + '/' + 'train.out', encoding='utf-8') as f:
    #     for line in f:
    #         if not line == '\n':
    #             a = line.strip().split('\t')
    #             charSet.update(a[0])  # 获取字符集合

    for char in string.printable:
        char2idx[char] = len(char2idx)
    char2idx['**'] = len(char2idx)  # 用于那些未收录的字符
    print(char2idx)
    return char2idx


def padCharacters(data, max_char):
    # data = np.asarray(data)
    for senIdx in tqdm(range(len(data))):
        for tokenIdx in range(len(data[senIdx])):
            token = data[senIdx][tokenIdx]
            try:
                lenth = max_char - len(token)
                if lenth >= 0:
                    data[senIdx][tokenIdx] = np.pad(token, (0, lenth), 'constant')
                else:
                    data[senIdx][tokenIdx] = token[:max_char]
                assert len(data[senIdx][tokenIdx])==max_char
            except:
                print(max_char, token)
                raise


if __name__ == '__main__':

    # 初始化
    nb_word = 0
    char2idx = create_char_dict()

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
                if nb_word > sentence_lenlist[-1]:
                    sentence_lenlist.append(nb_word)
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
                line_number +=1
                nb_word+=1
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
                word = wordNormalize(word)
                label = label2idx.get(label)
                if not label:
                    label = label2idx.get('O')
                label = np.eye(len(label2idx))[label]
                label = list(label)

                for char in word:
                    if not char2idx.get(char):
                        chars_not_exit.add(char)
                temp_character = [char2idx[char] if char2idx.get(char) else char2idx['**'] for char in word]

                char_sen.append(temp_character)
                data_sen.append(word)
                cap_sen.append(cap)
                labels_sen.append(label)

    print('chars not exits in the char2idx:{}'.format(chars_not_exit))
    print('longest char is', maxlen_w)  # 44
    print('longest word is', sentence_lenlist[-5:])  # 426

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters=' ')
    tokenizer.fit_on_texts(datasDic)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.\n' % len(word_index))  # 75305

    print('训练集data大小：', len(datasDic))  # 30802

    # 将训练数据序列化
    datasDic = tokenizer.texts_to_sequences(datasDic)

    # 补齐字符向量长度 word_maxlen
    padCharacters(charsDic, maxlen_w)

    # 获取词向量矩阵
    embedding_matrix = produce_matrix(word_index)

    # 保存文件
    with open(trainCorpus+'/pkl/train.pkl', "wb") as f:
        pkl.dump((datasDic, labelsDic, charsDic, capDic), f, -1)

    with open('embedding/emb.pkl', "wb") as f:
        pkl.dump((embedding_matrix, maxlen_w, maxlen_s, char2idx), f, -1)
    embedding_matrix = {}

    print('\n保存成功')
