import pickle as pkl

import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from sample.keraslayers.ChainCRF import create_custom_objects
from sample.utils.write_test_result2 import writeOutputToFile


def getTestData():
    # print(string.punctuation)   # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # print(string.printable)

    rootCorpus = r'data'
    embeddingPath = r'/home/administrator/PycharmProjects/embedding'
    with open(rootCorpus + '/test.pkl', "rb") as f:
        test_x, test_y, test_char, test_cap, test_pos, test_chunk = pkl.load(f)
    with open(embeddingPath+'/length.pkl', "rb") as f:
        word_maxlen, sentence_maxlen = pkl.load(f)

    dataSet = {}
    dataSet['test'] = [test_x, test_cap, test_pos, test_chunk]

    # pad the sequences with zero
    for key, value in dataSet.items():
        for i in range(len(value)):
            dataSet[key][i] = pad_sequences(value[i], maxlen=sentence_maxlen, padding='post')

    # pad the char sequences with zero list
    for j in range(len(test_char)):
        if len(test_char[j]) < sentence_maxlen:
            test_char[j].extend(np.asarray([[0] * word_maxlen] * (sentence_maxlen - len(test_char[j]))))

    dataSet['test'].insert(1, np.asarray(test_char))

    test_y = pad_sequences(test_y, maxlen=sentence_maxlen, padding='post')

    print(np.asarray(test_char).shape)     # (4528, 418, 23)
    print(test_y.shape)    # (4528, 639, 5)

    print('create test set done!\n')
    return dataSet


import os
if not os.path.exists('predictions.pkl'):
    dataSet = getTestData()
    model = load_model('model/Model_3_74.07.h5', custom_objects=create_custom_objects())
    print('加载模型成功!!')

    predictions = model.predict(dataSet['test'])
    y_pred = predictions.argmax(axis=-1)

    with open('result/prediction.txt', 'w') as f:
        for line in y_pred:
            for k in line:
                f.write(str(k))
            f.write('\n')

    with open('predictions.pkl', "wb") as f:
        pkl.dump((y_pred), f, -1)
else:
    with open('predictions.pkl', "rb") as f:
        y_pred = pkl.load(f)


# 将实体预测结果，以特定格式写入XML文件，用于scorer进行评估
writeOutputToFile(r'data/test.out.txt', y_pred)


'''
python bioid_score.py --verbose 1 --force \
存放结果文件的目录 正确答案所在的目录 system1:预测结果所在的目录
'''