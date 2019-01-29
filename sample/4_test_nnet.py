import os
import pickle as pkl
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sample.keraslayers.ChainCRF import create_custom_objects
from sample.utils.write_test_result import writeOutputToFile
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction = 0.3    # 按比例
# # config.gpu_options.allow_growth = True  # 自适应分配
# set_session(tf.Session(config=config))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

'''
获取模型需要预测的测试数据
'''
def getTestData():

    with open('data/test.pkl', "rb") as f:
        test_x, test_elmo, test_y, test_char, test_cap, test_pos, test_chunk, test_dict = pkl.load(f)

    embeddingPath = r'/home/administrator/PycharmProjects/embedding'
    with open(embeddingPath+'/length.pkl', "rb") as f:
        word_maxlen, sentence_maxlen = pkl.load(f)

    dataSet = {}
    batch_size = 32
    sentence_maxlen = 400
    dataSet['test'] = [test_x, test_cap, test_pos, test_chunk, test_dict]


    # pad the sequences with zero
    for key, value in dataSet.items():
        for i in range(len(value)):
            dataSet[key][i] = pad_sequences(value[i], maxlen=sentence_maxlen, padding='post')

    # pad the char sequences with zero list
    for j in range(len(test_char)):
        test_char[j] = test_char[j][:sentence_maxlen]
        if len(test_char[j]) < sentence_maxlen:
            test_char[j].extend(np.asarray([[0] * word_maxlen] * (sentence_maxlen - len(test_char[j]))))

    new_test_elmo = []
    for seq in test_elmo:
        new_seq = []
        for i in range(sentence_maxlen):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_test_elmo.append(new_seq)
    test_elmo = np.array(new_test_elmo)

    dataSet['test'].insert(1, np.asarray(test_char))
    dataSet['test'].append(test_elmo)
    test_y = pad_sequences(test_y, maxlen=sentence_maxlen, padding='post')

    end2 = len(dataSet['test'][0]) // batch_size
    for i in range(len(dataSet['test'])):
        dataSet['test'][i] = dataSet['test'][i][:end2 * batch_size]

    dataSet['test'][1] = np.array(dataSet['test'][1])

    print(np.array(test_x).shape)  # (4528, )
    print(np.asarray(test_char).shape)     # (4528, 455, 21)
    print(test_y.shape)    # (4528, 455, 5)
    print('create test set done!\n')
    return dataSet


def main(ned_model=None, prob=5.5):
    root = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/'
    if not os.path.exists(root + 'result/predictions.pkl'):
        dataSet = getTestData()

        model = load_model('model/Model_4_75.00.h5')

        # # 报错
        # import importlib
        # m = importlib.import_module("3_nnet_trainer")
        # model = m.buildModel()
        # path = '/home/administrator/PycharmProjects/keras_bc6_track1/sample/model/Model_Best.h5'
        # model.load_weights(path)

        print('加载模型成功!!')

        predictions = model.predict(dataSet['test'])    # 预测
        y_pred = predictions.argmax(axis=-1)

        with open(root + 'result/predictions.pkl', "wb") as f:
            pkl.dump((y_pred), f, -1)
    else:
        # with open(root + 'result/predictions_76.31.pkl', "rb") as f:
        with open(root + 'result/predictions.pkl', "rb") as f:
            y_pred = pkl.load(f)

    # 对实体预测结果y_pred进行链接，以特定格式写入XML文件
    writeOutputToFile(r'/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test.final.txt', y_pred, ned_model, prob)

    '''
    利用scorer进行评估
    python bioid_score.py --verbose 1 --force \
    存放结果文件的目录 正确答案所在的目录 system1:预测结果所在的目录
    '''


if __name__ == '__main__':

    main()