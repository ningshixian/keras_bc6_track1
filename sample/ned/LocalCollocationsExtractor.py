import codecs
import os
import csv
import pickle as pkl
import string
from tqdm import tqdm
import numpy as np
from collections import defaultdict


def LocalCollocationsExtractor(id, offset, length, tokens):
    '''

    :param id: 实体的索引位置
    :param offset: 实体包含的词数
    :param length: 句子长度
    :param tokens: 句子
    :return:
    '''
    features = []
    DEFAULT_COLLOCATIONS = ["-2,-2", "-1,-1", "1,1", "2,2", "-2,-1", "-1,1", "1,2", "-3,-1", "-2,1", "-1,2", "1,3"]
    for collocation in DEFAULT_COLLOCATIONS:
        indexes = collocation.split(",")
        i = int(indexes[0])
        j = int(indexes[1])

        if (i <= j):
            name = ""
            start = id + i
            end = id + j

            for k in range(start, end+1):
                if k != id:
                    index = k+offset-1 if k>id else k

                    if index > -1 and index < length:
                        token = tokens[index]
                    else:
                        token = '0'

                    name += " " + str(token)

            features.append(name.strip())

        else:
            print('Invalid Local Collocations')

    return features


if __name__ == '__main__':
    '''
    收集训练集中Local Collocations字典
    '''

    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/train.pkl', "rb") as f:
        train_x, train_y, train_char, train_cap, train_pos, train_chunk = pkl.load(f)
    train_label_list = []
    for y in train_y:
        y = np.array(y).argmax(axis=-1)
        train_label_list.append(y)

    with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test.pkl', "rb") as f:
        test_x, test_y, test_char, test_cap, test_pos, test_chunk = pkl.load(f)
    test_label_list = []
    for y in test_y:
        y = np.array(y).argmax(axis=-1)
        test_label_list.append(y)


    prex = 0
    features_dict = defaultdict()
    DEFAULT_COLLOCATIONS = ["-2,-2", "-1,-1", "1,1", "2,2", "-2,-1", "-1,1", "1,2", "-3,-1", "-2,1", "-1,2", "1,3"]

    print('\n开始获取SVM特征...')
    for senIdx in tqdm(range(len(train_x))):
        x_sen = train_x[senIdx]
        y_sen = train_label_list[senIdx]

        # for item in DEFAULT_COLLOCATIONS:
        #     features_dict[item] = set()

        if not len(x_sen) == len(y_sen):
            print(x_sen, y_sen)
            print(len(x_sen), len(y_sen))
            print(senIdx)
        assert len(x_sen) == len(y_sen)

        entity = ''
        for j in range(len(x_sen)):
            wordId = x_sen[j]
            label = y_sen[j]
            if label == 1 or label==0:
                if entity:
                    entity = entity.strip()
                    position = j - len(entity.split())
                    features = LocalCollocationsExtractor(position, len(entity.split()), len(x_sen), x_sen)
                    if entity not in features_dict:
                        features_dict[entity] = {}
                        for item in DEFAULT_COLLOCATIONS:
                            features_dict[entity][item] = set()
                    for i in range(len(features)):
                        fea = features[i]
                        features_dict[entity][DEFAULT_COLLOCATIONS[i]].add(fea)
                    entity=''
                if label==1:
                    entity = str(wordId) + ' '
            else:
                if prex == 1 or prex == 2:
                    entity += str(wordId) + ' '
                else:
                    print('标签错误！跳过 {}-->{}'.format(prex, label))
            prex = label

        if not entity == '':
            entity = entity.strip()
            position = j - len(entity.split())
            features = LocalCollocationsExtractor(position, len(entity.split()), len(x_sen), x_sen)
            if entity not in features_dict:
                features_dict[entity] = {}
                for item in DEFAULT_COLLOCATIONS:
                    features_dict[entity][item] = set()
            for i in range(len(features)):
                fea = features[i]
                features_dict[entity][DEFAULT_COLLOCATIONS[i]].add(fea)
            entity = ''

        # if senIdx==0:
        #     for key,value in features_dict.items():
        #         print(key, value)
        #         print('\n')



    for senIdx in tqdm(range(len(test_x))):
        x_sen = train_x[senIdx]
        y_sen = train_label_list[senIdx]

        # for item in DEFAULT_COLLOCATIONS:
        #     features_dict[item] = set()

        assert len(x_sen) == len(y_sen)

        entity = ''
        for j in range(len(x_sen)):
            wordId = x_sen[j]
            label = y_sen[j]
            if label == 1 or label==0:
                if entity:
                    entity = entity.strip()
                    position = j - len(entity.split())
                    features = LocalCollocationsExtractor(position, len(entity.split()), len(x_sen), x_sen)
                    if entity not in features_dict:
                        features_dict[entity] = {}
                        for item in DEFAULT_COLLOCATIONS:
                            features_dict[entity][item] = set()
                    for i in range(len(features)):
                        fea = features[i]
                        features_dict[entity][DEFAULT_COLLOCATIONS[i]].add(fea)
                    entity=''
                if label==1:
                    entity = str(wordId) + ' '
            else:
                if prex == 1 or prex == 2:
                    entity += str(wordId) + ' '
                else:
                    print('标签错误！跳过 {}-->{}'.format(prex, label))
            prex = label

        if not entity == '':
            entity = entity.strip()
            position = j - len(entity.split())
            features = LocalCollocationsExtractor(position, len(entity.split()), len(x_sen), x_sen)
            if entity not in features_dict:
                features_dict[entity] = {}
                for item in DEFAULT_COLLOCATIONS:
                    features_dict[entity][item] = set()
            for i in range(len(features)):
                fea = features[i]
                features_dict[entity][DEFAULT_COLLOCATIONS[i]].add(fea)
            entity = ''


    with open('data/LocalCollocations.pkl', 'wb') as f:
        pkl.dump((features_dict), f, -1)

    # with open('data/LocalCollocations.pkl', 'rb') as f:
    #     features_dict = pkl.load(f)
    #
    # print(features_dict.keys())
    # print(features_dict.get('26281'))