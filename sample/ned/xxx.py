import string

# print(string.punctuation)
# print(string.printable)

# x='34..'
# y=x
# y='34'
# print(x,y)

# if 'DF43'.isupper():
#     print('df')
# if '112'.isdigit():
#     print(112)

# sen = 'A Representative Gli1ZF/DNA structure extrapolated from MD trajectories. Gli1ZF is'
# entity = 'Gli1'
# print(sen.find(entity))
#
# sen = sen.encode('utf-8')
# entity = entity.encode('utf-8')
# print(sen.find(entity))

# a={'a':'a'}
# print(a.get('2'))

# import numpy as np
# a = ['a', 'b']
# b = list(np.array([1,1,1,1]))
# print(b)




# import os
# import csv
# from tqdm import tqdm
# from sample.utils.write_test_result import extract_id_from_res
# import Levenshtein
# from bioservices import UniProt
# u = UniProt()
#
#
# def getCSVData(csv_path):
#     '''
#     获取实体ID词典 'entity':[id1, id2, ...]
#     实体全部小写!!
#     只用到gene和protein类别的部分
#     '''
#     num_word_multiID = 0
#     entity2id_new = {}
#     entity2id = {}
#     with open(csv_path) as f:
#         f_csv = csv.DictReader(f)
#         for row in f_csv:
#             if row['obj'].startswith('NCBI gene:') or \
#                     row['obj'].startswith('Uniprot:'):
#
#                 text = row['text'].lower()
#                 if text not in entity2id:
#                     entity2id[text] = []
#                 if row['obj'] not in entity2id[text]:
#                     entity2id[text].append(row['obj'])
#                 # entity2id[row['text']] = list(set(entity2id[row['text']]))
#         print('entity2id字典总长度：{}'.format(len(entity2id)))   # 4221
#     return entity2id
#
#
#
# def search(entity):
#     id_list = []
#     # 词典精确匹配
#     if entity.lower() in entity2id:
#         id_list.extend(entity2id[entity.lower()])
#         return id_list
#
#     # 数据库API查询1-reviewed
#     res_reviewed = u.search(entity + '+reviewed:yes', frmt="tab", columns="id", limit=3)
#     if res_reviewed == 400:
#         print('请求无效\n')
#         return id_list
#     elif res_reviewed:  # 若是有返回结果
#         Ids = extract_id_from_res(res_reviewed)
#         for item in Ids:
#             id_list.extend(['Uniprot:' + item])
#         entity2id[entity.lower()] = id_list  # 将未登录实体添加到实体ID词典中
#         return id_list
#     else:
#         return id_list
#
#
# def getData(g2e, p2e):
#     '''
#     获取实体ID词典 'entity':[id1, id2, ...]
#     实体全部小写!!
#     只用到gene和protein类别的部分
#     '''
#     csv_list = []
#     with open(csv_path) as f:
#         f_csv = csv.DictReader(f)   # 102717 line
#         for row in f_csv:
#             csv_list.append(row)
#         for i in tqdm(range(len(csv_list))):
#             row = csv_list[i]
#             entity = row['text'].lower()
#
#             if row['obj'].startswith('NCBI gene:'):
#                 ID = row['obj'].split('|')[0] if '|' in row['obj'] else row['obj']
#                 ID = ID.split(':')[1]
#                 if ID not in g2e:
#                     g2e[ID] = []
#                 if entity not in g2e[ID]:
#                     g2e[ID].append(entity)
#
#             if row['obj'].startswith('Uniprot:'):
#                 ID = row['obj'].split('|')[0] if '|' in row['obj'] else row['obj']
#                 ID = ID.split(':')[1]
#                 if ID not in p2e:
#                     p2e[ID] = []
#                 if entity not in p2e[ID]:
#                     p2e[ID].append(entity)
#
#             id_list = search(entity)
#             # print(id_list)
#             for id in id_list:
#                 id = id.split('|')[0] if '|' in id else id
#                 id = id.split(':')[1]
#                 if id.isdigit():
#                     if id not in g2e:
#                         g2e[id] = []
#                     if entity not in g2e[id]:
#                         g2e[id].append(entity)
#                 else:
#                     if id not in p2e:
#                         p2e[id] = []
#                     if entity not in p2e[id]:
#                         p2e[id].append(entity)
#
#
#
# g2e = {}
# p2e = {}
# base = r'/home/administrator/桌面/BC6_Track1'
# BioC_path = base + '/' + 'BioIDtraining_2/caption_bioc'
# BioC_path_test = base + '/' + 'test_corpus_20170804/caption_bioc/'
# csv_path = base + '/' + 'BioIDtraining_2/annotations.csv'   # 实体ID查找词典文件
#
# entity2id = getCSVData(csv_path)    # 读取实体ID查找词典
# result = getData(g2e, p2e)
# # getID(BioC_path, gene2entity, prot2entity)
# # getID(BioC_path_test, gene2entity, prot2entity)
#
# with open('gene.txt', 'w') as f:
#     for key,value in g2e.items():
#         f.write('{} {}'.format(key, '::,'.join(value)))
#         f.write('\n')
#
# with open('protein.txt', 'w') as f:
#     for key,value in p2e.items():
#         f.write('{} {}'.format(key, '::,'.join(value)))
#         f.write('\n')


a = ['a']
print(a[1:])