import string
from tqdm import tqdm

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



#
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


'''词典的key区分大小写'''
# text = 'df'
# text2 = 'DF'
# entity2id = {}
# if text not in entity2id:
#     entity2id[text] = []
# if 'obj' not in entity2id[text]:
#     entity2id[text].append('obj')
# if text2 not in entity2id:
#     entity2id[text2] = []
# if 'obj' not in entity2id[text2]:
#     entity2id[text2].append('obj')
# print(entity2id)


# a='df df '
# print(a.split())

# dd = 'df / dd'
# print(dd.split('/'))
#
# for id in ['a', 'b']:
#     if 'b' == id:
#         print(id)
# else:
#     print('df')

# '''比较两个array是否相同'''
# import numpy as np
# a = np.array([1,2,3])
# b = np.array([1,2,4])
# print((a==b).all())


'''
#tax_id	GeneID	Symbol	LocusTag	Synonyms	dbXrefs	chromosome	map_location	description	type_of_gene	Symbol_from_nomenclature_authority	Full_name_from_nomenclature_authority	Nomenclature_status	Other_designations	Modification_date	Feature_type
7	5692769	NEWENTRY	-	-	-	-	-	Record to support submission of GeneRIFs for a gene not in Gene (Azotirhizobium caulinodans.  Use when strain, subtype, isolate, etc. is unspecified, or when different from all specified ones in Gene.).	other	-	-	-	-	20171118	-
9	1246500	repA1	pLeuDn_01	-	-	-	-	putative replication-associated protein	protein-coding	-	-	-	-	20180129	-

'''
# gene_path_new = '/media/administrator/疯狂的大菠萝/B910/AutoExtend-master/geneExtract/NCBI-gene-data/gene_info2'
# ff = open(gene_path_new, 'w')
#
# gene_path = '/media/administrator/疯狂的大菠萝/B910/AutoExtend-master/geneExtract/NCBI-gene-data/gene_info'
# with open(gene_path) as f:
#     for line in f:
#         if line.startswith('#'):
#             continue
#         GeneID = line.split('\t')[1]
#         Symbol = line.split('\t')[2]
#         if Symbol=='NEWENTRY':
#             continue
#         ff.write(GeneID + '\t' + Symbol)
#         ff.write('\n')
# ff.close()


#
#
# gene_path_new = '/Volumes/疯狂的大菠萝/B910/AutoExtend-master/proteinExtract/proteinData/uniprot_sprot.dat2'
# ff = open(gene_path_new, 'w')
#
# entity = []
# id = ''
# gene_path = '/Volumes/疯狂的大菠萝/B910/AutoExtend-master/proteinExtract/proteinData/uniprot_sprot.dat'
# with open(gene_path) as f:
#     for line in tqdm(f):
#         if line.startswith('//'):
#             if entity and id:
#                 ff.write(id + '\t' + ';'.join(entity))
#                 ff.write('\n')
#             entity = []
#             id = ''
#         else:
#             if line.startswith('DE'):
#                 if '            ' in line:
#                     splited = line.split('            ')
#                 elif '   ' in line:
#                     splited = line.split('   ')
#                 if splited[1].startswith('RecName: Full='):
#                     idx = splited[1].index('Full=')
#                     item = splited[1].replace('\n', '').strip()[idx + 5:-1]
#                     if '{' in item:
#                         idx = item.index('{')
#                         item = item[:idx]
#                     entity.append(item)
#                 elif splited[1].startswith('Short='):
#                     if id=='P17046':
#                         print(splited[1])
#                     idx = splited[1].index('Short=')
#                     item = splited[1].replace('\n', '').strip()[idx + 6:-1]
#                     if '{' in item:
#                         idx = item.index('{')
#                         item = item[:idx]
#                     entity.append(item)
#                 elif splited[1].startswith('AltName: Full='):
#                     idx = splited[1].index('Full=')
#                     item = splited[1].replace('\n', '').strip()[idx + 5:-1]
#                     if '{' in item:
#                         idx = item.index('{')
#                         item = item[:idx]
#                     entity.append(item)
#                 elif splited[1].startswith('AltName: CD_antigen='):
#                     idx = splited[1].index('CD_antigen=')
#                     item = splited[1].replace('\n', '').strip()[idx + 11:-1]
#                     if '{' in item:
#                         idx = item.index('{')
#                         item = item[:idx]
#                     entity.append(item)
#             elif line.startswith('AC'):
#                 splited = line.split('   ')
#                 id = splited[1].strip('\n')
#                 id = id.replace(' ', '')
#             # elif line.startswith('GN'):
#             #     splited = line.split('   ')
#             #     if splited[1].startswith('ORFNames='):
#             #         idx = splited[1].index('=')
#             #         item = splited[1].strip('\n')[idx + 1:-1]
#             #         if '{' in item:
#             #             idx = item.index('{')
#             #             item = item[:idx]
#             #         entity.append(item)
# ff.close()


# protein2id = {}
# gene2id = {}
# protein_path = '/home/administrator/PycharmProjects/embedding/uniprot_sprot.dat2'
# gene_path = '/home/administrator/PycharmProjects/embedding/gene_info2'
# with open(gene_path) as f:
#     for line in tqdm(f):
#         splited = line.split('\t')
#         id_list = splited[0].split(';')
#         e_list = splited[1].split(';')
#         for e in e_list:
#             if e not in gene2id:
#                 gene2id[e] = []
#             for id in id_list:
#                 if id not in gene2id[e]:
#                     gene2id[e].append(id)
# with open(protein_path) as f:
#     for line in tqdm(f):
#         splited = line.split('\t')
#         id_list = splited[0].split(';')
#         e_list = splited[1].split(';')
#         for e in e_list:
#             if e not in protein2id:
#                 protein2id[e] = []
#             for id in id_list:
#                 if id not in protein2id[e]:
#                     protein2id[e].append(id)
#
# c = 0
# for key,value in protein2id.items():
#     print(key)
#     print(value)
#     c += 1
#     if c>10:
#         break
#
# c = 0
# for key,value in gene2id.items():
#     print(key)
#     print(value)
#     c += 1
#     if c>10:
#         break


# a = []
# p = list(string.punctuation)
# a.extend(p)
# print(a)


# a = {'entity':{}}
# a['entity']['id1']=1
# a['entity']['id2']=1
# a['entity']['id1']+=1
# print(a['entity']['id1'])
# entity2id_new = {}
# for key, value in a.items():
#     value_sorted = sorted(value.items(), key=lambda item: item[1], reverse=True)
#     entity2id_new[key] = [item[0] for item in value_sorted]
# print(entity2id_new)

import re
s='3a4b5cdd7e'
a = re.findall(r'[0-9]+|[a-z]+',s)
print(' '.join(a))


da = []
for item in ['a', 'b', 'c']:
    if item=='b':
        da.append(item)
        break




def getCSVData(csv_path, entity2id):
    '''
    获取实体ID词典 {'entity':[id1, id2, ...]}
    实体区分大小写
    '''
    with open(csv_path) as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            id = row['obj']
            entity = row['text']
            # text = row['text'].lower()
            if id.startswith('NCBI gene:') or id.startswith('Uniprot:') or \
                    id.startswith('gene:') or id.startswith('protein:'):
                if entity not in entity2id:
                    entity2id[entity] = []
                if id not in entity2id[entity]:
                    entity2id[entity].append(id)
        print('entity2id字典总长度：{}'.format(len(entity2id)))  # 5096

    num_word_multiID = 0
    entity2id_new = entity2id.copy()
    # 拓展实体词典
    for key, value in entity2id.items():
        if len(value)>1:
            num_word_multiID+=1
        for char in string.punctuation:
            if char in key:
                key = key.replace(char, ' ' + char + ' ')
        key = key.strip().replace('  ', ' ')
        if key not in entity2id_new:
            entity2id_new[key] = value
        key = key.strip().replace(' ', '')  # 去掉所有空格
        if key not in entity2id_new:
            entity2id_new[key] = value
    entity2id = {}
    del entity2id
    print('F4/80: {}'.format(entity2id_new['F4/80']))
    print('其中，多ID实体的个数：{}'.format(num_word_multiID))    # 1538
    return entity2id_new