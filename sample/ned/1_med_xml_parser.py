import pubmed_parser as pp
from tqdm import tqdm

'''
从XML大文件中，抽取相应的摘要和题目
'''

# PMID = []
# ArticleTitle = []
# AbstractText = []
# path_xml = '/home/administrator/桌面'
# with open(path_xml + '/' + 'corpus.txt', 'w') as f:
#     pubmed_list = pp.parse_medline_xml(path_xml + '/' + 'pubmed_result_.xml',
#                                        year_info_only=False,
#                                        nlm_category=False)
#     for i in tqdm(range(len(pubmed_list))):
#         pubmed_dict = pubmed_list[i]
#         # PMID.append(pubmed_dict['pmid'])
#         # ArticleTitle.append(pubmed_dict['title'])
#         # AbstractText.append(pubmed_dict['abstract'])
#         f.write(pubmed_dict['title'] + '\n')
#         f.write(pubmed_dict['abstract'] + '\n')



# from xml.dom.minidom import parse
#
# path_xml = '/home/administrator/桌面'
# wf = open(path_xml + '/' + 'corpus.txt', 'w')
#
# f = '/home/administrator/桌面/pubmed_result.xml'
# DOMTree = parse(f)  # 使用minidom解析器打开 XML 文档
# collection = DOMTree.documentElement  # 得到了根元素对象
#
# # 在集合中获取所有 document 的内容
# documents = collection.getElementsByTagName("PubmedArticle")
# for i in tqdm(range(documents)):
#     doc = documents[i]
#     MedlineCitation = doc.getElementsByTagName("MedlineCitation")[0].childNodes[0].data
#     Article = MedlineCitation.getElementsByTagName("Article")[0].childNodes[0].data
#     ArticleTitle = Article.getElementsByTagName("ArticleTitle")[0].childNodes[0].data
#     Abstract = Article.getElementsByTagName("Abstract")[0]
#     AbstractTxt = Abstract.getElementsByTagName("AbstractText")[0].childNodes[0].data
#     wf.write(AbstractTxt)
#     wf.write('\n')
#
# wf.close()



# path_xml = '/home/administrator/桌面'
# out = open(path_xml + '/' + 'corpus.txt', 'w')
#
#
# from lxml import etree
# infile = '/home/administrator/桌面/pubmed_gene_protein.xml'
#
# '''
# xml 的 iterparse 方法是 ElementTree API 的扩展。
# iterparse 为所选的元素上下文返回一个 Python 迭代器。
# 它接受两个有用的参数：要监视的事件元组和标记名。
# 在本例中，我只对 <AbstractText> 的文本内容感兴趣（达到 end 事件即可获得）。
# https://www.ibm.com/developerworks/cn/xml/x-hiperfparse/
# '''
# context = etree.iterparse(infile, events=('end',), tag='AbstractText')
#
# count = 0
# for event, elem in context:
#     out.write('%s\n' % elem.text)
#
#     # It's safe to call clear() here because no descendants will be accessed
#     elem.clear()
#
#     # Also eliminate now-empty references from the root node to <Title>
#     while elem.getprevious() is not None:
#         del elem.getparent()[0]
#
#     count+=1
#     if count%20000==0:
#         print(count)
#         out.close()
#         out = open(path_xml + '/' + 'corpus.txt', 'a')
#
# del context
# out.close()



try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

'''
解析XML文件首选ET！
ET可以将XML文档加载为保存在内存里的树（in-memory tree），然后再进行处理。
但是在解析大文件时，会出现和内存消耗大的问题。
为了解决这个问题，ET提供了一个类似SAX的特殊工具——iterparse，可以循序地解析XML。
通过调用elem.clear()，废弃掉不需要的元素，就相当于废弃了整个树，释放出系统分配的内存。
'''

count = 0
tag = 0
path_xml = '/home/administrator/桌面'
infile = '/home/administrator/桌面/pubmed_result.xml'
corpus_path = path_xml + '/' + 'corpus.txt'
corpus_path2 = path_xml + '/' + 'corpus2.txt'

# out = open(corpus_path, 'w')
#
# for event, elem in ET.iterparse(infile, events=('end',)):   # 注意这里只使用end进行触发即可
#     count += 1
#     if count%100000==0:
#         print(count)
#     if elem.tag=='Pagination':
#         tag=1
#     elif elem.tag=='VernacularTitle':
#         tag=0
#
#     elif elem.tag == 'ArticleTitle':
#         if elem.text:
#             if elem.text.startswith('[') and elem.text.endswith(']'):
#                 title = elem.text[1:-1]
#             else:
#                 title = elem.text
#             out.write('%s\n' % title)
#     elif elem.tag=='AbstractText':
#         if tag:
#             out.write('%s\n' % elem.text)
#     elem.clear()     # 非常关键：将元素废弃，释放出系统分配的内存。
#
# out.close()


'''
将训练预料和测试预料加入到pubmed摘要中，用于训练词向量
'''
sen_list = []
with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/train_raw.txt') as f:
    for line in f:
        sen_list.append(line)
with open('/home/administrator/PycharmProjects/keras_bc6_track1/sample/data/test_raw.txt') as f:
    for line in f:
        sen_list.append(line)
with open(corpus_path) as f:
    for line in f:
        sen_list.append(line)
with open(corpus_path2, 'w') as f:
    for line in sen_list:
        f.write(line)