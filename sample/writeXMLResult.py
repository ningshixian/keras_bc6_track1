import datetime
import xml.dom.minidom
import Levenshtein  # pip install python-Levenshtein
from tqdm import tqdm
from helpers import makeEasyTag, Indent, convert_2_BIO
import os
from xml.dom.minidom import parse
import xml.dom.minidom
import codecs
from math import ceil


csv_path = r'/home/administrator/桌面/BC6_Track1/BioIDtraining_2/annotations.csv'


def getXlsxData(path):
    from openpyxl import load_workbook

    wb = load_workbook(path)  # 加载一个工作簿
    sheets = wb.get_sheet_names()  # 获取各个sheet的名字
    sheet0 = sheets[0]  # 第一个表格的名称
    ws = wb.get_sheet_by_name(sheet0)  # 获取特定的 worksheet

    # 获取表格所有行和列，两者都是可迭代的
    rows = ws.rows
    # columns = ws.columns

    # 行迭代
    content = []
    for row in rows:
        line = [col.value for col in row]
        content.append(line)


def getCSVData(path):
    import csv
    entity2id = {}
    with open(path) as f:
        # f_csv = csv.reader(f)
        # headers = next(f_csv)
        # print(headers)
        f_csv = csv.DictReader(f)
        for row in f_csv:
            entity2id[row['text']] = row['obj']
    return entity2id

entity2id = getCSVData(csv_path)


def writeOutputToFile(sentences, predLabels, path):
    """
    写入预测结果至XML文件
    <collection>
        <source>SourceData</source>
        <date>00000000</date>
        <key>sourcedata.key</key>
        <document>
            <id>1-A</id>
            <infon key="sourcedata_document">?</infon>
            <infon key="doi">?</infon>
            <infon key="pmc_id">?</infon>
            <infon key="figure">?</infon>
            <infon key="sourcedata_figure_dir">?</infon>
            <passage>
                <offset>0</offset>
                <text>XXX</text>
                <annotation id="1">
                    <infon key="type">GO:0005764</infon>
                    <infon key="sourcedata_figure_annot_id">1</infon>
                    <infon key="sourcedata_article_annot_id">1</infon>
                    <location offset="16" length="9"/>
                    <text>lysosomes</text>
                </annotation>
            </passage>
        </document>
    """
    # # 读取基因的 ID词典
    # gene_dic = self.readGeneLexicon(config.LEXICON_FILE)


    datasDic = []
    with open(path, encoding='utf-8') as f:
        data_sen = []
        for line in f:
            if line == '\n':
                datasDic.append(data_sen)  # ' '.join()
                data_sen = []
            else:
                token = line.replace('\n', '').split('\t')
                word = token[0]
                data_sen.append(word)


    maxmax = ceil(len(sentences) * 0.8)
    print(maxmax)

    BioC_PATH = r'/home/administrator/桌面/BC6_Track1/BioIDtraining_2/caption_bioc'
    result_path = r'/home/administrator/桌面/BC6_Track1/BioIDtraining_2/train/result'
    files = os.listdir(BioC_PATH)  # 得到文件夹下的所有文件名称
    files.sort()

    num_sentence = 0
    # f = codecs.open('new_valid.txt', 'w', encoding='utf-8')

    for j in tqdm(range(len(files))):  # 遍历文件夹
        file = files[j]
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = BioC_PATH + "/" + file
            try:
                DOMTree = parse(f)  # 使用minidom解析器打开 XML 文档
                collection = DOMTree.documentElement  # 得到了根元素对象
                # print('结点名字:{}\n'.format(collection.nodeName),
                #       '结点的值，只对文本结点有效:{}\n'.format(collection.nodeValue),
                #       '结点的类型:{}'.format(collection.nodeType))
            except:
                print(f)
                continue

            '''
            获得子标签，使用getElementsByTagName方法获取
            获得标签属性值，getAttribute方法可以获得元素的属性所对应的值。
            获得标签对之间的数据，firstChild.data返回被选节点的第一个子节点的数据
            '''
            source = collection.getElementsByTagName("source")[0].childNodes[0].data    # .firstChild.data
            date = collection.getElementsByTagName("date")[0].childNodes[0].data
            key = collection.getElementsByTagName("key")[0].childNodes[0].data

            # 一、生成dom对象，根元素名collection
            impl = xml.dom.minidom.getDOMImplementation()
            dom = impl.createDocument(None, 'collection', None)   # 创建DOM文档对象
            root = dom.documentElement       # 创建根元素

            source = makeEasyTag(dom, 'source', source)
            date = makeEasyTag(dom, 'date', datetime.datetime.now().strftime('%Y-%m-%d'))
            key = makeEasyTag(dom, 'key', key)

            # 给根节点添加子节点
            root.appendChild(source)
            root.appendChild(date)
            root.appendChild(key)

            # 在集合中获取所有 document 的内容
            documents = collection.getElementsByTagName("document")
            for doc in documents:
                id = doc.getElementsByTagName("id")[0].childNodes[0].data
                sourcedata_document = doc.getElementsByTagName("infon")[0].childNodes[0].data
                doi = doc.getElementsByTagName("infon")[1].childNodes[0].data
                pmc_id = doc.getElementsByTagName("infon")[2].childNodes[0].data
                figure = doc.getElementsByTagName("infon")[3].childNodes[0].data
                sourcedata_figure_dir = doc.getElementsByTagName("infon")[4].childNodes[0].data

                passages = doc.getElementsByTagName("passage")
                for passage in passages:
                    text = passage.getElementsByTagName('text')[0].childNodes[0].data
                    annotationSet = []
                    if num_sentence < maxmax:   # 取验证集的数据集
                        continue
                    else:
                        print('当前第{}个句子'.format(num_sentence))
                        # 每读取一篇passage，记录识别实体
                        document = dom.createElement('document')
                        id_node = makeEasyTag(dom, 'id', str(id))
                        s_d_node = makeEasyTag(dom, 'infon', str(sourcedata_document))
                        doi_node = makeEasyTag(dom, 'infon', str(doi))
                        pmc_id_node = makeEasyTag(dom, 'infon', str(pmc_id))
                        figure_node = makeEasyTag(dom, 'infon', str(figure))
                        s_f_d_node = makeEasyTag(dom, 'infon', str(sourcedata_figure_dir))
                        s_d_node.setAttribute('key', 'sourcedata_document')  # 向元素中加入属性
                        doi_node.setAttribute('key', 'doi')  # 向元素中加入属性
                        pmc_id_node.setAttribute('key', 'pmc_id')  # 向元素中加入属性
                        figure_node.setAttribute('key', 'figure')  # 向元素中加入属性
                        s_f_d_node.setAttribute('key', 'sourcedata_figure_dir')  # 向元素中加入属性
                        document.appendChild(id_node)
                        document.appendChild(s_d_node)
                        document.appendChild(doi_node)
                        document.appendChild(pmc_id_node)
                        document.appendChild(figure_node)
                        document.appendChild(s_f_d_node)

                        passage1 = dom.createElement('passage')
                        offset1 = makeEasyTag(dom, 'offset', '0')
                        text1 = makeEasyTag(dom, 'text', text)

                        # 根据预测结果来抽取句子中的所有实体，放入 entities
                        s = datasDic[num_sentence][:180]    # 句子截断长度
                        sen = ' '.join(datasDic[num_sentence][:180])    # 用于查找实体的索引
                        idx = num_sentence - maxmax
                        entities = []
                        id_list = []
                        result=''
                        prex = 0
                        for tokenIdx in range(len(s)):
                            label = predLabels[idx][tokenIdx]
                            word = s[tokenIdx]
                            if label == 1:
                                if result:
                                    entities.append(result.strip())
                                    result = ''
                                prex = label
                                result = word + ' '
                            elif label == 2:
                                if prex==1:
                                    result += word + ' '
                            elif label == 3:
                                if result:
                                    entities.append(result.strip())
                                    result = ''
                                prex = label
                                result = word + ' '
                            elif label == 4:
                                if prex == 3:
                                    result += word + ' '
                            else:
                                if not result == '':
                                    entities.append(result.strip())
                                    result = ''
                                else:
                                    result = ''
                        if not result == '':
                            entities.append(result.strip())
                        l2 = list(set(entities))    # 去除相同元素
                        entities = sorted(l2, key=entities.index)   # 不改变原list顺序
                        # print(entities)

                        # 对识别的实体进行ID链接
                        for i in range(len(entities)):
                            entity = entities[i].strip()
                            Id = 'None'
                            if entity in entity2id:
                                Id = entity2id.get(entity) # 先查词典
                            else:
                                """
                                若不能精确匹配，
                                模糊匹配--计算 Jaro–Winkler 距离
                                """
                                max_score = 0
                                for key in entity2id.keys():
                                    score = Levenshtein.jaro_winkler(key, entity)
                                    if score > max_score:
                                        max_score = score
                                        max_score_key = key
                                Id = entity2id.get(max_score_key)
                            id_list.append(Id)


                        # 生成标注的结点
                        for i in range(len(entities)):
                            entity = entities[i]
                            annotation = dom.createElement('annotation')
                            annotation.setAttribute('id', str(i+1))
                            infon1 = makeEasyTag(dom, 'infon', id_list[i])
                            infon1.setAttribute('key', 'type')
                            infon2 = makeEasyTag(dom, 'infon', str(i+1))
                            infon2.setAttribute('key', 'sourcedata_figure_annot_id')
                            infon3 = makeEasyTag(dom, 'infon', str(i+1))
                            infon3.setAttribute('key', 'sourcedata_article_annot_id')
                            location = dom.createElement('location')
                            try:
                                # 存在实体找不到的情况??
                                location.setAttribute('offset', str(sen.index(entity)))
                            except:
                                print(entity, s)
                                continue
                            location.setAttribute('length', str(len(entity)))
                            text = makeEasyTag(dom, 'text', entity)
                            annotation.appendChild(infon1)
                            annotation.appendChild(infon2)
                            annotation.appendChild(infon3)
                            annotation.appendChild(location)
                            annotation.appendChild(text)
                            annotationSet.append(annotation)

                        # 最后串到根结点上，形成一棵树
                        passage1.appendChild(offset1)
                        passage1.appendChild(text1)
                        for annotation in annotationSet:
                            passage1.appendChild(annotation)

                        # 给根节点添加子节点
                        document.appendChild(passage1)
                        root.appendChild(document)

                    num_sentence += 1

            if num_sentence >= maxmax:  # 取验证集的数据集
                '''将DOM对象doc写入文件...'''
                Indent(dom, dom.documentElement) # 美化
                outputName = result_path + '/' + file
                f = open(outputName, 'w')
                writer = codecs.lookup('utf-8')[3](f)
                dom.writexml(f, indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')
                writer.close()
                f.close()

    print('完结撒花')
