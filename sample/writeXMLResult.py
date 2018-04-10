import datetime
import xml.dom.minidom
import Levenshtein  # pip install python-Levenshtein
from tqdm import tqdm
import config
from util.utils import makeEasyTag, Indent, convert_2_BIO
import codecs


def writeOutputToFile(self, sentences, predLabels, name):
    """
    写入预测结果至XML文件
    :param sentences:
    :param predLabels:
    :param name:
    :return:
    """
    print('\n写入结果...')
    outputName = 'result/' + name + '.xml'
    fOut = open(outputName, 'w')

    # 读取基因的 ID词典
    gene_dic = self.readGeneLexicon(config.LEXICON_FILE)
    # 一、生成dom对象，根元素名collection
    impl = xml.dom.minidom.getDOMImplementation()
    dom = impl.createDocument(None, 'collection', None)
    root = dom.documentElement

    source = makeEasyTag(dom, 'source', 'PubTator')
    date = makeEasyTag(dom, 'date', datetime.datetime.now().strftime('%Y-%m-%d'))
    key = makeEasyTag(dom, 'key', 'collection.key')

    # 给根节点添加子节点
    root.appendChild(source)
    root.appendChild(date)
    root.appendChild(key)

    document = None
    for sentenceIdx in tqdm(range(len(sentences))):
        result = ''
        senten = ''
        offset = []
        label = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            word = sentences[sentenceIdx]['raw_tokens'][tokenIdx]
            senten += (word + ' ')
            predLabel = self.idx2Label[predLabels[sentenceIdx][tokenIdx]]
            label.append(predLabel)

        """ Convert inplace IOBES encoding to BIO encoding """
        label = convert_2_BIO(label)

        """后处理"""
        entities = []
        labelStarted = False
        for tokenIdx in range(len(label)):
            predLabel = label[tokenIdx]
            word = sentences[sentenceIdx]['raw_tokens'][tokenIdx]
            if predLabel == 'B':
                labelStarted = True
                result = word + ' '
                offset.append(tokenIdx)
            elif predLabel == 'O':
                labelStarted = False
                if not result == '':
                    entities.append(result.strip())
                    result = ''
            elif predLabel == 'I':
                if not labelStarted:
                    predLabel = 'O'
                    label[tokenIdx] = 'O'
                    labelStarted = False
                else:
                    result += word + ' '
        if not result == '':
            entities.append(result)

        # assert len(entities)==len(offset)
        if not len(entities) == len(offset):
            print(entities)
            print(offset)

        if document == None:
            document = dom.createElement('document')
        id = makeEasyTag(dom, 'id', sentences[sentenceIdx]['id'])
        passage1 = dom.createElement('passage')
        infon1 = makeEasyTag(dom, 'infon', sentences[sentenceIdx]['infon'])
        infon1.setAttribute('key', 'type')  # 向元素中加入属性
        offset1 = makeEasyTag(dom, 'offset', sentences[sentenceIdx]['offset'])
        text1 = makeEasyTag(dom, 'text', senten.strip())
        annotationSet = []

        for i in range(len(entities)):
            item = entities[i].strip()
            geneId = 'None'
            if item in gene_dic:
                geneId = gene_dic.get(item)
            else:
                """
                若不能精确匹配，
                模糊匹配--计算 Jaro–Winkler 距离
                """
                max_score = 0
                for key in gene_dic.keys():
                    score = Levenshtein.jaro_winkler(key, item)
                    if score > max_score:
                        max_score = score
                        max_score_key = key
                geneId = gene_dic.get(max_score_key)

            # for k, v in gene_dic.iteritems():
            #     if item in v:
            #         geneId = k
            #         break

            annotation = dom.createElement('annotation')
            annotation.setAttribute('id', str(i))
            infon3 = makeEasyTag(dom, 'infon', geneId)
            infon3.setAttribute('key', 'NCBI GENE')
            infon4 = makeEasyTag(dom, 'infon', 'Gene')
            infon4.setAttribute('key', 'type')
            location = dom.createElement('location')
            location.setAttribute('offset', str(offset[i]))
            location.setAttribute('length', str(len(item.split())))
            text = makeEasyTag(dom, 'text', str(item))

            annotation.appendChild(infon3)
            annotation.appendChild(infon4)
            annotation.appendChild(location)
            annotation.appendChild(text)

            annotationSet.append(annotation)

        # 最后串到根结点上，形成一棵树
        passage1.appendChild(infon1)
        passage1.appendChild(offset1)
        passage1.appendChild(text1)
        for annotation in annotationSet:
            passage1.appendChild(annotation)

        if sentenceIdx % 2 == 0:
            document.appendChild(id)
            document.appendChild(passage1)
        else:
            document.appendChild(passage1)
            root.appendChild(document)
            document = None

            # document.appendChild(id)
            # document.appendChild(passage1)
            # root.appendChild(document)

    # 美化
    Indent(dom, dom.documentElement)
    # 写入到XML文件中
    f = file(outputName, 'w')
    writer = codecs.lookup('utf-8')[3](f)
    dom.writexml(writer, encoding='utf-8')
    writer.close()