'''
Trying to get Uniprot ID from Entrez Gene ID with Python script (solved)

python示例代码: https://www.uniprot.org/help/api_idmapping
UniProtKB 列名columns对照获取: http://www.uniprot.org/help/uniprotkb_column_names
Mapping From:to 含义解释: http://www.uniprot.org/help/api_idmapping

问题
A lot of the gene names used in microarray are synonyms, for example AOF2, which in RNA-seq is KDM1A.
'''
import requests


def getHtmlText(url, params=None, headers=None):
    # requests抓取网页的通用框架
    try:
        r = requests.get(url, timeout=30, params=params, headers=headers)
        # 如果状态码不是200 则应发HTTOError异常
        r.raise_for_status()
        # 设置正确的编码方式
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return "Something Wrong!"



def getIdFromApi(entity):

    # url = 'http://www.uniprot.org/uniprot/'
    # params = {
    # 'format':'tab',
    # 'columns':'id',
    # 'query':'reviewed:yes+AND+' + entity,
    # }

    url = 'http://www.uniprot.org/mapping/'
    params = {
        'from': 'GENENAME',  # P_ENTREZGENEID
        'to': 'ACC',
        'format': 'tab',
        'query': entity,  # Uniprot:P10636  NCBI gene:17762
        'fil': 'reviewed3%Ayes', }

    contact = "# Please set your email address here to help us debug in case of problems."
    hd = {'User-agent': 'Python %s' % contact}

    results = getHtmlText(url, params=params, headers=hd)
    # print(len(results))
    ids = []
    results = results.split('\n')[1:-1]     # 去除开头一行和最后的''
    for line in results:
        id = line.split('\t')[1]
        ids.append(id)
    return ids


if __name__ == '__main__':
    ids = getIdFromApi('tau')
    print(ids)