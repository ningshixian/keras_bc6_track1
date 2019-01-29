
'''
Trying to get Uniprot ID from Entrez Gene ID with Python script (solved)

python示例代码: https://www.uniprot.org/help/api_idmapping
UniProtKB 列名columns对照获取: http://www.uniprot.org/help/uniprotkb_column_names
Mapping From:to 含义解释: http://www.uniprot.org/help/api_idmapping

问题
A lot of the gene names used in microarray are synonyms, for example AOF2, which in RNA-seq is KDM1A.
'''


# 通过爬虫获取实体的ID
def getIdFromApi(entity):

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


def get_id_from_bioservice(entity):

    from bioservices import UniProt
    u = UniProt(cache=True)

    # 数据库API查询
    Ids = None
    temp = []
    temp_cc = []
    # df = u.get_df(["Q9HCK8"])
    # print(df)
    res_reviewed = u.search(entity + '+reviewed:yes', frmt="tab", columns="id, entry name, genes, comment(FUNCTION)", limit=5)   # , protein names
    res_unreviewed = u.search(entity, frmt="tab", columns="id, entry name, genes, genes(PREFERRED)", limit=5)
    # print(res_reviewed)
    # print(res_unreviewed)

    if res_reviewed == 400:
        print('请求无效\n')
        return Ids

    if res_reviewed:  # 若是有返回结果
        results = res_reviewed.split('\n')[1:-1]  # 去除开头一行和最后的''
        for line in results:
            results = line.split('\t')
            temp.append(results[0])
            temp_cc.append(results[-1])
            # break
    return temp, temp_cc


if __name__ == '__main__':

    # Ptch1 CHDB
    Ids, func = get_id_from_bioservice('snx5')   # Uniprot:P47806|NCBI gene:14632|Uniprot:P08151|NCBI gene:2735
    print(Ids)
    print(func)

    # 2. mapping from/to uniprot identifiers
    # UniProtKB AC: ACC
    # Entrez Gene (GeneID): P_ENTREZGENEID
    # GeneID (Entrez Gene): P_ENTREZGENEID

    # mapper = u.mapping(fr="GENENAME", to="ACC", query='Gli1')
    # print(mapper['Gli1'])
