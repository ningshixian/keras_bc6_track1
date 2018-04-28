
from bioservices.uniprot import UniProt
import pytest

uniprot = UniProt(verbose=False, cache=False)
uniprot.logging.level = "ERROR"

# uniprot.search('zap70+AND+organism:9606', frmt='list')
# print(uniprot.search("zap70+and+taxonomy:9606", frmt="tab", limit=3,
#             columns="entry name,length,id, genes, genes(PREFERRED), interpro, interactor"))
print(uniprot.search("zap70+and+taxonomy:9606", frmt="tab", limit=3,
            columns="entry name, comment(FUNCTION)"))
print(uniprot.search("zap70+and+taxonomy:9606", frmt="tab", limit=3,
            columns="entry name, comment(DOMAIN)"))
# uniprot.search("ZAP70_HUMAN", frmt="tab", columns="sequence", limit=1)
# uniprot.quick_search("ZAP70")