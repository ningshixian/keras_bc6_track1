'''
http://people.duke.edu/~ccc14/pcfb/biopython/BiopythonEntrez.html
'''

from Bio import Entrez

# # What databases do I have access to?
# # ['pubmed', 'protein', 'gene', 'mesh', 'unigene']
# handle = Entrez.einfo()
# record = Entrez.read(handle)
# print(record["DbList"])

# # Info about a database
# handle = Entrez.einfo(db="gene")
# record = Entrez.read(handle)
# print(record["DbInfo"]["Description"])
# print(record["DbInfo"]["Count"])

# search a db for a given term
# Spaces may be replaced by '+' signs.
handle = Entrez.esearch(db="gene", sort='relevance', term="CHD8")
record = Entrez.read(handle)
print(record["IdList"])
print(record["Count"])

# sort: ‘relevance’ and ‘name’ for Gene
handle = Entrez.esearch(db="gene", idtype="acc", sort='relevance', term="miR-517a")
record = Entrez.read(handle)
print(record["IdList"])
print(record["Count"])

