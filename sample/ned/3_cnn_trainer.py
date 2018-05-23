'''
利用CNN计算候选candidate与实体mention（质心？）的相似度
并对所有<m,c1>...<m,cx>的得分进行排序 ranking
得分最高者作为mention的id

组成：
semantic representation layer
convolution layer
pooling layer
concatenation layer (Vm + Vc + Vsim)    Vsim=Vm·M·Vc
hidden lyaer
softmax layer

参考 BMC Bioinformatics
《CNN-based ranking for biomedical entity normalization》
'''