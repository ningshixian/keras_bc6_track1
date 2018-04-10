'''
用gensim来训练Word2Vec：
1、联合训练语料和测试语料一起训练；
2、经过测试用skip gram效果会好些。
'''
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train(data, word_size):
    word2vec = gensim.models.Word2Vec(data,
                                      min_count=1,
                                      size=word_size,
                                      workers=20,
                                      iter=20,
                                      window=8,
                                      negative=8,
                                      sg=1)
    word2vec.save('word2vec_words.model')
    word2vec.init_sims(replace=True)  # 预先归一化，使得词向量不受尺度影响
    return word2vec