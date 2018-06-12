'''
用gensim来训练Word2Vec：
1、联合训练语料和测试语料一起训练；
2、经过测试用skip gram效果会好些。
'''
import os
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# def train_again(temp_path):
#     # 增量训练
#     model = gensim.models.Word2Vec.load(temp_path)
#     more_sentences = [
#         ['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more',
#          'sentences']]
#     model.build_vocab(more_sentences, update=True)
#     model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)


def train(data, word_size):
    word2vec = gensim.models.Word2Vec(data,
                                      min_count=1,
                                      size=word_size,
                                      workers=20,
                                      iter=20,
                                      window=5,
                                      negative=8,
                                      sg=1)     # skip-gram算法
    # 保存模型，以便重用
    # word2vec.save('word2vec.model')
    # 以一种C语言可以解析的形式存储词向量
    word2vec.save_word2vec_format("word2vec.model.bin", binary=True)
    word2vec.init_sims(replace=True)  # 预先归一化，使得词向量不受尺度影响
    return word2vec


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
            for line in open(self.dirname):
                # Each sentence a list of words (utf8 strings):
                yield line.strip('\n').split()


path_xml = '/home/administrator/桌面'
corpus_path2 = path_xml + '/' + 'corpus2.txt'

sentences = MySentences(corpus_path2)  # a memory-friendly iterator
model = train(sentences, 50)
print(model['a'])


# 对应的加载方式
# model_2 = gensim.models.Word2Vec.load("word2vec.model")

# model_3 = gensim.models.Word2Vec.load_word2vec_format("word2vec.model.bin", binary=True)