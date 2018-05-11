'''
用gensim来训练Word2Vec：
1、联合训练语料和测试语料一起训练；
2、经过测试用skip gram效果会好些。
'''
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_again(temp_path):
    # 增量训练
    model = gensim.models.Word2Vec.load(temp_path)
    more_sentences = [
        ['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more',
         'sentences']]
    model.build_vocab(more_sentences, update=True)
    model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)


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
    # new_model=gensim.models.Word2Vec.load('word2vec_words.model')
    return word2vec


path_xml = '/home/administrator/桌面'
corpus_path2 = path_xml + '/' + 'corpus2.txt'

data = []
with open(corpus_path2) as f:
    for line in f:
        data.append(line.strip('\n'))
train(data, 50)