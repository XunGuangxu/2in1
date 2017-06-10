import numpy as np
import scipy.stats
DATA_DIR = '/media/O_o/UB/research/dataset/20newsgroups/CoEmbedding/'


vocab_pt = DATA_DIR + 'vocab.txt'
#embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/results/Embeddings_K50_iter19.npz'
embedding_pt = DATA_DIR + 'results_parallel/Embeddings_K50_iter19.npz'

'''
vocab_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/vocab.txt'
embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/svd.npz'
'''
'''
vocab_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/vocab.txt'
embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/w2v_cbow_iter20.npz'
'''
'''
vocab_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/GloVe/vocab.txt'
embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/GloVe/glove.npz'
'''
test_pt = '/media/O_o/UB/research/dataset/embeddingTestSet/ws/ws353.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/ws353_relatedness.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/ws353_similarity.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/bruni_men.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/radinsky_mturk.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/simlex_999a.txt'
#test_pt = 'D:/UB/research/dataset/embeddingTestSet/ws/luong_rare.txt'

w2id = {}
id2w = {}
list_our = []
list_ground = []

def _read_vocab(voca_pt):
    for l in open(voca_pt):
        ws = l.strip().split()
        w2id[ws[1]] = int(ws[0])
        id2w[int(ws[0])] = ws[1]

def most_similar(dword, topn):
    wid = w2id[dword]
    wvec = gamma[wid]
    distances = np.inner(-wvec, gamma)
    most_extreme = np.argpartition(distances, topn)[:topn]
    #print(np.sort(distances.take(most_extreme)))
    return [id2w[t] for t in most_extreme.take(np.argsort(distances.take(most_extreme)))]

def cosine_distance(word1, word2):
    wid1 = w2id[word1]
    wid2 = w2id[word2]
    wvec1 = gamma[wid1]
    wvec2 = gamma[wid2]
    return np.inner(wvec1, wvec2)

data = np.load(embedding_pt)
gamma = data['C']
data.close()
print(gamma.shape)
_read_vocab(vocab_pt)
#normalize gamma
normss = np.linalg.norm(gamma, axis = 1, keepdims = True)
gamma = gamma/normss

for line in open(test_pt):
    ws = line.strip().split()
    if ws[0] in w2id and ws[1] in w2id:
        print(line + '\t' + str(cosine_distance(ws[0], ws[1])))
        list_ground.append(float(ws[2]))
        list_our.append(cosine_distance(ws[0], ws[1]))

rank_our = scipy.stats.rankdata(list_our)
rank_ground = scipy.stats.rankdata(list_ground)
print(rank_our)
print(rank_ground)
print(len(rank_ground))
result = scipy.stats.spearmanr(rank_ground, rank_our)
print(result)


print(most_similar('car', 10))
print(most_similar('jesus', 10))
print(most_similar('hockey', 10))
print(cosine_distance('car', 'cars'))
print(cosine_distance('car', 'medical'))
