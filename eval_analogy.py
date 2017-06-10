import numpy as np
import scipy.stats

'''
vocab_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/vocab.txt'
embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/results/Embeddings_K50_iter19.npz'
#embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/svd_gradient.npz'
'''
'''
vocab_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/vocab.txt'
embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/svd.npz'
'''
'''
vocab_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/vocab.txt'
embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/w2v_cbow_iter20.npz'
'''
vocab_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/GloVe/vocab.txt'
embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/word2vec/GloVe/glove.npz'

#test_pt = 'D:/UB/research/dataset/embeddingTestSet/analogy/google.txt'
test_pt = 'D:/UB/research/dataset/embeddingTestSet/analogy/msr.txt'
#measurement = 'raw'
measurement = '3cosadd'
#measurement = '3cosmul'

w2id = {}
id2w = {}

def _read_vocab(voca_pt):
    for l in open(voca_pt):
        ws = l.strip().split()
        w2id[ws[1]] = int(ws[0])
        id2w[int(ws[0])] = ws[1]

def most_similar_3cosadd(dword1, dword2, dword3):
    wid1, wid2, wid3 = w2id[dword1], w2id[dword2], w2id[dword3]
    wvec = gamma[wid3] - gamma[wid1] + gamma[wid2]
    temp_gamma = gamma.copy()
    temp_gamma[wid1] = temp_gamma[wid2] = temp_gamma[wid3] = 0
    cos_distances = np.inner(wvec, temp_gamma)
    return id2w[np.argmax(cos_distances)]

def most_similar_3cosmul(dword1, dword2, dword3):
    wid1, wid2, wid3 = w2id[dword1], w2id[dword2], w2id[dword3]
    wvec1, wvec2, wvec3 = gamma[wid1], gamma[wid2], gamma[wid3]
    temp_gamma = gamma.copy()
    temp_gamma[wid1] = temp_gamma[wid2] = temp_gamma[wid3] = 0
    cos_dst1, cos_dst2, cos_dst3 = np.inner(wvec1, temp_gamma), np.inner(wvec2, temp_gamma), np.inner(wvec3, temp_gamma)
    similarity = cos_dst3 * cos_dst2 / (cos_dst1 + 1e-3)
    return id2w[np.argmax(similarity)]

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
if measurement != 'raw':
    normss = np.linalg.norm(gamma, axis = 1, keepdims = True)
    gamma = gamma/normss

n_total = n_correct = 0
for line in open(test_pt):
    ws = line.strip().split()
    if ws[0] in w2id and ws[1] in w2id and ws[2] in w2id and ws[3] in w2id:
        n_total = n_total + 1
        if most_similar_3cosadd(ws[0], ws[1], ws[2]) == ws[3]:
            n_correct = n_correct + 1
        '''else:
            print(ws[0], ws[1], ws[2], ws[3])
            print(most_similar_3cosadd(ws[0], ws[1], ws[2]))'''

print(n_correct, n_total, n_correct/n_total)


print(most_similar_3cosadd('london', 'england', 'paris'))
print(most_similar_3cosmul('london', 'england', 'paris'))
print(cosine_distance('car', 'cars'))
print(cosine_distance('car', 'medical'))
