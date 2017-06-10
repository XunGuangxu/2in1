# translate word into id in documents
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import itertools
import os

w2id = {}
DATA_DIR = 'D:/UB/research/dataset/20newsgroups/'

def indexFile(pt, res_pt, matrix_pt, batch_size, window_size):
    print('index file: ', pt)
    wf = open(res_pt, 'w')
    wf_matrix = open(matrix_pt, 'w')
    wf_matrix.writelines('doc_id,word_id')
    docid = 0
    saveid = 0
    rows = []
    cols = []
    for l in open(pt):
        ws = l.strip().split()
        for w in ws:
            if w not in w2id:
                w2id[w] = len(w2id)
                
        wids = [w2id[w] for w in ws]        
        #print >>wf, ' '.join(map(str, wids))
        wf.writelines(' '.join(map(str, wids)))
        wf.writelines('\n')
        tem_sep = '\n' + str(docid) +','
        wf_matrix.writelines('\n'+str(docid)+','+tem_sep.join(map(str,wids)))
        docid = docid + 1
        
        for ind_focus, wid_focus in enumerate(wids):
            ind_lo = max(0, ind_focus-window_size)
            ind_hi = min(len(wids), ind_focus+window_size+1)
            '''
            for wid_con in wids[ind_lo: ind_hi]:
                rows.append(wid_focus)
                cols.append(wid_con)
            '''
            for ind_c in range(ind_lo, ind_hi):
                if ind_c == ind_focus:
                    continue
                '''diagonals are zeros or not'''
                if wid_focus == wids[ind_c]:
                    continue
                rows.append(wid_focus)
                cols.append(wids[ind_c])
        if docid%batch_size == 0 and docid != 0:
            np.save(os.path.join(DATA_DIR, 'CoEmbedding/intermediate/coo_%d_%d.npy' % (saveid, docid)), np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))
            saveid = saveid + batch_size
            print('%dth doc, %dth doc' % (saveid, docid))
            rows = []
            cols = []
    np.save(os.path.join(DATA_DIR, 'CoEmbedding/intermediate/coo_%d_%d.npy' % (saveid, docid)), np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))      
    
    wf.close()
    wf_matrix.close()
    print('write file: ', res_pt)
    print('write file: ', matrix_pt)
    return docid


def write_w2id(res_pt):
    print('write:', res_pt)
    wf = open(res_pt, 'w')
    for w, wid in sorted(w2id.items(), key=lambda d:d[1]):
        #print >>wf, '%d\t%s' % (wid, w)
        wf.writelines('%d\t%s' % (wid, w))
        wf.writelines('\n')
    wf.close()
    
def load_data(csv_file, shape):
    print('loading data')
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['doc_id']), np.array(tp['word_id'])
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data

def _coord_batch(lo, hi, train_data):
    rows = []
    cols = []
    for u in range(lo, hi):
        for w, c in itertools.permutations(train_data[u].nonzero()[1], 2):
            rows.append(w)
            cols.append(c)
        if u%1000 == 0:
            print('%dth doc' % u)
    np.save(os.path.join(DATA_DIR, 'CoEmbedding/intermediate/coo_%d_%d.npy' % (lo, hi)), np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))

def _matrixw_batch(lo, hi, matW):
    coords = np.load(os.path.join(DATA_DIR, 'CoEmbedding/intermediate/coo_%d_%d.npy' % (lo, hi)))    
    rows = coords[:, 0]
    cols = coords[:, 1]    
    tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_words, n_words), dtype='float32').tocsr()
    matW = matW + tmp
    print("User %d to %d finished" % (lo, hi))
    sys.stdout.flush()
    return matW
        
if __name__ == '__main__':
    #doc_pt = DATA_DIR+'20newsForLDA.txt'
    doc_pt = DATA_DIR+'CoEmbedding/20news_min_cnt.txt'
    dwid_pt = DATA_DIR+'CoEmbedding/doc_id.txt'
    dwmatrix_pt = DATA_DIR+'CoEmbedding/dw_matrix.csv'
    voca_pt = DATA_DIR+'CoEmbedding/vocab.txt'
    batch_size = 1000
    window_size = 5 #actually half window size
    n_docs = indexFile(doc_pt, dwid_pt, dwmatrix_pt, batch_size, window_size)
    n_words = len(w2id)
    print('n(d)=', n_docs, 'n(w)=', n_words) 
    write_w2id(voca_pt)
    matrixD = load_data(dwmatrix_pt, (n_docs, n_words))
    
    start_idx = list(range(0, n_docs, batch_size))
    end_idx = start_idx[1:] + [n_docs]
    #for lo, hi in zip(start_idx, end_idx):
        #_coord_batch(lo, hi, matrixD)
        
    matrixW = sparse.csr_matrix((n_words, n_words), dtype='float32')

    for lo, hi in zip(start_idx, end_idx):
        matrixW = _matrixw_batch(lo, hi, matrixW)
        print(float(matrixW.nnz) / np.prod(matrixW.shape))
    
    np.save(os.path.join(DATA_DIR, 'CoEmbedding/coordinate_co_binary_data.npy'), matrixW.data)
    np.save(os.path.join(DATA_DIR, 'CoEmbedding/coordinate_co_binary_indices.npy'), matrixW.indices)
    np.save(os.path.join(DATA_DIR, 'CoEmbedding/coordinate_co_binary_indptr.npy'), matrixW.indptr)

    
        
    print(matrixD.shape, matrixW.shape)
    