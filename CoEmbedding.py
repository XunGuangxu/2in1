import os
import sys
import time

import numpy as np
from numpy import linalg as LA, argpartition
from scipy import sparse

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin



class CoEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, n_embeddings=100, K=20, max_iter=10, batch_size=1000,
                 init_std=0.01, dtype='float32', n_jobs=8, random_state=None,
                 save_params=False, save_dir='.', verbose=False, **kwargs):
        '''
        Parameters
        ---------
        n_embeddings : int
            Dimensionality of embeddings
        max_iter : int
            Maximal number of iterations to perform
        batch_size: int
            Batch size to perform parallel update
        init_std: float
            The latent factors will be initialized as Normal(0, init_std**2)
        dtype: str or type
            Data-type for the parameters, default 'float32' (np.float32)
        n_jobs: int
            Number of parallel jobs to update latent factors
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        save_params: bool
            Whether to save parameters after each iteration
        save_dir: str
            The directory to save the parameters
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
        '''
        self.n_embeddings = n_embeddings
        self.n_topics = K
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.init_std = init_std
        self.dtype = dtype
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.set_state(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        ''' Model hyperparameters
        Parameters
        ---------
        lambda: float
            Regularization parameter.
        c0, c1: float
            Confidence for 0 and 1 in Hu et al., c0 must be less than c1
        '''
        self.lam_sparse_d = float(kwargs.get('lam_sparse_d', 1e-5))
        self.lam_sparse = float(kwargs.get('lam_sparse', 1e-5))
        self.lam_d = float(kwargs.get('lam_d', 1e0))
        self.lam_w = float(kwargs.get('lam_w', 1e-2))
        self.lam_t = float(kwargs.get('lam_t', 1e-2))
        self.c0 = float(kwargs.get('c0', 0.01))
        self.c1 = float(kwargs.get('c1', 1.0))
        assert self.c0 <= self.c1, "c0 must be smaller than c1"

    def _init_params(self, n_docs, n_words):
        ''' Initialize all the latent factors and biases '''
        self.theta = self.init_std * np.random.randn(n_docs, self.n_topics).astype(self.dtype) + 1
        self.topic = self.init_std * np.random.randn(n_words, self.n_topics).astype(self.dtype) + 1
        self.beta = self.init_std * np.random.randn(n_words, self.n_embeddings).astype(self.dtype)
        self.gamma = self.init_std * np.random.randn(n_words, self.n_embeddings).astype(self.dtype)
        self.alpha = self.init_std * np.random.randn(self.n_topics, self.n_embeddings).astype(self.dtype)
        assert np.all(self.theta > 0)
        assert np.all(self.topic > 0)

    def fit(self, X, M, voca_pt):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix, shape (n_docs, n_words)
            Training click matrix.

        M : scipy.sparse.csr_matrix, shape (n_words, n_words)
            Training co-occurrence matrix.
            
        voca_pt : vocabulary file pointer string
        '''
        self._read_vocab(voca_pt)
        n_docs, n_words = X.shape
        assert M.shape == (n_words, n_words)

        self._init_params(n_docs, n_words)
        self._update(X, M)
        return self

    def _update(self, X, M):
        '''Model training and evaluation on validation set'''
        XT = X.T.tocsr()  # pre-compute this
        for i in range(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_factors(X, XT, M)
            if self.save_params:
                self._save_params(i)

    def _update_factors(self, X, XT, M):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating theta...')
        self.theta = update_theta(self.topic, X, self.c0, self.c1, self.lam_sparse_d, self.n_jobs, self.batch_size)
        if self.verbose:
            print('\r\tUpdating theta: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating topics...')
        self.topic = update_topic(self.theta, self.alpha, self.gamma, XT, self.c0, self.c1, self.lam_sparse_d, self.lam_d, self.lam_t, self.n_jobs, self.batch_size)
        self.topic = _normalize_topic(self.topic)
        if self.verbose:
            print('\r\tUpdating topics: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating context embeddings...')
        self.gamma = update_gamma(self.alpha, self.beta, self.topic, M, self.c0, self.c1, self.lam_sparse, self.lam_w, self.lam_t, self.n_jobs, self.batch_size)
        if self.verbose:
            print('\r\tUpdating context embeddings: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating word embeddings...')
        # here it really should be M^T and F^T, but both are symmetric
        self.beta = update_beta(self.gamma, M, self.lam_sparse, self.n_jobs, self.batch_size)
        if self.verbose:
            print('\r\tUpdating word embeddings: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating topic embeddings...')
        self.alpha = update_alpha(self.gamma, self.topic.T, self.lam_sparse, self.n_jobs, self.batch_size)
        if self.verbose:
            print('\r\tUpdating topic embeddings: time=%.2f' % (time.time() - start_t))

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'Embeddings_K%d_iter%d.npz' % (self.n_embeddings, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.theta, V=self.topic, C=self.gamma, B=self.beta, A=self.alpha)
        
    def most_similar(self, dword, topn):
        wid = self.w2id[dword]
        unibeta = self.gamma.copy()
        normss = np.linalg.norm(unibeta, axis = 1, keepdims = True)
        unibeta = unibeta/normss
        wvec = unibeta[wid]
        distances = np.inner(-wvec, unibeta)
        most_extreme = np.argpartition(distances, topn)[:topn]
        #print(np.sort(distances.take(most_extreme)))
        return [self.id2w[t] for t in most_extreme.take(np.argsort(distances.take(most_extreme)))]  # resort topn into order
    
    def show_save_topics(self, topn, filename):
        fout = open(filename, 'w')
        topicword = [[] for i in range(self.n_topics)]
        tempT = - self.topic.copy().T
        for i in range(self.n_topics):
            most_extreme = np.argpartition(tempT[i], topn)[:topn]
            topicword[i] = [self.id2w[t] for t in most_extreme.take(np.argsort(tempT[i].take(most_extreme)))]
            fout.writelines('Topic %dth:\n' % i)
            for word in topicword[i]:
                fout.writelines('\t%s\n' % word)
        fout.close()
        return topicword  # resort topn into order
    
    def show_topics_embeddings(self, topn):
        topicword = [[] for i in range(self.n_topics)]
        unigamma = self.gamma.copy()
        #normss = np.linalg.norm(unigamma, axis = 1, keepdims = True)
        #unigamma = unigamma/normss
        unialpha = self.alpha.copy()
        #normsss = np.linalg.norm(unialpha, axis = 1, keepdims = True)
        #unialpha = unialpha/normsss
        for i in range(self.n_topics):
            tvec = unialpha[i]
            distances = np.inner(-tvec, unigamma)
            most_extreme = np.argpartition(distances, topn)[:topn]
            topicword[i] = [self.id2w[t] for t in most_extreme.take(np.argsort(distances.take(most_extreme)))]
        return topicword  # resort topn into order
    
    def topic_similarity(self, topn=10):
        topicword = [[] for i in range(self.n_topics)]
        unialpha = self.alpha.copy()
        normsss = np.linalg.norm(unialpha, axis = 1, keepdims = True)
        unialpha = unialpha/normsss
        for i in range(self.n_topics):
            tvec = unialpha[i]
            distances = np.inner(-tvec, unialpha)
            #print('topic',distances)
            topicword[i] = np.argsort(distances)
        return topicword
    
    def _read_vocab(self, voca_pt):
        self.w2id = {}
        self.id2w = {}
        for l in open(voca_pt):
            ws = l.strip().split()
            self.w2id[ws[1]] = int(ws[0])
            self.id2w[int(ws[0])] = ws[1]


# Utility functions #
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]


def update_theta(topic, X, c0, c1, lam_sparse_d, n_jobs, batch_size):
    '''Update user latent factors'''
    m, n = X.shape  # m: number of users, n: number of items
    f = topic.shape[1]  # f: number of factors

    BTB = c0 * np.dot(topic.T, topic)  # precompute this
    BTBpR = BTB + lam_sparse_d * np.eye(f, dtype=topic.dtype)
    
    start_idx = list(range(0, m, batch_size))
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs)(delayed(_solve_theta)(lo, hi, topic, X, BTBpR, c0, c1, f) for lo, hi in zip(start_idx, end_idx))
    theta = np.vstack(res)
    return theta


def _solve_theta(lo, hi, topic, X, BTBpR, c0, c1, f):
    theta_batch = np.empty((hi - lo, f), dtype=topic.dtype)
    for ib, u in enumerate(range(lo, hi)):
        x_u, idx_u = get_row(X, u)
        B_u = topic[idx_u]
        a = x_u.dot(c1 * B_u)
        '''
        non-zero elements handled in this loop
        '''
        B = BTBpR + B_u.T.dot((c1 - c0) * B_u)#B_u only contains vectors corresponding to non-zero doc-word occurence
        theta_batch[ib] = LA.solve(B, a)
    theta_batch = theta_batch.clip(0)
    return theta_batch


def update_topic(theta, alpha, gamma, XT, c0, c1, lam_sparse_d, lam_d, lam_t, n_jobs, batch_size):
    '''Update user latent factors'''
    m, n = XT.shape  # m: number of users, n: number of items
    f = theta.shape[1]  # f: number of factors

    BTB = c0 * np.dot(theta.T, theta)  # precompute this
    BTBpR = lam_d * BTB + lam_sparse_d * np.eye(f, dtype=theta.dtype) + lam_t * np.eye(f, dtype=theta.dtype)

    start_idx = list(range(0, m, batch_size))
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs)(delayed(_solve_topic)(lo, hi, theta, alpha, gamma, XT, BTBpR, c0, c1, f, lam_d, lam_t) for lo, hi in zip(start_idx, end_idx))
    topic = np.vstack(res)
    return topic


def _solve_topic(lo, hi, theta, alpha, gamma, XT, BTBpR, c0, c1, f, lam_d, lam_t):
    topic_batch = np.empty((hi - lo, f), dtype=theta.dtype)
    for ib, u in enumerate(range(lo, hi)):
        x_u, idx_u = get_row(XT, u)
        B_u = theta[idx_u]
        cpAT = gamma[u].dot(alpha.T)
        a = lam_d * x_u.dot(c1 * B_u) + lam_t * cpAT
        '''
        non-zero elements handled in this loop
        '''
        B = BTBpR + B_u.T.dot((c1 - c0) * B_u)#B_u only contains vectors corresponding to non-zero doc-word occurence
        topic_batch[ib] = LA.solve(B, a)
    topic_batch = topic_batch.clip(0)
    return topic_batch


def update_gamma(alpha, beta, topic, M, c0, c1, lam_sparse, lam_w, lam_t, n_jobs, batch_size):
    '''Update item latent factors/embeddings'''
    n, m = topic.shape  # m: number of users, n: number of items
    f = alpha.shape[1]
    assert alpha.shape[0] == m
    assert beta.shape == (n, f)
    ATA = alpha.T.dot(alpha)
    BTB = beta.T.dot(beta)
    B = lam_t * ATA + lam_w * BTB + lam_sparse * np.eye(f, dtype=beta.dtype)

    start_idx = list(range(0, n, batch_size))
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs)(delayed(_solve_gamma)(lo, hi, alpha, beta, topic, M, B, c0, c1, f, lam_w, lam_t) for lo, hi in zip(start_idx, end_idx))
    gamma = np.vstack(res)
    return gamma


def _solve_gamma(lo, hi, alpha, beta, topic, M, B, c0, c1, f, lam_w, lam_t):
    gamma_batch = np.empty((hi - lo, f), dtype=alpha.dtype)
    for ib, i in enumerate(range(lo, hi)):
        t_i = topic[i,:]
        m_i, idx_m_i = get_row(M, i)
        B_i = beta[idx_m_i]
        '''
        the reason why they put G_i in the loop instead of calculate GTG = gamma.T * gamma is that in the objective function,
        we currently only consider the non-zero elements in matrix W.
        '''
        a = lam_t * np.dot(t_i, alpha) + lam_w * np.dot(m_i, B_i)
        gamma_batch[ib] = LA.solve(B, a)
    return gamma_batch


def update_beta(gamma, MT, lam_sparse, n_jobs, batch_size):
    '''Update context latent factors'''
    n, f = gamma.shape  # n: number of items, f: number of factors
    CTC = gamma.T.dot(gamma)
    B = CTC + lam_sparse * np.eye(f, dtype=gamma.dtype)

    start_idx = list(range(0, n, batch_size))
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs)(delayed(_solve_beta)(lo, hi, gamma, MT, B, f) for lo, hi in zip(start_idx, end_idx))
    beta = np.vstack(res)
    return beta


def _solve_beta(lo, hi, gamma, MT, B, f):
    beta_batch = np.empty((hi - lo, f), dtype=gamma.dtype)
    for ib, j in enumerate(range(lo, hi)):
        m_j, idx_m_j = get_row(MT, j)
        C_j = gamma[idx_m_j]
        a = np.dot(m_j, C_j)
        beta_batch[ib] = LA.solve(B, a)
    return beta_batch


def update_alpha(gamma, topicT, lam_sparse, n_jobs, batch_size):
    '''Update context latent factors'''
    n, f = gamma.shape  # n: number of items, f: number of factors
    k = topicT.shape[0]
    CTC = gamma.T.dot(gamma)
    B = CTC + lam_sparse * np.eye(f, dtype=gamma.dtype)
    '''In the future, more weight could go to non-zero entries, just like matrix D'''
    start_idx = list(range(0, k, batch_size))
    end_idx = start_idx[1:] + [k]
    res = Parallel(n_jobs)(delayed(_solve_alpha)(lo, hi, gamma, topicT, B, f) for lo, hi in zip(start_idx, end_idx))
    alpha = np.vstack(res)
    return alpha


def _solve_alpha(lo, hi, gamma, topicT, B, f):
    alpha_batch = np.empty((hi - lo, f), dtype=gamma.dtype)
    for ib, j in enumerate(range(lo, hi)):
        t_j = topicT[j,:]        
        a = np.dot(t_j, gamma)
        alpha_batch[ib] = LA.solve(B, a)
    return alpha_batch


def _normalize_topic(topic):
    norms = np.sum(topic, axis=0)
    normtopic = topic/norms
    return normtopic
