'''
Remove words occur less than min_count times throughout the corpus
'''
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import itertools
import os

w2cnt = {}
DATA_DIR = 'D:/UB/research/dataset/20newsgroups/'

def wordCount(pt):
    print('counting...')
    for l in open(pt):
        ws = l.strip().split()
        for w in ws:
            w2cnt[w] = w2cnt.get(w,0) + 1

def indexFile(pt, out_pt, min_count = 10):
    print('index file: ', pt)
    wf = open(out_pt, 'w')
    for l in open(pt):
        ws = l.strip().split()
        for w in ws:
            if w2cnt[w] >= min_count:
                wf.writelines(w + ' ')
        wf.writelines('\n')
                
                
                
if __name__ == '__main__':
    doc_pt = DATA_DIR+'20newsForW2V.txt'
    dminc_pt = DATA_DIR+'CoEmbedding/20news_min_cnt.txt'
    min_count = 10
    wordCount(doc_pt)
    indexFile(doc_pt, dminc_pt, min_count)
