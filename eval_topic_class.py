from sklearn import svm, metrics
from sklearn.datasets import load_svmlight_file
import sys
import numpy as np

#embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/results/Embeddings_K50_iter19.npz'
#embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/topicmodeling/LDA/model-final.theta'
embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/topicmodeling/PLSI/280topics/20newsgroup.pzd'
#embedding_pt = 'D:/UB/research/dataset/20newsgroups/CoEmbedding/topicmodeling/NMF/nmf.npz'
n_docs = 18827
n_topics = 20

def getScores( true_classes, pred_classes, average):
    precision = metrics.precision_score( true_classes, pred_classes, average=average )
    recall = metrics.recall_score( true_classes, pred_classes, average=average )
    f1 = metrics.f1_score( true_classes, pred_classes, average=average )
    accuracy = metrics.accuracy_score( true_classes, pred_classes )
    return precision, recall, f1, accuracy
    
if embedding_pt.startswith('D:/UB/research/dataset/20newsgroups/CoEmbedding/results/') or embedding_pt.startswith('D:/UB/research/dataset/20newsgroups/CoEmbedding/topicmodeling/NMF/'):
    data = np.load(embedding_pt)
    theta = data['U']
    data.close()
    print('Normalizing theta from npz')
    norms = np.sum(theta, axis=1, keepdims=True)
    theta = theta/norms
elif embedding_pt.startswith('D:/UB/research/dataset/20newsgroups/CoEmbedding/topicmodeling/LDA/'):
    print('Reading theta from txt')
    theta = np.zeros((n_docs,280),dtype=np.float32)
    for cnt,line in enumerate(open(embedding_pt)):
        ws = line.strip().split()
        theta[cnt] = np.array(list(map(float,ws)))
else:
    print('Reading theta from plsi')
    theta = np.zeros((280, n_docs),dtype=np.float32)
    for cnt,line in enumerate(open(embedding_pt)):
        ws = line.strip().split()
        theta[cnt] = np.array(list(map(float,ws)))
    theta = theta.T
    theta = theta[ :, 0:239 ]

print(theta.shape)
groundtruth_class = np.zeros(n_docs, dtype=np.int)
n_docs_each_class = [799, 973, 984, 982, 961, 980, 972, 990, 994, 994, 999, 991, 981, 990, 987, 997, 910, 940, 775, 628]
temp_idx = 0
for k in range(n_topics):
    groundtruth_class[temp_idx:temp_idx+n_docs_each_class[k]] = k
    temp_idx = temp_idx + n_docs_each_class[k]
    
np.savetxt('D:/UB/research/dataset/20newsgroups/CoEmbedding/label.txt', groundtruth_class, fmt='%d')

#model = svm.LinearSVC(penalty='l2', dual=True)
model = svm.LinearSVC(penalty='l1', dual=False)
print("Training...")
model.fit(theta, groundtruth_class)
print("Done.")

predict_classes = model.predict(theta)
#predict_classes = np.random.randint(n_topics, size=n_docs) #random results: Train Prec (macro average): 0.051, recall: 0.051, F1: 0.051, Acc: 0.051

print(metrics.classification_report(groundtruth_class, predict_classes, digits=3))

for average in ['micro', 'macro']:
    train_precision, train_recall, train_f1, train_acc = getScores( groundtruth_class, predict_classes, average )
    print("Train Prec (%s average): %.3f, recall: %.3f, F1: %.3f, Acc: %.3f" % (average, train_precision, train_recall, train_f1, train_acc))
