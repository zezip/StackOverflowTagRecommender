from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import sys
import os, os.path
from pathlib import Path

SRC = Path(os.path.abspath('')).resolve().parent
sys.path.append(str(SRC/"data"))

from get_split import get_split
from bpe_tokenizer import load_bpe_encoder, load_bpe_tfidf_encoder

SEED = 1
MAX_ITER = 10000

def main():
    (X_train, y_train, X_test, y_test), index_to_tag = get_split()
    X_train = X_train.flatten()
    X_test = X_test.flatten()

    run_countvectorizer_lda1_svm(X_train, y_train, X_test, y_test)
    run_countvectorizer_lda30_svm(X_train, y_train, X_test, y_test)
    run_bpe_svm(X_train, y_train, X_test, y_test)
    run_bpe_tfidf_svm(X_train, y_train, X_test, y_test)
    run_bpe_lda2_svm(X_train, y_train, X_test, y_test)
    run_bpe_lda30_svm(X_train, y_train, X_test, y_test)

def run_countvectorizer_lda1_svm(X_train, y_train, X_test, y_test):
    vectorizer = CountVectorizer()
    X_train_embedded = vectorizer.fit_transform(X_train)
    X_test_embedded = vectorizer.transform(X_test)

    lda = LatentDirichletAllocation(n_components=1, random_state=SEED)
    X_train_topic = lda.fit_transform(X_train_embedded)
    X_test_topic = lda.transform(X_test_embedded)

    svm = LinearSVC(C=1, max_iter=MAX_ITER)
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf = multilabel_clf.fit(X_train_topic, y_train)
    y_pred = multilabel_clf.predict(X_test_topic)
    evaluate(y_test, y_pred, 'CountVectorizer_LDA1_SVM')

def run_countvectorizer_lda30_svm(X_train, y_train, X_test, y_test):
    vectorizer = CountVectorizer()
    X_train_embedded = vectorizer.fit_transform(X_train)
    X_test_embedded = vectorizer.transform(X_test)

    lda = LatentDirichletAllocation(n_components=30, random_state=SEED)
    X_train_topic = lda.fit_transform(X_train_embedded)
    X_test_topic = lda.transform(X_test_embedded)

    svm = LinearSVC(C=1, max_iter=MAX_ITER)
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf = multilabel_clf.fit(X_train_topic, y_train)
    y_pred = multilabel_clf.predict(X_test_topic)
    evaluate(y_test, y_pred, 'CountVectorizer_LDA30_SVM')

def run_bpe_svm(X_train, y_train, X_test, y_test):
    vectorizer = load_bpe_encoder()
    X_train_embedded = vectorizer.fit_transform(X_train)
    X_test_embedded = vectorizer.transform(X_test)

    svm = LinearSVC(C=1, max_iter=MAX_ITER)
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf = multilabel_clf.fit(X_train_embedded, y_train)
    y_pred = multilabel_clf.predict(X_test_embedded)
    evaluate(y_test, y_pred, 'BPEVectorizer_SVM')

def run_bpe_tfidf_svm(X_train, y_train, X_test, y_test):
    vectorizer = load_bpe_tfidf_encoder()
    X_train_embedded = vectorizer.fit_transform(X_train)
    X_test_embedded = vectorizer.transform(X_test)

    svm = LinearSVC(C=1, max_iter=MAX_ITER)
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf = multilabel_clf.fit(X_train_embedded, y_train)
    y_pred = multilabel_clf.predict(X_test_embedded)
    evaluate(y_test, y_pred, 'BPE_TFIDF_Vectorizer_SVM')

def run_bpe_lda2_svm(X_train, y_train, X_test, y_test):
    vectorizer = load_bpe_encoder()
    X_train_embedded = vectorizer.fit_transform(X_train)
    X_test_embedded = vectorizer.transform(X_test)

    lda = LatentDirichletAllocation(n_components=2, random_state=SEED)
    X_train_topic = lda.fit_transform(X_train_embedded)
    X_test_topic = lda.transform(X_test_embedded)

    svm = LinearSVC(C=1, max_iter=MAX_ITER)
    multilabel_clf = MultiOutputClassifier(svm,)
    multilabel_clf = multilabel_clf.fit(X_train_topic, y_train)
    y_pred = multilabel_clf.predict(X_test_topic)
    evaluate(y_test, y_pred, 'BPEVectorizer_LDA2_SVM')

def run_bpe_lda30_svm(X_train, y_train, X_test, y_test):
    vectorizer = load_bpe_encoder()
    X_train_embedded = vectorizer.fit_transform(X_train)
    X_test_embedded = vectorizer.transform(X_test)

    lda = LatentDirichletAllocation(n_components=30, random_state=SEED)
    X_train_topic = lda.fit_transform(X_train_embedded)
    X_test_topic = lda.transform(X_test_embedded)

    svm = LinearSVC(C=1, max_iter=MAX_ITER)
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf = multilabel_clf.fit(X_train_topic, y_train)
    y_pred = multilabel_clf.predict(X_test_topic)
    evaluate(y_test, y_pred, 'BPEVectorizer_LDA30_SVM')

 
def evaluate(y_test, y_pred, name):
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)
    print('-------------------')
    print(f"{name}_accuracy:", accuracy)
    print(f'{name}_precision:', np.mean(precision))
    print(f'{name}_recall:', np.mean(recall))
    print(f'{name}_f1:', np.mean(fscore))
    print('-------------------')


if __name__ == "__main__":
    main()
    

    

