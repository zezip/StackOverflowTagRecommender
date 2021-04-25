from pathlib import Path
import pandas as pd
import os
from os import listdir
from get_split import get_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

DIR = Path(os.path.abspath('')).resolve()
DATA = str(DIR/"data"/"chunked")

C_range = [0.001,0.01,0.1,1,10,100,1000]
n_range = [8000]

def tfidf_tokenize(X_train, y_train, X_test, y_test):
    vectorizer = TfidfVectorizer()
    Xtrain = vectorizer.fit_transform(X_train.ravel())
    Xtest = vectorizer.transform(X_test.ravel())
    
    """
    SVC with linear kernel
    """
    svm = LinearSVC(C=1)
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf = multilabel_clf.fit(Xtrain, y_train)
    y_test_pred = multilabel_clf.predict(Xtest)
    score = multilabel_clf.score(Xtest, y_test)
    print('')
    print('accuracy ',score)
    print('precision_recall_fscore_support ', precision_recall_fscore_support(y_test, y_test_pred, average='weighted'))
    
    """
    dummy classifier
    """
    dummy_clf = DummyClassifier()
    dummy_clf.fit(Xtrain, y_train)
    dummy_pred = dummy_clf.predict(Xtest)
    dummy_score = dummy_clf.score(Xtest, y_test)
    print('')
    print('dummy accuracy ', dummy_score)
    print('precision_recall_fscore_support ', precision_recall_fscore_support(y_test, dummy_pred, average='weighted'))
    
    """
    SVC with rbf kernel with Truncated SVD
    """
    
    for n in n_range:
        svd = TruncatedSVD(n_components=n)
        svd_fit = svd.fit(Xtrain)
        var_explained = svd.explained_variance_ratio_.sum()
        print(str(n) + ' variance: ',var_explained)
        
    Xtrain_SVD = svd.fit_transform(Xtrain)
    Xtest_SVD = svd.fit_transform(Xtest)
    
    svm = SVC(kernel='rbf', gamma='auto')
    multilabel_clf = MultiOutputClassifier(svm)
    clf_SVD = multilabel_clf.fit(Xtrain_SVD, y_train)

    y_pred_SVD = clf_SVD.predict(Xtest_SVD)
    score = clf_SVD.score(Xtest_SVD, y_test)
    print('')
    print('SVD accuracy for '+str(n)+' ', score)
    print('SVD precision_recall_fscore_support ', precision_recall_fscore_support(y_test, y_pred_SVD, average='weighted'))
    print('')
    return None
    
def count_tokenize(X_train, y_train, X_test, y_test):
    vectorizer = CountVectorizer()
    Xtrain = vectorizer.fit_transform(X_train.ravel())
    Xtest = vectorizer.transform(X_test.ravel())
    
    """
    SVC with linear kernel
    """
    svm = LinearSVC(C=0.01)
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf = multilabel_clf.fit(Xtrain, y_train)
    y_test_pred = multilabel_clf.predict(Xtest)
    score = multilabel_clf.score(Xtest, y_test)
    
    print('')
    print('accuracy ',score)
    print('precision_recall_fscore_support ', precision_recall_fscore_support(y_test, y_test_pred, average='weighted'))
    
    """
    dummy classifier
    """
    dummy_clf = DummyClassifier()
    dummy_clf.fit(Xtrain, y_train)
    dummy_pred = dummy_clf.predict(Xtest)
    dummy_score = dummy_clf.score(Xtest, y_test)
    print('')
    print('dummy accuracy ', dummy_score)
    print('precision_recall_fscore_support ', precision_recall_fscore_support(y_test, dummy_pred, average='weighted'))
    
    """
    SVC with rbf kernel with Truncated SVD
    """
    
    for n in n_range:
        svd = TruncatedSVD(n_components=n)
        svd_fit = svd.fit(Xtrain)
        var_explained = svd.explained_variance_ratio_.sum()
        print(str(n) + ' variance: ',var_explained)
        
    Xtrain_SVD = svd.fit_transform(Xtrain)
    Xtest_SVD = svd.transform(Xtest)
    
    svm = SVC(kernel='rbf', gamma='auto')
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf_SVD = multilabel_clf.fit(Xtrain_SVD, y_train)
    
    y_pred_SVD = multilabel_clf_SVD.predict(Xtest_SVD)
    score = multilabel_clf.score(Xtest_SVD, y_test)
    print('')
    print('SVD accuracy ', score)
    print('SVD precision_recall_fscore_support ', precision_recall_fscore_support(y_test, y_pred_SVD, average='weighted'))
    
    return None
if __name__ == "__main__":
    (X_train, y_train, X_test, y_test), index_to_tag = get_split()
    
    print("==TFIDF==")
    tfidf_tokenize(X_train, y_train, X_test, y_test)
    
    print("==COUNT==")
    count_tokenize(X_train, y_train, X_test, y_test)
    
    
   
