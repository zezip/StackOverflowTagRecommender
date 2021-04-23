from pathlib import Path
import pandas as pd
import os
from os import listdir
from get_split import get_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

DIR = Path(os.path.abspath('')).resolve()
DATA = str(DIR/"data"/"chunked")

C_range = [0.001,0.01,0.1,1,10,100,1000]

def tfidf_tokenize(X_train, y_train, X_test, y_test):
    vectorizer = TfidfVectorizer()
    
    Mtrain = vectorizer.fit_transform(X_train.ravel())
    Mtest = vectorizer.transform(X_test.ravel())
    
    svm = LinearSVC(C=1)
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf = multilabel_clf.fit(Mtrain, y_train)
    y_test_pred = multilabel_clf.predict(Mtest)
    score = multilabel_clf.score(Mtest, y_test)
    print('Score ',score)
    #predictions = clf.predict_proba(X_test)
    print('ROC-AUC yields ' + str(roc_auc_score(y_test, y_test_pred)))
    print('precision_recall_fscore_support ', precision_recall_fscore_support(y_test, y_test_pred))
    #return Mtrain, vectorizer
    return score
    
def count_tokenize(X_train, y_train, X_test, y_test):
    vectorizer = CountVectorizer()
    Mtrain = vectorizer.fit_transform(X_train.ravel())
    Mtest = vectorizer.transform(X_test.ravel())
  
    svm = LinearSVC(C=0.01)
    multilabel_clf = MultiOutputClassifier(svm)
    multilabel_clf = multilabel_clf.fit(Mtrain, y_train)
    y_test_pred = multilabel_clf.predict(Mtest)
    score = multilabel_clf.score(Mtest, y_test)
    
    print('Score ',score)
    print('ROC-AUC yields ' + str(roc_auc_score(y_test, y_test_pred)))
    print('precision_recall_fscore_support ', precision_recall_fscore_support(y_test, y_test_pred))
    return score
if __name__ == "__main__":
    (X_train, y_train, X_test, y_test), index_to_tag = get_split()
    
    score1 = tfidf_tokenize(X_train, y_train, X_test, y_test)
    print('')
    score2 = count_tokenize(X_train, y_train, X_test, y_test)
    
    #print(score1, score2)
    

    

