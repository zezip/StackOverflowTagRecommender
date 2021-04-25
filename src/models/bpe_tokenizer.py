import sys
from pathlib import Path
import pandas as pd
import os, os.path
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents
from collections import Counter
import errno
import numpy as np

DIR = Path(os.path.abspath('')).resolve()
SRC = DIR.parent
ROOT = SRC.parent
TOKENIZER = str(ROOT/"models"/"tokenizers"/"bpe.json")
sys.path.append(str(SRC/"data"))

from load_and_clean_chunked import load_and_clean_chunked

VOCAB_SIZE = 30000

def load_bpe_encoder(verbose=False):
    """
        Loads the encoder as an object that matches the CountVectorizer API/signature
    """
    return BPEVectorizer(Tokenizer.from_file(TOKENIZER))

def load_bpe_tfidf_encoder(verbose=False):
    """
        Loads the encoder as an object that matches the CountVectorizer API/signature
        This version is the same, except it does TFIDF transforms
    """
    return BPEVectorizerTFIDF(Tokenizer.from_file(TOKENIZER))

class BPEVectorizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.size = tokenizer.get_vocab_size()

    def fit(self, X):
        new_tokens = self.tokenizer.encode(' '.join(X)).tokens
        self.tokenizer.add_tokens(new_tokens)

    def transform(self, X):
        y = self.tokenizer.encode_batch(X)
        res = np.zeros((len(X), self.size))
        for i in range(len(y)):
            for j in y[i].ids:
                res[i, j] += 1
        return res

    def fit_transform(self, X):
        """
            Return X as a sparse matrix (bag-of-words)
        """
        self.fit(X)
        return self.transform(X)

class BPEVectorizerTFIDF:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.size = tokenizer.get_vocab_size()
        self.df_arr = None

    def fit(self, X):
        # TODO: Relearn IDFs over
        # The current approach is valid (and analogous to how the sklearn TFIDF works)
        # But learning IDFs over a larger corpus (that we learned our vocabularly from) should work better
        new_tokens = self.tokenizer.encode(' '.join(X)).tokens
        self.tokenizer.add_tokens(new_tokens)
        self.size = self.tokenizer.get_vocab_size()

        y = self.tokenizer.encode_batch(X)
        res = np.zeros((len(X), self.size), dtype=float)

        self.df_arr = np.zeros(self.size)
        for i in range(len(y)):
            for j in y[i].ids:
                res[i, j] += 1
            self.df_arr += (res[i] > 0).astype(int)
        
        df_arr_zero_indices = np.where(self.df_arr == 0)[0]
        self.df_arr[df_arr_zero_indices] = 1

    def transform(self, X):
        y = self.tokenizer.encode_batch(X)
        res = np.zeros((len(X), self.size), dtype=float)
        for i in range(len(y)):
            for j in y[i].ids:
                res[i, j] += 1

        # Apply IDF gathered from the fit function
        for i in range(len(y)):
            nonzero_indices = np.where(res[i] > 0)[0]
            res[i, nonzero_indices] /= self.df_arr[nonzero_indices]

        return res

    def fit_transform(self, X):
        """
            Return X as a sparse matrix (bag-of-words)
        """
        self.fit(X)
        return self.transform(X)

def try_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

def train_bpe_encoder(verbose=False):
    X = load_and_clean_chunked(verbose=True, supervised=False)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=VOCAB_SIZE)
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    
    if verbose:
        print('Training encoder...')
    tokenizer.train_from_iterator(X, trainer, length=len(X))

    try_mkdir(str(ROOT/"models"))
    try_mkdir(str(ROOT/"models"/"tokenizers"))

    tokenizer.save(TOKENIZER)
    return tokenizer
  
if __name__ == "__main__":
    tokenizer = train_bpe_encoder(verbose=True)
    # print(tokenizer.encode("Hello, y'all! How are you?"))
    # print(tokenizer.encode_batch(["Hello, y'all!", "How are you?"]))
    # countvec = load_bpe_encoder()
    # tfidfvec = load_bpe_tfidf_encoder()
    # X = [
    #     "the quick brown fox",
    #     "jumped over the",
    #     "lazy dog"
    # ]
    # count_transform = countvec.fit_transform(X)
    # tfidf_transform = tfidfvec.fit_transform(X)
    # print('count_transform:\n', count_transform.sum(axis=1, keepdims=True))
    # print('tfidf_transform:\n', tfidf_transform.sum(axis=1, keepdims=True))