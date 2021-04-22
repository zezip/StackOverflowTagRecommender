import sys
from pathlib import Path
import pandas as pd
import os, os.path
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import errno

DIR = Path(os.path.abspath('')).resolve()
SRC = DIR.parent
ROOT = SRC.parent
TOKENIZER = str(ROOT/"models"/"tokenizers"/"bpe.json")
sys.path.append(str(SRC/"data"))

from load_and_clean_chunked import load_and_clean_chunked

VOCAB_SIZE = 30000

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
    
    if verbose:
        print('Training encoder...')
    tokenizer.train_from_iterator(X, trainer, length=len(X))

    try_mkdir(str(ROOT/"models"))
    try_mkdir(str(ROOT/"models"/"tokenizers"))

    tokenizer.save(TOKENIZER)
    return tokenizer

def load_bpe_encoder(verbose=False):
    return Tokenizer.from_file(TOKENIZER)
    
if __name__ == "__main__":
    tokenizer = train_bpe_encoder(verbose=True)
    print(tokenizer.encode("Hello, y'all! How are you?"))
    print(tokenizer.encode_batch(["Hello, y'all!", "How are you?"]))