import numpy as np
from pathlib import Path
import os.path
from glob import glob

DIR = Path(os.path.abspath('')).resolve()
ROOT = DIR.parent.parent
# SPLIT = str(ROOT/"data"/"split.npz")
SPLIT = str(ROOT/"data"/"split.npz")

def get_split(verbose=False):
    """
        If this doesn't work (file not found), you might need to run the main of make_split.py (which calls 'make_split') first
        Loads and returns a tuple of:
            (X_train, y_train, X_test, y_test), index_to_tag
        s.t.:
            *_train is a 1d numpy array of strings (one per sample)
            *_test is a 2d numpy array of tag indices (len(kept_tags) per sample)
            index_to_tag is a numpy array mapping indices to their tag string
    """
    # with open(SPLIT, 'rb') as f:
    #     return pickle.load(f)
    loaded = np.load(SPLIT)
    X_train = loaded['X_train']
    X_test = loaded['X_test']
    y_train = loaded['y_train']
    y_test = loaded['y_test']
    index_to_tag = loaded['index_to_tag']
    return (X_train, y_train, X_test, y_test), index_to_tag

def save_split(X_train, y_train, X_test, y_test, index_to_tag, verbose=False):
    """
        Called by 'make_split' to serialize the split to a shared location
    """
    # with open(SPLIT, 'wb') as f:
    #     pickle.dump(data, f)
    np.savez_compressed(SPLIT, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, index_to_tag=index_to_tag)


if __name__ == "__main__":
    (X_train, y_train, X_test, y_test), index_to_tag = get_split(verbose=True)
    print('X_train', X_train.shape, type(X_train))
    print('y_train', y_train.shape, type(y_train))
    print('X_test', X_test.shape, type(X_test))
    print('y_test', y_test.shape, type(y_test))
    print('index_to_tag', len(index_to_tag), type(index_to_tag))
