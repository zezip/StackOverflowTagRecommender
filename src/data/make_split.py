from load_and_clean_chunked import load_and_clean_chunked
from get_split import save_split, SPLIT
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
# see http://scikit.ml/stratification.html to see how the multilabel stratification works
# You probably need to run `pip install scikit-multilearn`
from skmultilearn.model_selection import iterative_train_test_split as train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

SEED = 1
TEST_SIZE = 0.2
# To be determined by frequency analysis
MIN_N = 1000

def make_split(verbose=False):
    """
        Do not call this function to get the split, call 'get_split()' in get_split.py
        Call this function to filter, split, and serialize your data
    """
    df = load_and_clean_chunked(verbose=verbose)

    kept_tags = get_kept_tags(df)
    index_to_tag = np.array(list(kept_tags))
    tag_to_index = {tag : index for index, tag in enumerate(index_to_tag)}

    X, y = [], []

    # Iterate through df, keep entries that have at least one label in our 'kept_tags' set, squish title and body to make X
    if verbose:
        print("Removing infrequent tags...")
    for _, row in tqdm(df.iterrows(), disable=not verbose, total=len(df)):
        y_init = np.zeros(len(kept_tags))
        for tag in row['tags']:
            try:
                y_init[tag_to_index[tag]] = 1
            except KeyError:
                pass
        # If sum is 0, no relevant tags are kept
        if y_init.sum() > 0:
            y.append(y_init)
            X.append(row['title'] + ' ' + row['body'])
    if verbose:
        print(f"Kept {len(kept_tags)} tags and {len(X)} samples...")
        print('Tags:', kept_tags)
        print("Shuffling...")
    X, y = shuffle(np.array(X).reshape(-1, 1), np.array(y), random_state=SEED)
    if verbose:
        print("Splitting...")
    X_train, y_train, X_test, y_test = train_test_split(X, y, TEST_SIZE)
    save_split(X_train, y_train, X_test, y_test, index_to_tag)
    print(f"Split has been serialized to {SPLIT}.")

def get_kept_tags(df, make_power_law_graph=False):
    tag_to_frequency = Counter()
    # Counting tag frequencies
    counts = []
    for tags in df['tags']:
        for tag in tags:
            tag_to_frequency[tag] += 1
        counts.append(len(tags))

    if make_power_law_graph:
        print("Total tags:", len(tag_to_frequency))
        print('Mean tag count:', np.mean(counts))
        print('Std. dev of tag counts:', np.std(counts))
        tags, freqs = [], []
        for tag, freq in tag_to_frequency.items():
            tags.append(tag)
            freqs.append(freq)
        df = pd.DataFrame({
            'tag' : tags,
            'tag frequency' : freqs
        })
        sns.histplot(data=df, x="tag frequency", bins=50, log_scale=True)
        plt.title('tag frequency distribution')
        # plt.show()
        plt.savefig('tag_frequencies.png')

    return {tag for tag, frequency in tag_to_frequency.items() if frequency >= MIN_N}

if __name__ == "__main__":
    make_split(verbose=True)