import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os, os.path
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
sns.set(rc={'figure.figsize':(5,5)})
sns.color_palette("mako", as_cmap=True)

from bpe_tokenizer import load_bpe_encoder

DIR = Path(os.path.abspath('')).resolve()
SRC = DIR.parent
ROOT = SRC.parent
sys.path.append(str(SRC/"data"))

from get_split import get_split

SEED = 0

if __name__ == "__main__":
    # Load dataset
    (X_train, y_train, X_test, y_test), index_to_tag = get_split(verbose=True)
    X_train = X_train.flatten()
    X_test = X_test.flatten()

    # Load vectorizer and fit transform
    vectorizer = load_bpe_encoder()
    X_train_embedded = vectorizer.fit_transform(X_train)

    # Calculating a perplexity score for each model- a good model minimizes perplexity
    i_vals = []
    perplexities = []
    for i in tqdm(range(1, 6)):

        # Average perplexity cross validated with 5 folds
        inner_perplexities = []
        kf = KFold(n_splits=5, random_state=SEED, shuffle=True)
        for train_index, test_index in kf.split(X_train_embedded):

            # Construct lda, fit to train, take perplexity score on test
            lda = LatentDirichletAllocation(n_components=i, random_state=SEED)
            lda.fit(X_train_embedded[train_index])
            perplexity = lda.perplexity(X_train_embedded[test_index])
            inner_perplexities.append(perplexity)

        i_vals.append(i)
        perplexities.append(np.mean(inner_perplexities))

    print('perps:', perplexities)
    df = pd.DataFrame({
        '# Topics' : i_vals,
        'Perplexities' : perplexities
    })
    sns.lineplot(data=df, x='# Topics', y='Perplexities')
    plt.savefig('perps_small_scale.png')