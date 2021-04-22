from xml.sax.saxutils import unescape
from pathlib import Path
import pandas as pd
import os.path
import re
import numpy as np
from glob import glob
from tqdm import tqdm

DIR = Path(os.path.abspath('')).resolve()
ROOT = DIR.parent.parent
DATASET_GLOB = str(ROOT/"data"/"chunked"/"*")
CHUNKS_USED = 1

CLEANR = re.compile('<.*?>')
def clean_html(raw_html):
    return re.sub(CLEANR, '', raw_html)

# Returns an empty string if the key doesnt exist
def try_parse(dict_in, key_in):
    try:
        return dict_in[key_in]
    except KeyError:
        return ''

def load_and_clean_chunked(verbose=False, supervised=True):
    """
        Call this function to get the dataset
        Returns a pandas DataFrame where each row is an entry, with 3 columns (title, body, tags)
        Supervised returns the number of chunks used
        Unsupervised returns the Xs of the unused chunks
    """

    glob_res = glob(DATASET_GLOB)
    if verbose:
        print(f"Loading data for {'un' if not supervised else ''}supervised learning")
        print(f"Taking {CHUNKS_USED if supervised else len(glob_res) - CHUNKS_USED} chunk{'' if supervised else 's'} out of {len(glob_res)}")

    
    files = glob_res[:CHUNKS_USED] if supervised else glob_res[CHUNKS_USED:]

    if len(files) == 0:
        raise ValueError("No files are set aside for unsupervised learning- check the constants in 'load_and_clean_chunked.py'")
    title_list = []
    body_list = []
    tags_list = []
    if verbose:
        print('Loading dataset...')
    for f in tqdm(files, disable=not verbose):
        df_interim = pd.read_csv(f)
        for _, row in df_interim.iterrows():
            parsed_title = unescape(row['Title'])
            parsed_body = clean_html(unescape(row['Body']))
            parsed_tags = unescape(row['Tags'])[1:-1].split('><')
            title_list.append(parsed_title)
            body_list.append(parsed_body)
            tags_list.append(parsed_tags)

    if supervised:
        return pd.DataFrame({
            'title' : title_list,
            'body' : body_list,
            'tags' : tags_list
        })
    else:
        return np.array([title + ' ' + body for  title, body in zip(title_list, body_list)])
if __name__ == "__main__":
    # Load data like this
    df = load_and_clean_chunked(verbose=True)
    print(df)