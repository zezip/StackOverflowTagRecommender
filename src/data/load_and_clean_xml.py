import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import os.path
import re
from tqdm import tqdm

DIR = Path(os.path.abspath('')).resolve()
ROOT = DIR.parent.parent
# Change middle element in path to change dataset
DATASET = str(ROOT/"data"/"meta.stackoverflow.com"/"Posts.xml")

CLEANR = re.compile('<.*?>')
def cleanhtml(raw_html):
    return re.sub(CLEANR, '', raw_html)

# Returns an empty string if the key doesnt exist
def try_parse(dict_in, key_in):
    try:
        return dict_in[key_in]
    except KeyError:
        return ''

def load_and_clean(require_title=True, require_body=True, require_tags=True, verbose=False):
    """
        Call this function to get the dataset
        Returns a pandas DataFrame where each row is an entry, with 3 columns (title, body, tags)
        Uses arguments to specify if an entry is required for each field
        If an entry is null and not required, returns an empty string (list for tags)
    """
    raise DeprecationWarning("This function is deprecated because the data dump is corrupted \
                            (but kept because we might have need for xml parsing later). \
                            Use \"load_and_clean_chunked.py\" and the associated function instead")
    df = load_and_parse_dataset(verbose)
    title_list = []
    body_list = []
    tags_list = []
    if verbose:
        print('Filtering dataset...')
    for _, row in tqdm(df.iterrows(), disable=not verbose, total=len(df)):
        if require_title and row['title'] == '':
            continue
        elif require_body and row['body'] == '':
            continue
        elif require_tags and len(row['tags']) == 0:
            continue
        else:
            title_list.append(row['title'])
            body_list.append(row['body'])
            tags_list.append(row['tags'])
    return pd.DataFrame({
        'title' : title_list,
        'body' : body_list,
        'tags' : tags_list
    })
    print('df')

def load_and_parse_dataset(verbose):
    """
        Loads and parses the dataset:
            - Takes in the .xml file specified by the DATASET global var
            - Returns a pandas DataFrame where each row is an entry, with 3 columns (title, body, tags)
        If there is not an entry for an element, gets an empty string
    """
    tree = ET.parse(DATASET)
    root = tree.getroot()
    title_list = []
    body_list = []
    tag_list = []
    if verbose:
        print('Loading dataset...')
    for child in tqdm(root, disable=not verbose):
        title = try_parse(child.attrib, 'Title')
        body = try_parse(child.attrib, 'Body')
        tags = try_parse(child.attrib, 'Tags')
        parsed_body = cleanhtml(body)
        parsed_tags = tags[1:-1].split('><')
        title_list.append(title)
        body_list.append(parsed_body)
        tag_list.append(parsed_tags)
    return pd.DataFrame({
        'title' : title_list,
        'body' : body_list,
        'tags' : tag_list
    })

if __name__ == "__main__":
    print(load_and_clean(verbose=True))