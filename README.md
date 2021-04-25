# StackOverflowTagRecommender
EECS 476 term project by Zach Zipper and Yoko Nagafuchi. Predicting tags for Stack Overflow posts using multinomial models. We try multiple
tokenization strategies (sklearn's Vectorizers and a Byte-Pair Encoding tokenizer, trained on 250,000 posts). We try multiple dimensionality
reduction strategies (SVD and topic modeling with LDA). We also try multiple classification models (SVMs and RNNs).

Dataset sourced from https://archive-org.proxy.lib.umich.edu/details/stackexchange and https://data.stackexchange.com/stackoverflow/queries.

To run all main experiments: `chmod +x experiments.sh; ./experiments.sh` (Note that running this will download a ~1GB copy of GloVe)

Alternatively, you can run the python files outlined `experiments.sh` individually (in the case that one is incompatible with your environment)

To remake the preset data split: `python src/data/make_split.py`

To retrain the BPE encoder on the unsupervised dataset: `python src/models/bpe_tokenizer.py`