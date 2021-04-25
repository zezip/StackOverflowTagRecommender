#!/bin/bash
chmod +x get_glove.sh;
./get_glove.sh;
cd src;
echo "Running tokenizer.py..."
python tokenizer.py
echo "Running lda_bpe_trials.py..."
python lda_bpe_trials.py
echo "Running lstm.py..."
python lstm.py