from numpy.random import seed
seed(1)

import tensorflow as tf
tf.random.set_seed(1)
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D
import matplotlib.pyplot as plt
import numpy as np
import sys
import os, os.path
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

from keras.layers import Input
from keras.models import Model

DIR = Path(os.path.abspath('')).resolve()
SRC = DIR.parent
ROOT = SRC.parent
GLOVE = str(ROOT/"models"/"glove.6B"/"glove.6B.100D.txt")
sys.path.append(str(SRC/"data"))

from get_split import get_split

# Constants
GLOVE_EMBEDDING_SIZE = 100

# Shared Hyperparameters
MAX_SEQUENCE_LEN = 200
MAX_WORDS = 30000
LSTM_SIZE = 128
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# Vanilla Hyperparameters
VANILLA_EPOCHS = 25

# Conv Hyperparameters
CONV_EPOCHS = 36
NUM_FILTERS = 64
KERNEL_SIZE = 5

def run_models(X_train, y_train, X_test, y_test, verbose=False):
    num_labels = y_train.shape[1]

    if verbose:
        print('Tokenizing...')
    X_train, X_test, vocab_size, tokenizer = tokenize_and_pad(X_train, X_test)

    if verbose:
        print('Loading GloVe embeddings...')
    embedding_matrix = build_embedding_matrix(vocab_size, tokenizer)

    if verbose:
        print('Compiling models...')

    # vanilla_lstm = compile_vanilla_functional_model(embedding_matrix, vocab_size, num_labels)
    conv_lstm = compile_conv_functional_model(embedding_matrix, vocab_size, num_labels)
    
    # run_model('vanilla_lstm', vanilla_lstm, X_train, y_train, X_test, y_test, verbose=verbose)
    run_model('conv_lstm', conv_lstm, X_train, y_train, X_test, y_test, verbose=verbose)

    
def run_model(name, model, X_train, y_train, X_test, y_test, verbose=False):
    if verbose:
        print('Training...')
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=VANILLA_EPOCHS if name == 'vanilla_lstm' else CONV_EPOCHS, verbose=1, validation_split=VALIDATION_SPLIT)
    if verbose:
        print('Scoring...')

    score = model.evaluate(X_test, y_test, verbose=1)
    y_pred_logits = model.predict(X_test)
    y_pred = (y_pred_logits > 0.5).astype(int)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)
    print('-------------------')
    print(f"{name}_accuracy:", score[1])
    print(f'{name}_precision:', np.mean(precision))
    print(f'{name}_recall:', np.mean(recall))
    print(f'{name}_f1:', np.mean(fscore))
    print('-------------------')
    plot_history(history, name)
    tf.keras.utils.plot_model(model, to_file=f'{name}_model.png', show_shapes=True, show_layer_names=True)
    
def plot_history(history, name):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(f'{name}_training_accuracy.png')
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    plt.savefig(f'{name}_training_loss.png')
    plt.close()

def compile_vanilla_functional_model(embedding_matrix, vocab_size, num_labels):
    deep_inputs = Input(shape=(MAX_SEQUENCE_LEN,), dtype='int32')
    
    embedding_layer = Embedding(vocab_size, GLOVE_EMBEDDING_SIZE, input_length=MAX_SEQUENCE_LEN, weights=[embedding_matrix], trainable=False)(deep_inputs)
    LSTM_layer = LSTM(LSTM_SIZE,)(embedding_layer)
    dense_layer = Dense(num_labels, activation='sigmoid')(LSTM_layer)
    model = Model(inputs=deep_inputs, outputs=dense_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def compile_conv_functional_model(embedding_matrix, vocab_size, num_labels):
    deep_inputs = Input(shape=(MAX_SEQUENCE_LEN,))
    embedding_layer = Embedding(vocab_size, GLOVE_EMBEDDING_SIZE, weights=[embedding_matrix], trainable=False)(deep_inputs)
    conv_layer = Conv1D(NUM_FILTERS, KERNEL_SIZE, activation='relu')(embedding_layer)
    LSTM_layer = LSTM(LSTM_SIZE,)(conv_layer)
    dense_layer = Dense(num_labels, activation='sigmoid')(LSTM_layer)
    model = Model(inputs=deep_inputs, outputs=dense_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def build_embedding_matrix(vocab_size, tokenizer):
    embeddings_dictionary = dict()
    with open(GLOVE, encoding="utf8") as glove_file:
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions

    embedding_matrix = np.zeros((vocab_size, GLOVE_EMBEDDING_SIZE))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

    
def tokenize_and_pad(X_train, X_test):
    X_train = X_train.flatten()
    X_test = X_test.flatten()
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    vocab_size = len(tokenizer.word_index) + 1
    X_train = pad_sequences(X_train, padding='post', maxlen=MAX_SEQUENCE_LEN)
    X_test = pad_sequences(X_test, padding='post', maxlen=MAX_SEQUENCE_LEN)

    return X_train, X_test, vocab_size, tokenizer


def main():
    (X_train, y_train, X_test, y_test), index_to_tag = get_split()
    run_models(X_train, y_train, X_test, y_test, verbose=True)

if __name__ == "__main__":
    main()