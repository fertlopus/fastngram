import os
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
import math
from tqdm import tqdm
import multiprocessing
import locale
import argparse
import random
import re
import typing
from preprocessing.main import REPLACE_DICTIONARY

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def clean(line: typing.AnyStr):
    for old_character, new_character in REPLACE_DICTIONARY.items():
        line = line.replace(old_character, new_character)
    line = '_'.join(line.split())
    return line


def get_ngram_occurrence(path: typing.AnyStr, verbose=True):
    ngram_occurrence = {}
    if verbose:
        stats = pd.read_csv(path, sep=' ', names=["ngram", "count"])
        for _, row in tqdm(stats.iterrows(), total=stats.shape[0]):
            ngram_occurrence[row['ngram']] = locale.atoi(row['count'])
    return ngram_occurrence


def get_association(a, b, ngram_occurrence, corpus_length):
    return math.log((ngram_occurrence.get(a+b, 1) * corpus_length) /
                    (ngram_occurrence.get(a, 1) * ngram_occurrence.get(b, 1)))


def stochastic_word_segmentation():
    pass