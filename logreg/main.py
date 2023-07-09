import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV as LRC
import math
from tqdm import tqdm
import multiprocessing
import locale
import argparse
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
    return math.log((ngram_occurrence.get(a + b, 1) * corpus_length) /
                    (ngram_occurrence.get(a, 1) * ngram_occurrence.get(b, 1)))


def stochastic_word_segmentation(process_num, indexes, processed_corpora, config, ngram_occurrences, corpora_length):
    verbose = False
    if process_num == 0:
        verbose = True
    if verbose:
        bar = tqdm(desc="Word Boundaries", total=len(indexes), mininterval=1,
                   bar_format='{desc}: {percentage:3.0f}% ({remaining} left')
    X_each = np.empty((len(indexes), config.max_n ** 2))
    for i, j in enumerate(indexes):
        if verbose:
            bar.update()
        explanatory_vars = list()
        for a in range(1, config.max_n + 1):
            for b in range(1, config.max_n + 1):
                explanatory_vars.append(get_association(
                    processed_corpora[j - a:j],
                    processed_corpora[j:j + b],
                    ngram_occurrences,
                    corpora_length
                ))
        X_each[i] = np.array(explanatory_vars)
    if verbose:
        bar.close()
    return X_each


def read_file(file_path: typing.AnyStr):
    with open(file_path, mode='r') as file:
        return file.readlines()


def write_to_file(file_path: typing.AnyStr, data: typing.Any):
    with h5py.File(file_path, mode='w') as file:
        info = file.create_dataset('word_boundaries', data=data)


def main():
    parser = argparse.ArgumentParser(description="Probability predictions of word boundaries for WNE algorithm")
    parser.add_argument('--corpora_path', type=str, default="../data/raw/sample.txt", required=True)
    parser.add_argument('--segmented_corpus_path', type=str, default="../data/processed/segmented.txt")
    parser.add_argument('--ngram_count_path', type=str, default='../data/processed/ngram_frequency.csv')
    parser.add_argument('--processed_path', type=str, default='../data/processed/processed.txt', required=True)
    parser.add_argument('--word_boundaries', type=str, default='../data/processed/word_boundaries.hdf5', required=True)
    parser.add_argument('--ratio', type=float, default=0.15, required=True)
    parser.add_argument('--random_seed', type=int, default=42, required=False)
    parser.add_argument('--max_n', type=int, default=4, required=False)
    parser.add_argument('--n_jobs', type=int, default=8, required=False)
    configuration = parser.parse_args()
    np.random.seed(configuration.random_seed)

    # gathering the data
    sentences = read_file(configuration.corpora_path)
    segmented_sentences = read_file(configuration.segmented_corpus_path)
    concatenated_sentences = str()
    labels = list()

    if len(sentences) != len(segmented_sentences):
        raise ValueError("All arrays must be the same length")

    usage_sentence_num = int(len(sentences) * configuration.ratio)
    index = np.random.randint(0, len(sentences) - usage_sentence_num)
    sentences = sentences[index:index + usage_sentence_num]
    segmented_sentences = segmented_sentences[index:index + usage_sentence_num]
    print("Using {}% of the corpora ({} sentences)".format(int(configuration.ratio * 100),
                                                           len(sentences)))

    for i, sentence in enumerate(sentences):
        sentence = clean(sentence.strip() + "_")
        concatenated_sentences += sentence
        segmented_sentence = clean(segmented_sentences[i].strip() + "_")
        gap = 0
        is_new = 1
        label = list()
        for j, character in enumerate(sentence):
            character_segmented_sentence = segmented_sentence[j + gap]
            if character == "_":
                is_new = 1
                label.append(is_new)
                if character != character_segmented_sentence:
                    gap -= 1
                continue
            while character != character_segmented_sentence:
                gap += 1
                is_new = 1
                character_segmented_sentence = segmented_sentence[j + gap]
            label.append(is_new)
            is_new = 0
        labels.extend(label)

    # processed corpora
    processed_corpora = read_file(configuration.ngram_count_path)[0]
    corpora_length = len(processed_corpora)
    ngram_occurrence = get_ngram_occurrence(configuration.ngram_count_path)

    # Forming matrix X and y labels for logistic regression model
    X = np.empty((len(concatenated_sentences) - configuration.max_n - (configuration.max_n - 1),
                  configuration.max_n ** 2))
    y = np.array(labels)[configuration.max_n:-(configuration.max_n - 1)]
    for i in range(configuration.max_n, len(concatenated_sentences) - (configuration.max_n - 1)):
        explanatory_vars = list()
        for a in range(1, configuration.max_n + 1):
            for b in range(1, configuration.max_n + 1):
                explanatory_vars.append(get_association(concatenated_sentences[i - a:i],
                                                        concatenated_sentences[i:i + b],
                                                        ngram_occurrence,
                                                        corpora_length))
        X[i - configuration.max_n] = np.array(explanatory_vars)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, shuffle=True)

    # model training
    lr = LRC(cv=5, n_jobs=-1, verbose=False)
    lr.fit(X_train, y_train)

    word_boundaries = np.array([1])
    p = multiprocessing.Pool(configuration.n_jobs)
    chunk_size = int(((len(processed_corpora) - 1) / configuration.n_jobs) + 1)
    output = p.map(stochastic_word_segmentation,
                   [(process_num, range(len(processed_corpora))[i:i + chunk_size],
                     processed_corpora, configuration, ngram_occurrence, corpora_length)
                    for process_num, i in enumerate(range(1, len(processed_corpora), chunk_size))])

    # Final preprocessing
    X = np.vstack(output)
    word_boundaries = np.hstack((word_boundaries, lr.predict_proba(X)[:, 1]))
    print("The process is finished. Sample: ", word_boundaries[:10])
    # Save the model
    write_to_file(configuration.word_boundaries, word_boundaries)


if __name__ == "__main__":
    main()
