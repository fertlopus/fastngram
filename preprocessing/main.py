# Module for corpus forming and sentence concatenation and processing
import os
import argparse
from tqdm import tqdm
import typing


REPLACE_DICTIONARY = {
    '\x0b': '\n', '\x0c': '\n',
    '\t': ' ',  '\x85': ' ', '\xa0': ' ',
    '\u2000': ' ', '\u2001': ' ', '\u2002': ' ', '\u2003': ' ', '\u2004': ' ',
    '\u2005': ' ', '\u2006': ' ', '\u2007': ' ', '\u2008': ' ', '\u2009': ' ',
    '\u200a': ' ', '\u2028': ' ', '\u2029': ' ', '\u202f': ' ', '\u205f': ' ',
    '\u3000': ' ',
}


def process_line(line: typing.AnyStr):
    line = line.strip()
    for old_character, new_character in REPLACE_DICTIONARY.items():
        line = line.replace(old_character, new_character)
    line = ' '.join(line.split())
    line = '\n'.join(line.split('\n'))
    lines = [l.strip() for l in line.split('\n') if l.strip() != '']
    return lines


def process_corpora(configuration):
    processed_corpus = []
    with open(configuration.corpora_path, mode='r', encoding=configuration.encoding, errors='ignore') as file:
        lines = file.readlines()
    if configuration.verbose:
        bar = tqdm(desc="Replacing spaces into another character.",  total=len(lines), mininterval=1,
                   bar_format="{desc}: {percentage:3.0f}% ({remaining} left")
    for line in lines:
        if configuration.verbose:
            bar.update()
        processed_lines = process_line(line)
        processed_corpus.extend(processed_lines)

    processed_corpus = '_'.join(processed_corpus)
    processed_corpus = '_'.join(processed_corpus.split())
    processed_corpus = processed_corpus.strip()

    with open(configuration.processed_path, mode='w', encoding=configuration.encoding, errors='ignore') as final_file:
        final_file.write(processed_corpus)

    if configuration.verbose:
        bar.close()

    file.close()
    final_file.close()
    print("Finished.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WNE_embeddings/preprocessing/main.py')
    parser.add_argument('--corpora_path', type=str, default='../data/raw/sample.txt', help='Corpora Path in str format')
    parser.add_argument('--processed_path', type=str, default='../data/processed/processed.txt', help='Output in str')
    parser.add_argument('--encoding', type=str, default='utf-8', help='Encoding for the text file. Default is utf-8')
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose option')
    configuration = parser.parse_args()
    print(configuration)
    process_corpora(configuration=configuration)
