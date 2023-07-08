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
    """
    Function to preprocess the text file. Replace the character
    in read line and joining them into one array.
    :param line: str
    :return: an array of preprocessed strings
    """
    line = line.strip()
    for old_character, new_character in REPLACE_DICTIONARY.items():
        line = line.replace(old_character, new_character)
    line = ' '.join(line.split())
    line = '\n'.join(line.split('\n'))
    lines = [l.strip() for l in line.split('\n') if l.strip() != '']
    return lines

