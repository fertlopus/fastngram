# Module for corpus forming and sentence concatenation and processing
import os
import argparse
from tqdm import tqdm


REPLACE_DICTIONARY = {
    '\x0b': '\n', '\x0c': '\n',
    '\t': ' ',  '\x85': ' ', '\xa0': ' ',
    '\u2000': ' ', '\u2001': ' ', '\u2002': ' ', '\u2003': ' ', '\u2004': ' ',
    '\u2005': ' ', '\u2006': ' ', '\u2007': ' ', '\u2008': ' ', '\u2009': ' ',
    '\u200a': ' ', '\u2028': ' ', '\u2029': ' ', '\u202f': ' ', '\u205f': ' ',
    '\u3000': ' ',
}

