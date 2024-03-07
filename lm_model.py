import random
from collections import Counter
import numpy as np
import math

"""
CS6120 Homework 2 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int) -> list:  # correct=================================
    """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
    # STUDENTS IMPLEMENT
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[j] for j in range(i, i + n))
        ngrams.append(ngram)
    return ngrams
    pass


def read_file(path: str) -> list:
    """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
    # PROVIDED
    f = open(path, "r", encoding="utf-8")
    contents = f.readlines()
    f.close()
    return contents


def tokenize_line(line: str, ngram: int,
                  by_char: bool = True,
                  sentence_begin: str = SENTENCE_BEGIN,
                  sentence_end: str = SENTENCE_END):
    """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
    # PROVIDED
    inner_pieces = None
    if by_char:
        inner_pieces = list(line)
    else:
        # otherwise split on white space
        inner_pieces = line.split()

    if ngram == 1:
        tokens = [sentence_begin] + inner_pieces + [sentence_end]
    else:
        tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
    # always count the unigrams
    return tokens


def tokenize(data: list, ngram: int,
             by_char: bool = True,
             sentence_begin: str = SENTENCE_BEGIN,
             sentence_end: str = SENTENCE_END):
    """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
    # PROVIDED
    total = []
    # also glue on sentence begin and end items
    for line in data:
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue
        tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
        total += tokens
    return total


class LanguageModel:

    def __init__(self, n_gram):
        """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
        # STUDENTS IMPLEMENT

        self.n_gram = n_gram
        self.n_gram_counts = {}
        self.context_counts = {}  # model
        self.token_counts = {}  # Stores counts of individual tokens

    # ****************************************************************************************************************
    def train(self, tokens: list, verbose: bool = False) -> None:
        """Trains the language model on the given data. Assumes that the given data
            has tokens that are white-space separated, has one sentence per line, and
            that the sentences begin with <s> and end with </s>
        Args:
        tokens (list): tokenized data to be trained on as a single list
        verbose (bool): default value False, to be used to turn on/off debugging prints"""

        # Replace single-occurrence tokens with UNK in the token list
        self.token_counts = Counter(tokens)
        tokens = [UNK if self.token_counts[t] < 2 else t for t in tokens]
        self.token_counts = dict(Counter(tokens))
        # print(tokens)

        # Create and count n-grams with UNK replacements
        n_grams = create_ngrams(tokens, self.n_gram)
        # print(n_grams)
        for n in n_grams:
            self.n_gram_counts[n] = self.n_gram_counts.get(n, 0) + 1

        # print(self.n_gram_counts)

        if self.n_gram > 1:
            context = create_ngrams(tokens, self.n_gram - 1)
            for n in context:
                self.context_counts[n] = self.context_counts.get(n, 0) + 1

        # print(self.context_counts)
        # print(f"{self.n_gram_counts=}")
        # print(f"{self.context_counts=}")

        # Verbose output
        if verbose:
            print(f"Trained on {len(tokens)} tokens, with {len(tokens)} after UNK replacement.")
            print(f"Total unique {self.n_gram}-grams: {len(self.n_gram_counts)}")
            if self.n_gram > 1:
                print(f"Total unique {self.n_gram - 1}-grams: {len(self.context_counts)}")

    # ****************************************************************************************************************
    def score(self, sentence_tokens: list) -> float:
        """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model

    Returns:
      float: the probability value of the given tokens for this model
    """

        v = len(set(self.token_counts.keys()))
        # print('v=', v)

        # Initialize the probability
        prob_score = 1.0
        # print(f'{sentence_tokens=}')

        for i in range(len(sentence_tokens)):
            if sentence_tokens[i] not in self.token_counts:
                sentence_tokens[i] = UNK

        # print(f'{sentence_tokens=}')

        # Create n-grams from the sentence tokens
        grams = create_ngrams(sentence_tokens, self.n_gram)
        # print(f"{grams=}")

        for token_set in grams:
            # print(f'{token_set=}')
            context = token_set[:-1]
            word = token_set[-1]
            # print(context,'----context-------')
            #finding the numerator and denominator if context is present or not:
            try:
                if context:
                    denom = self.context_counts[context]
            except KeyError:
                denom = 0

            try:
                if not context:
                    denom = sum(self.n_gram_counts.values())
            except KeyError:
                denom = 0

            try:
                if not context:
                    numerate = self.token_counts[word]

            except KeyError:
                numerate = self.token_counts[UNK]

            try:
                if context:
                    numerate = self.n_gram_counts[token_set]


            except KeyError:
                numerate = 0

            # Laplace Smoothing
            # print(f"{numerate=} {denom=}")
            set_prob = (numerate + 1) / (denom + v)

            prob_score *= set_prob

        return prob_score

    # *********************************************************************************************************************
    def generate_sentence(self) -> list:
        """Generates a single sentence from a trained language model using the Shannon technique.

    Returns:
      list: the generated sentence as a list of tokens
    """
        # print(f'{self.n_gram=}')
        sentence = ([SENTENCE_BEGIN] * (self.n_gram - 1)) if self.n_gram > 1 else [SENTENCE_BEGIN]
        # print(f'{sentence=}')
        word_chosen = None

        while word_chosen != SENTENCE_END:
            words_possible = {}
            window_start = len(sentence) - self.n_gram + 1

            gram_window = tuple(sentence[window_start:])
            # print(f'{gram_window=}')

            # Iterating through the dictionary to print keys and values together
            for key, value in self.n_gram_counts.items():
                target = key[:-1]
                if gram_window == target:
                    words_possible[key[-1]] = value

            # print(f'{words_possible}')
            # words_possible = dict(filter(lambda item: item[0] != SENTENCE_BEGIN, words_possible.items()))
            words_possible = dict(filter(lambda item: item[0] != SENTENCE_BEGIN, words_possible.items()))
            total_context_words = sum(words_possible.values())
            # print(f'{total_context_words=}')

            probablities = [(v / total_context_words) for v in words_possible.values()]
            # print(f'{probablities=}')
            words_possible = list(words_possible.keys())
            # print(words_possible)

            words_possible_idx = list(range(len(words_possible)))
            # print(f'{words_possible_idx=}')
            word_chosen_idx = np.random.choice(words_possible_idx, size=1, p=probablities)
            # print(f'{word_chosen_idx=}')

            word_chosen = words_possible[word_chosen_idx[0]]
            sentence.append(word_chosen)
            # print(sentence)

        sentence = sentence + ([SENTENCE_END] * (self.n_gram - 2))
        return sentence

    # *****************************************************************************************************************
    def generate(self, n: int) -> list:
        """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate

    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
        # PROVIDED
        return [self.generate_sentence() for i in range(n)]

    # ****************************************************************************************************************
    def perplexity(self, sequence: list) -> float:
        """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model

    Returns:
      float: the perplexity value of the given sequence for this model
    """
        # STUDENT IMPLEMENT
        perplex = 1 / self.score(sequence)
        perplexity = perplex ** (1 / len(sequence))
        # print(f'{perplexity=}')

        return perplexity

        pass


# not required
if __name__ == '__main__':
    print()
    print("if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)")
    print("call a function")
    print(tokenize_line("tokenize this sentence!", 2, by_char=False))
    print(tokenize(["apples are fruit", "bananas are too"], 2, by_char=False))

