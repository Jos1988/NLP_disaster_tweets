import string
import re
import logging
import multiprocessing as mp
from abc import ABC, abstractmethod
from copy import copy
from typing import Union, Callable, List
from itertools import product

from nltk import TweetTokenizer, SnowballStemmer
from nltk.corpus import stopwords


class mp_transformer(ABC):

    def __init__(self, processes: Union[None, int] = None, verbosity: int = 1):
        self.verbosity = verbosity

        if processes is None:
            processes = mp.cpu_count() - 1

        self.processes = processes

    def fit(self, X):
        pass

    @abstractmethod
    def pre_process(self, X):
        pass

    @abstractmethod
    def get_process_fn(self) -> Callable:
        pass

    @abstractmethod
    def post_process(self, result):
        pass

    def transform(self, X):
        if 0 == len(X):
            return X

        self.X = copy(X)

        input = self.pre_process(X)

        if self.verbosity > 0:
            logging.info(f'{self.__class__.__name__} transforming data on {self.processes} processes.')

        pool = mp.Pool(self.processes)
        result = pool.map(self.get_process_fn(), input)
        pool.close()

        result = self.post_process(result)

        return result


def tokenize_process(input) -> list:
    tokenizer, input = input
    return tokenizer.tokenize(input)


class tokenizer(mp_transformer):

    def __init__(self, processes: Union[None, int] = None, verbosity: int = 1):
        super().__init__(processes, verbosity)
        self.tokenizer = TweetTokenizer(reduce_len=True)

    def pre_process(self, X):
        return [(self.tokenizer, text) for text in X['text'].values]

    def get_process_fn(self) -> Callable:
        return tokenize_process

    def post_process(self, result):
        self.X['tokens'] = result
        return self.X


def process_urls(input):
    return [word for word in input if word[0:4] != 'http']


class urlRemover(mp_transformer):

    def pre_process(self, X):
        return X['tokens']

    def get_process_fn(self) -> Callable:
        return process_urls

    def post_process(self, result):
        self.X['tokens'] = result
        return self.X


def process_hashtags(input):
    return [word for word in input if word[0] == '#']


class hashtagSeparator(mp_transformer):

    def pre_process(self, X):
        return X['tokens']

    def get_process_fn(self) -> Callable:
        return process_hashtags

    def post_process(self, result):
        self.X['hashtags'] = result
        return self.X


def process_mentions(input):
    return [word for word in input if word[0] == '@']


class mentionSeparator(mp_transformer):

    def pre_process(self, X):
        return X['tokens']

    def get_process_fn(self) -> Callable:
        return process_mentions

    def post_process(self, result):
        self.X['mentions'] = result
        return self.X


# translation_target = ' '*len(string.punctuation)


# def process_punctuation(input):
#     return [word.translate(str.maketrans(string.punctuation, translation_target)) for word in input]


def process_punctuation(input):
    results = []
    for word in input:
        result = re.sub("'", '', word)
        result = re.sub(f'[{string.punctuation}]+', ' ', result)
        results = results + result.split()

    return results


class punctuationRemover(mp_transformer):

    def pre_process(self, X):
        return X['tokens']

    def get_process_fn(self) -> Callable:
        return process_punctuation

    def post_process(self, result):
        self.X['tokens'] = result
        return self.X


class WordSplitter:
    def __init__(self, fastTextModel, max_split):
        self.max_split = max_split
        self.fastTextModel = fastTextModel
        self.log = {}

    def split_snake_case(self, word: str):
        if len(word) < 2:
            return [word]

        word = word[0].upper() + word[1:]
        split = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', word)
        return split

    def split_composite_word(self, word: str):
        all_substrings = [word[i: j] for i in range(len(word)) for j in range(i + 1, len(word) + 1)]
        all_subwords = [substring for substring in all_substrings if len(substring) > 2 and substring in self.fastTextModel]
        potential_splits = []
        for split_size in range(self.max_split):
            for split in product(*[all_subwords for i in range(split_size)]):
                if word == ''.join(split):
                    potential_splits.append(list(split))

            if len(potential_splits) > 0:
                break

        potential_splits.append([word])
        return potential_splits[0]

    def split_words(self, words: List[str]):
        result = []
        for word in words:
            if word in self.fastTextModel:
                result.append(word)
                continue

            split = self.split_snake_case(word)
            if len(split) > 1:
                result = result + split
                self.log[word] = split
                continue

            split = self.split_composite_word(word)
            if len(split) > 1:
                self.log[word] = split

            result = result + split
        return result


def split_words(input):
    tokens, splitter = input
    return splitter.split_words(tokens)


class WordSplitting(mp_transformer):

    def __init__(self, fastText_model, processes: Union[None, int] = None, verbosity: int = 1):
        super().__init__(processes, verbosity)
        self.splitter = WordSplitter(fastText_model.get_words(), 4)

    def pre_process(self, X):
        return [(tokens, self.splitter) for tokens in X['tokens']]

    def get_process_fn(self) -> Callable:
        return split_words

    def post_process(self, result):
        self.X['tokens'] = result
        return self.X


def split_snake_case(input):
    tokens, splitter = input
    new_tokens = []
    for word in tokens:
        result = splitter.split_snake_case(word)
        if len(result) > 1:
            new_tokens = new_tokens + result

    return new_tokens


class SnakeCaseSplitting(mp_transformer):

    def __init__(self, processes: Union[None, int] = None, verbosity: int = 1):
        super().__init__(processes, verbosity)
        self.splitter = WordSplitter([], 4)

    def pre_process(self, X):
        return [(tokens, self.splitter) for tokens in X['tokens']]

    def get_process_fn(self) -> Callable:
        return split_snake_case

    def post_process(self, result):
        self.X['tokens'] = result
        return self.X


def process_numeric_filter(input):
    return [word for word in input if len(word) > 1 and word.isnumeric() is False]


class numericsFilter(mp_transformer):

    def pre_process(self, X):
        return X['tokens']

    def get_process_fn(self) -> Callable:
        return process_numeric_filter

    def post_process(self, result):
        self.X['tokens'] = result
        return self.X


def process_lowering(input):
    return [word.lower() for word in input]


class textLowerer(mp_transformer):

    def pre_process(self, X):
        return X['tokens']

    def get_process_fn(self) -> Callable:
        return process_lowering

    def post_process(self, result):
        self.X['tokens'] = result
        return self.X


def process_stopword_filter(input):
    tokens, stopwords_list = input
    return [word for word in tokens if word not in stopwords_list]


class stopwordsFilter(mp_transformer):

    def pre_process(self, X):
        stopwords_list = stopwords.words('english')
        return [(tokens, stopwords_list) for tokens in X['tokens']]

    def get_process_fn(self) -> Callable:
        return process_stopword_filter

    def post_process(self, result):
        self.X['tokens'] = result
        return self.X


def process_stemming(input):
    tokens, stemmer = input
    return [stemmer.stem(word) for word in tokens]


class stemmer(mp_transformer):

    def pre_process(self, X):
        snowball_stemmer = SnowballStemmer('english')
        return [(tokens, snowball_stemmer) for tokens in X['tokens']]

    def get_process_fn(self) -> Callable:
        return process_stemming

    def post_process(self, result):
        self.X['tokens'] = result
        return self.X
