from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import IMDB
import torch
import pandas as pd
from collections import Counter
from typing import Tuple, Iterator

import configs as cfg1
import configs_advanced as cfg2


# Gets the tokenizer
tokenizer = get_tokenizer("basic_english")

def partition_data(
        config: cfg1.Config | cfg2.Config
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions the data into train and test sets
    :param data_df: Dataframe (table) of the data
    :param config: Configuration object
    :returns: Train and test dataframes
    """
    data_df = pd.read_csv(config.all_data_path)
    train_df = data_df.sample(frac=config.test_fraction_split, random_state=config.random_state)
    test_df = data_df.drop(train_df.index)
    return train_df, test_df

# 2. Define iterator for CSV
def yield_tokens(df: pd.DataFrame) -> Iterator:
    for text in df['review']:
        yield tokenizer(text)

def compute_vocabulary(
  config: cfg1.Config | cfg2.Config
  ) -> Tuple[Vocab, Iterator, Iterator]:
    """
    Computes the vocabulary object for a corpus of data
    :param cfg: Configuration object
    :returns: Computed vocabulary
    """    
    # Gets the IMDB train and test iterators
    train_df, test_df = partition_data(config)

    # Gets the vocabulary from the iterator 
    vocab = build_vocab_from_iterator(
        yield_tokens(train_df),
        max_tokens=config.vocab_size,
        specials=["<pad>", "<unk>"]
    )

    # Set the "<unk>" as the default toke
    vocab.set_default_index(vocab["<unk>"])

    return vocab, train_df, test_df

def text_pipeline(
  text: str,
  config: cfg1.Config | cfg2.Config,
  vocab: Vocab
  ) -> torch.Tensor:
    """
    Creates the integer encoding of tokens in a text (sentence)
    :param text: Input text/sentence
    :returns: Tensor of the text encoding
    """
    tokens = tokenizer(text)
    ids = vocab(tokens)
    if len(ids) > config.maxlen:
        ids = ids[:config.maxlen]
    else:
        ids += [vocab["<pad>"]] * (config.maxlen - len(ids))
    return torch.tensor(ids)

def check_labels(df: pd.DataFrame) -> None:
    """
    Checks the label class frequency
    :param df: Dataframe
    """
    labels = df['sentiment'].tolist()
    counter = Counter(labels)
    print(f"Counter items: {counter.items()}")

def create_data_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a dataframe (tabular) representation of the textual data iterator
    :param df: Dataset
    :returns: Dataframe (table)
    """
    labels, reviews = [], []
    for label, line in zip(df['sentiment'], df['review']):
        labels.append(label)
        reviews.append(line)
    data_df = pd.DataFrame({'sentiment': labels, 'review': reviews})
    data_df['sentiment'] = data_df['sentiment'].map({"negative": 0, "positive": 1})
    return data_df


