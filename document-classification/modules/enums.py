from enum import Enum


class VectorizerType(Enum):
    """
    Enum for different types of text vectorization methods.
    """
    TFIDF = "tfidf"
    COUNT = "count"
    WORD2VEC = "word2vec"
    GLOVE = "glove"