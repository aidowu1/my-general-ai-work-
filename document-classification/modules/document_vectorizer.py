from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import pandas as pd
from typing import List

from modules.enums import VectorizerType


class DocumentVectorizer:
    """
    A class to convert text documents into numerical vectors using different vectorization techniques.
    """

    def __init__(
            self, 
            vectorizer_type: VectorizerType = VectorizerType.COUNT, 
            max_features: int = 5000
        ):
        """
        Initializes the DocumentVectorizer with the specified vectorization method.
        :param method: The vectorization method to use ('count' or 'tfidf').
        :param max_features: The maximum number of features to consider.
        """
        self._vectorizer_type = vectorizer_type
        self._documents_as_array = None
        self._new_documents_as_array = None
        if self._vectorizer_type == VectorizerType.COUNT:
            self.vectorizer = CountVectorizer(max_features=max_features)
        elif self._vectorizer_type == VectorizerType.TFIDF:
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            raise ValueError("Unsupported vectorization method. Use 'COUNT' or 'TFIDF'.")

    def fit_transform(self, documents: List[str]) -> scipy.sparse.csr.csr_matrix:
        """
        Fits the vectorizer to the documents and transforms them into vectors.
        :param documents: A list of text documents.
        :return: A sparse matrix of shape (n_samples, n_features).
        """
        self._documents_as_array = self.vectorizer.fit_transform(documents)
        return self._documents_as_array

    def transform(self, documents: List[str]) -> scipy.sparse.csr.csr_matrix:
        """
        Transforms new documents into vectors using the fitted vectorizer.
        :param documents: A list of text documents.
        :return: A sparse matrix of shape (n_samples, n_features).
        """
        self._new_documents_as_array = self.vectorizer.transform(documents)
        return self._new_documents_as_array
    
    @property
    def document_term_matrix(self) -> pd.DataFrame:
        """
        Returns the document-term matrix.
        :return: The document-term matrix as a table.
        """
        if self._vectorizer_type == VectorizerType.COUNT or self._vectorizer_type == VectorizerType.TFIDF:
            dtm = pd.DataFrame(self._documents_as_array.toarray(),
                    columns=self.vectorizer.get_feature_names_out())
            return dtm
        else:
             raise ValueError("Vectorization methods other than 'BOW' or 'TF-IDF' do not support the dtm property")   
        
        