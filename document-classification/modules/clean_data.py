import re
from nltk import download
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

download("stopwords")
download('wordnet')

class CleanData:
    """
    Text data cleaning using NLP
    """
    @staticmethod
    def _remove_special_chars(doc: str) -> str:
        """ 
        Function to remove special characters from document
        :param: doc: Document
        :return: Cleaned document
        """
        tweet = re.sub("[0-9@,'\.#\)\(\*\?!\$\^\-\_\+\=~]+", "", doc)
        return tweet

    @staticmethod
    def _lemmatize_document(doc: str) -> str:
        """
        Lemmatize document tokens
        :param: doc: Document
        :return: Lemmentized document
        """
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(doc)

    @staticmethod
    def _word_tokenizer(doc: str) -> str:
        """
        NLTK word Tokenizer
        :param: doc: Document
        :return: Tokenized document
        """
        regex_tokenizer = RegexpTokenizer(r"\w{3,}")
        return regex_tokenizer.tokenize(doc)
    
    @staticmethod
    def _remove_stopwords(doc: str) -> str:
        """
        Stop words removal function
        :param: doc: Document
        :return: Document with removed stop words
        """
        stopwords_list = stopwords.words("english")
        for word in doc:
            if word in stopwords_list:
                doc.remove(word)
        return doc

    @staticmethod
    def run(doc: str) -> str:
        """
        Runs the text cleaning process
        :param: doc: Document
        :return: Cleaned document
        """
        doc_lower = doc.lower()
        doc_remove_special_chars = CleanData._remove_special_chars(doc_lower)
        doc_lemmatized = CleanData._lemmatize_document(doc_remove_special_chars)
        doc_tokenized = CleanData._word_tokenizer(doc_lemmatized)
        doc_remove_stop_words = CleanData._remove_stopwords(doc_tokenized)
        doc_clean = " ".join(doc_remove_stop_words)
        return doc_clean
        
        
    

    