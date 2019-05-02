import numpy as np


class TfIdf:
    def __init__(self, lowercase=True):
        self.lowercase = lowercase
        self.word_num = dict()
        self.word_list = list()
        self.word_in_doc = dict()

    def get_total_words(self, corpus):
        # lower case every word
        if self.lowercase:
            lower_corpus = [sent.lower() for sent in corpus]
        # first split string to list
        for sent in lower_corpus:
            doc = sent.split()
            for word in doc:
                if word not in self.word_list:
                    self.word_list.append(word)
        # get index of every word in the corpus
        self.word_num = {word: i for i, word in enumerate(self.word_list)}
        # count number of documents that word appears in
        for word in self.word_list:
            for sent in lower_corpus:
                if word in sent:
                    self.word_in_doc[word] = self.word_in_doc.get(word, 0) + 1

    def get_tf_idf(self, corpus):
        if self.lowercase:
            lower_corpus = [sent.lower() for sent in corpus]
        self.get_total_words(corpus)
        corpus_size = len(corpus)
        x_tf = np.zeros(shape=(corpus_size, len(self.word_list)))
        x_idf = np.zeros(shape=(corpus_size, len(self.word_list)))
        for i in range(len(corpus)):
            doc = lower_corpus[i].split()
            for word in doc:
                idx = self.word_num[word]
                x_tf[i, idx] += 1
                x_idf[i, idx] = np.log((1 + corpus_size) / (1 + self.word_in_doc[word]))
        return x_tf * x_idf


if __name__ == "__main__":
    corpus = ['This is the first document',
              'This document is the second document',
              'And this is the third one', 'Is this the first document']
    tfidf_transformer = TfIdf()
    x = tfidf_transformer.get_tf_idf(corpus)
    print(x)
