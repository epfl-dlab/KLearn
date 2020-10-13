import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer(r'\w+')
stopset = frozenset(stopwords.words('english'))

##################################################################
# BASE CLASS
##################################################################


def pre_process_documents(documents):
    new_documents = []
    for d in documents:
        doc = []
        for s in d:
            doc.append(s.get_text())
        new_documents.append(doc)
    return new_documents


class DataConvertor(object):
    def __init__(self, documents, annotations, references):
        self.documents = documents
        self.annotations = annotations
        self.references = references

    def all_keys(self):
        keys = list(self.doc_freq.keys())
        for annot in self.annotations:
            keys.extend(annot['freq'].keys())

        return list(set(keys))

    def normalize_keysets(self, key_set):
        for k in key_set:
            if not(k in self.doc_freq):
                self.doc_freq[k] = 0
            for annot in self.annotations:
                if not(k in annot['freq'].keys()):
                    annot['freq'][k] = 0

    def min_freq(self):
        min_freq = 1
        min_doc = min(i for i in self.doc_freq.values() if i > 0)
        if min_doc < min_freq:
            min_freq = min_doc
        for annot in self.annotations:
            if sum(annot['freq'].values()) == 0:
                continue
            min_tmp = min(i for i in annot['freq'].values() if i > 0)
            if min_tmp < min_freq:
                min_freq = min_tmp
        return min_tmp

    def smooth(self, epsilon):
        self.doc_freq = dict((k, v + epsilon) for k, v in self.doc_freq.items())
        normalizing_cst = float(sum(self.doc_freq.values()))
        self.doc_freq = dict((k, v / normalizing_cst) for k, v in self.doc_freq.items())
        for annot in self.annotations:
            annot['freq'] = dict((k, v + epsilon) for k, v in annot['freq'].items())
            normalizing_cst = float(sum(annot['freq'].values()))
            annot['freq'] = dict((k, v / normalizing_cst) for k, v in annot['freq'].items())

    def get_vectors(self, tgt, optional_key_set=None):
        all_keys = sorted(self.all_keys())
        if optional_key_set:
            all_keys = optional_key_set
        self.normalize_keysets(all_keys)
        t = [self.doc_freq[k] for k in all_keys]
        X, Y = [], []
        for annot in self.annotations:
            X.append([annot['freq'][k] for k in all_keys])
            Y.append(annot[tgt])
        return np.array(t), np.array(X), np.array(Y)

    def key_set(self):
        return sorted(self.all_keys())


##################################################################
# N-Grams
##################################################################

class NgramConvertor(DataConvertor):
    def __init__(self, documents, annotations, references, N):
        super().__init__(documents, annotations, references)
        self.N = N
        self.__convert_annotations()
        self.__convert_documents()

    def is_ngram_content(self, ngram):
        for gram in ngram:
            if not(gram in stopset):
                return True
        return False

    def get_all_content_words(self, sentences):
        all_words = []
        for s in sentences:
            all_words.extend([stemmer.stem(r) for r in tokenizer.tokenize(s.lower())])

        if self.N == 1:
            # content_words = [w for w in all_words if w not in stopset]
            content_words = all_words
        else:
            content_words = all_words

        if self.N > 1:
            return [gram for gram in ngrams(content_words, self.N) if self.is_ngram_content(gram)]
        # if N > 1:
        #     return [gram for gram in ngrams(content_words, N) if self.is_ngram_content(gram)]
        return content_words

    def compute_word_freq(self, words):
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        return word_freq

    def compute_tf(self, sentences):
        content_words = self.get_all_content_words(sentences)  #  stemmed
        content_words_count = len(content_words)
        content_words_freq = self.compute_word_freq(content_words)

        content_word_tf = dict((w, f / float(content_words_count)) for w, f in list(content_words_freq.items()))
        return content_word_tf

    def __convert_documents(self):
        sentences = []
        for d in self.documents:
            sentences.extend(d)
        self.doc_freq = self.compute_tf(sentences)

    def convert_summary(self, summary):
        return self.compute_tf(summary)

    def __convert_annotations(self):
        for annot in self.annotations:
            annot['freq'] = self.convert_summary(annot['text'])


##################################################################
# LDA TOPIC MODELS
##################################################################

class DatasetLDA(object):
    def __init__(self, dataset, n_topics, n_features):
        all_texts = []
        for topic_name, topic in dataset.items():
            documents = topic['documents']
            annotations = topic['annotations']
            all_texts.extend([" ".join(d) for d in pre_process_documents(documents)])
            # for annot in annotations:
            #     all_texts.append(" ".join(annot['text']))

        # all_texts = documents
        # all_texts.extend(summaries)

        self.tf_vectorizer = CountVectorizer(max_features=n_features, stop_words='english')
        self.tf = self.tf_vectorizer.fit_transform(all_texts)
        self.tf_feature_names = self.tf_vectorizer.get_feature_names()

        self.lda = LatentDirichletAllocation(n_components=n_topics,
                                             max_iter=150,
                                             learning_method='online',
                                             topic_word_prior=1 / (2 * n_topics),
                                             learning_offset=50.,
                                             random_state=0).fit(self.tf)

        print('perplexity: {}'.format(self.lda.perplexity(self.tf)))

        self.topic_representations = dict(zip(all_texts, self.lda.transform(self.tf)))

    def topics(self, n_top_words=10):
        topics = []
        for topic in self.lda.components_:
            topics.append(" ".join([self.tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        return topics

    def top_n_topic(self, text, n):
        r = zip(self.topics(), self.lda.transform(self.tf_vectorizer.transform([text]))[0].tolist())
        return sorted(r, key=lambda tup: tup[1], reverse=True)[:n]

    def __call__(self, o):
        if o in self.topic_representations:
            return self.topic_representations[o]
        else:
            # if True:
            # print(o)
            # print(self.tf_vectorizer.transform([o]))
            vec = self.tf_vectorizer.transform([o])
            vec = vec.toarray()[0]
            # for k in range(len(vec)):
            #     if vec[k] != 0:
            #         print(self.tf_feature_names[k])
            # for k in self.top_n_topic(o, 10):
            #     print(k)
            # exit()
            return self.lda.transform(self.tf_vectorizer.transform([o])).tolist()[0]


class LDAConvertor(DataConvertor):
    def __init__(self, documents, annotations, references, dataset_lda, n_features=10):
        super().__init__(documents, annotations, references)
        self.dataset_lda = dataset_lda
        self.n_features = n_features
        self.__convert_annotations()
        self.__convert_documents()

    def __convert_annotations(self):
        for annot in self.annotations:
            annot['freq'] = dict(
                zip(self.dataset_lda.topics(self.n_features), self.dataset_lda(" ".join(s for s in annot['text']))))

    def __convert_documents(self):
        sentences = [" ".join(d) for d in self.documents]
        doc_freqs = [self.dataset_lda(d) for d in sentences]
        self.doc_freq = dict(zip(self.dataset_lda.topics(self.n_features), np.mean(np.array(doc_freqs), axis=0)))
        # self.doc_freq = self.dataset_lda(" ".join(sentences))
