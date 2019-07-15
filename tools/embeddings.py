"""Code for computing embeddings of text."""

import numpy as np
from lha.preprocessing.clean_text import clean_text
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


EMBED_MODES = ["avg", "sent2vec"]


def get_word_vectors(txt, vector_dict, vector_size, weights=None, skip_missing=True):
    vectors = []
    missing_words = []
    for word in txt:
        if word in vector_dict.keys():
            word_embedding = vector_dict[word]
            vectors.append(word_embedding)
        else:
            missing_words.append(word)
            if not skip_missing or weights is not None:
                vectors.append(np.zeros(vector_size))
    if weights is not None:
        return np.array(vectors) * weights[:, None]
    return vectors


def get_word_vector_dic(txt, w2v):
    vectors = {}
    for word in txt:
        if word in w2v.wv.vocab:
            word_embedding = w2v.wv[word]
            vectors[word] = word_embedding
    return vectors


def get_word_vector_list(doc, w2v):
    """Get all the vectors for a text"""
    vectors = []
    for word in doc:
        try:
            vectors.append(w2v.wv[word])
        except KeyError:
            continue
    return vectors


def average_word_embedding(txt, vector_dict, vector_size):
    """Compute averaged embedding"""
    vectors = get_word_vectors(txt, vector_dict, vector_size)
    if len(vectors) == 0:
        # logging.error("Couldn't produce an embedding for {}".format(txt))
        return np.ones(vector_size) * np.nan
    v = np.mean(vectors, 0)
    assert len(v) == vector_size
    return v


def get_tfidf_weights(sentence, model, vocab):
    """Get the tfidf weights of a text, normed to sum to 1."""
    tfidf_weights = model.transform([" ".join(sentence)])
    weights = np.zeros(len(sentence))
    for i, word in enumerate(sentence):
        try:
            word_idx = vocab.get(word)
            weights[i] = tfidf_weights[0, word_idx]
        except IndexError:
            weights[i] = 0.
    s = np.sum(weights)
    if s > 0.:
        weights /= s
    return weights


def tf_idf_top_k_embedding(sentence, vector_dict, vector_size, tfidf_model, top_k=5):
    """Compute tfidf embedding (weighted WCD)."""
    vocab = tfidf_model.get_params()['vect'].vocabulary_
    weights = get_tfidf_weights(sentence, tfidf_model, vocab)
    top_k_words = np.argsort(weights)[-top_k:][::-1]
    sent_vectors = get_word_vectors(sentence, vector_dict, vector_size, skip_missing=False)
    vectors = np.array(sent_vectors)
    k_vectors = np.array([sent_vectors[i] for i in top_k_words])
    try:
        projection = vectors.dot(k_vectors.T)
        v = np.sum(projection, 0)
        if len(v.shape) > 0 and v.shape[0] == top_k:
            return v
        return np.ones(top_k) * np.nan
    except Exception as e:
        print(sentence)
        raise e


def tf_idf_embedding(sentence, vector_dict, vector_size, tfidf_model):
    """Compute tfidf embedding (weighted WCD)."""
    vocab = tfidf_model.get_params()['vect'].vocabulary_
    weights = get_tfidf_weights(sentence, tfidf_model, vocab)
    vectors = get_word_vectors(sentence, vector_dict, vector_size, weights)
    if len(vectors) == 0:
        return np.ones(vector_size) * np.nan
    return np.sum(vectors, 0)


def infersent(sentence, model):
    embeddings = model.encode(sentence, tokenize=True)
    return embeddings


def embed_text(txt, model, vector_dict=None, vector_size=None, mode="sent2vec", clean=True):
    if clean:
        txt = clean_text(txt)

    if vector_dict is None and model is not None and mode != "sent2vec":
        vector_dict = get_word_vector_dic(txt, model)
        vector_size = model.vector_size

    assert mode in EMBED_MODES
    if mode == "avg":
        return average_word_embedding(txt, vector_dict, vector_size)
    elif mode == "sent2vec":
        if type(txt) == list:
            txt = " ".join(txt)
        return model.embed_sentence(txt)
    elif mode == "bert":
        return model(txt)


def embed_bulk_text(txt, w2v, tfidf_model=None, mode="avg", infersent_model=None):
    if tfidf_model is not None:
        mode = "tfidf"
    assert mode in EMBED_MODES
    if mode == "avg":
        return [average_word_embedding(t, w2v) for t in txt]
    elif mode == "tfidf":
        return tf_idf_embedding(txt, w2v, tfidf_model)
