"""Implements functions for computing similarity scores between texts."""

import logging

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from lha.tools.embeddings import get_word_vector_list
from lha.preprocessing.clean_text import clean_stem, clean_text

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# All the currently implemented metrics
METRICS_IMPLEMENTED = [None, "cosine", "wmd", "rwmd", "jaccard", "jaccard_stem", "copy_rate", "nlev",
                            "rwmd_right", "rwmd_left", "vocab_score", "rwmd_normed", "classifier", "bm25"]
WORD_METRICS = ["rwmd", "wmd", "rwmd_right", "rwmd_left", "rwmd_normed", "jaccard", "jaccard_stem", "copy_rate",
                     "nlev", "vocab_score", "classifier"]
SENT_METRICS = ["cosine"]
EMB_METRICS = ["cosine", "wmd", "rwmd", "rwmd_normed"]


def set_overlap(source_set, target_set):
    """Compute the overlap score between a source and a target set.
    It is the intersection of the two sets, divided by the length of the target set."""
    word_overlap = target_set.intersection(source_set)
    overlap = len(word_overlap) / float(len(target_set))
    assert 0. <= overlap <= 1.
    return overlap


def jaccard_similarity(source, target, stem=False, clean=True):
    """Compute the jaccard similarity between two texts."""
    if stem:
        source = clean_stem(source)
        target = clean_stem(target)
    elif clean:
        source = clean_text(source)
        target = clean_text(target)
    if len(source) == 0 or len(target) == 0:
        return 0.
    source_set = set(source)
    target_set = set(target)
    try:
        return set_overlap(source_set, target_set.union(source_set))
    except ZeroDivisionError:
        return 0.


def copy_rate(source, target, stem=False, clean=True):
    """
    Compute copy rate

    :param source:
    :param target:
    :param stem: whether to perform stemming using nltk
    :return:
    """
    if stem:
        source_arr = clean_stem(source)
        target_arr = clean_stem(target)
    elif clean:
        source_arr = clean_text(source)
        target_arr = clean_text(target)
    else:
        source_arr = source
        target_arr = target
    source_set = set(source_arr)
    target_set = set(target_arr)
    if len(source_set) == 0 or len(target_set) == 0:
        return 0.
    return set_overlap(source_set, target_set)


def repeat_rate(sents):
    """
    Compute the repeat rate of a text

    :param sents:
    :return:
    """
    if len(sents) == 1:
        return 0.
    else:
        repeat_rates = []
        for i, sent in enumerate(sents):
            rest = " ".join([sents[j] for j, s in enumerate(sents) if j != i])
            repeat_rates.append(copy_rate(rest, sent))
    return np.mean(repeat_rates)


def relaxed_wmd(doc_1, doc_2, w2v, distance_matrix=None, normed=False,
                doc_1_vectors=None, doc_2_vectors=None, use_distance=False):
    """
    Compute the Relaxed Word Mover's Distance score between doc_1 and doc_2.
    See http://proceedings.mlr.press/v37/kusnerb15.pdf for more info.

    :param doc_1:
    :param doc_2:
    :param w2v: the word2vec model to use
    :param distance_matrix:
    :param doc_1_vectors:
    :param doc_2_vectors:
    :param use_distance: if true, will use the euclidean distance; otherwise the cosine similarity
    :return:
    """
    # Compute the distance/similarity matrix between the documents
    if distance_matrix is None:
        if doc_1_vectors is None:
            doc_1_vectors = np.array(get_word_vector_list(doc_1, w2v), dtype=float)
        if doc_2_vectors is None:
            doc_2_vectors = np.array(get_word_vector_list(doc_2, w2v), dtype=float)
        if len(doc_1_vectors) == 0 or len(doc_2_vectors) == 0:
            if use_distance:
                return np.inf
            else:
                return 0.
        if use_distance:
            distance_matrix = cdist(doc_1_vectors, doc_2_vectors, "euclidean")
        else:
            distance_matrix = cosine_similarity(doc_1_vectors, doc_2_vectors)
            # Round and clip similarity matrix to make sure it is between 0 and 1
            distance_matrix = np.round(distance_matrix, 5)
            distance_matrix = np.nan_to_num(distance_matrix)
            distance_matrix[distance_matrix == np.inf] = 0.
            distance_matrix = distance_matrix.clip(min=0.)

    if use_distance:
        score = np.mean(np.min(distance_matrix, 1))
        return 1. / (1. + score)
    else:
        return np.mean(np.max(distance_matrix, 1))


def relaxed_wmd_combined(doc_1, doc_2, w2v, normed=False, distance=False,
                         combination="mean", return_parts=False):
    """


    :param doc_1:
    :param doc_2:
    :param w2v:
    :param normed:
    :param distance:
    :param combination:
    :return:
    """
    doc_1_vectors = np.array(get_word_vector_list(doc_1, w2v), dtype=float)
    doc_2_vectors = np.array(get_word_vector_list(doc_2, w2v), dtype=float)

    if len(doc_1_vectors) == 0 or len(doc_2_vectors) == 0:
        if distance:
            return np.inf if not return_parts else np.inf, np.inf, np.inf
        else:
            return 0. if not return_parts else 0., 0., 0.
    if distance:
        D = cdist(doc_1_vectors, doc_2_vectors, "euclidean")
    else:
        D = cosine_similarity(doc_1_vectors, doc_2_vectors)
        D = np.round(D, 5)
        D = np.nan_to_num(D)
        D[D == np.inf] = 0.
        D = D.clip(min=0.)

    # Compute left and right RWMD
    l1 = relaxed_wmd(doc_1, doc_2, w2v, distance_matrix=D, normed=normed, use_distance=distance)
    l2 = relaxed_wmd(doc_2, doc_1, w2v, distance_matrix=D.T, normed=normed, use_distance=distance)

    # Combine the two RWMD approximations
    if combination == "mean":
        combined = np.mean([l1, l2])
    elif combination == "min":
        combined = np.min([l1, l2])
    elif combination == "max":
        combined = np.max([l1, l2])

    if distance:
        combined = 1. / (1. + combined)

    if return_parts:
        return combined, l1, l2
    else:
        return combined


def extract_features(src, tgt, w2v_model):
    src_clean = clean_text(src)
    tgt_clean = clean_text(tgt)
    rwmd, rwmd_left, rwmd_right = relaxed_wmd_combined(src_clean, tgt_clean, w2v_model, return_parts=True)
    vocab_score_src = vocab_score(src)
    vocab_score_tgt = vocab_score(tgt)
    fk_score_src = fk_score(src)
    fk_score_tgt = fk_score(tgt)
    return [rwmd, rwmd_left, rwmd_right,
            relaxed_wmd(src_clean, tgt_clean, w2v_model),
            relaxed_wmd(tgt_clean, src_clean, w2v_model),
            vocab_score_src, vocab_score_tgt, vocab_score_src - vocab_score_tgt,
            fk_score_src, fk_score_tgt, fk_score_src - fk_score_tgt,
            copy_rate(src_clean, tgt_clean, clean=False),
            len(src_clean),
            len(tgt_clean)]


def prepare_emb_features(src_emb, tgt_emb):
    return np.concatenate((src_emb, tgt_emb, src_emb - tgt_emb, src_emb * tgt_emb))


def extract_sent2vec_features(src, tgt, model):
    src_clean = clean_text(src)
    tgt_clean = clean_text(tgt)
    src_emb = model.embed_sentence(" ".join(src_clean))
    tgt_emb = model.embed_sentence(" ".join(tgt_clean))
    return prepare_emb_features(src_emb, tgt_emb)


def classifier_prob(doc_1, doc_2, w2v, classifier, doc_1_features=None, doc_2_features=None):
    if doc_1_features is None or doc_2_features is None:
        features = extract_sent2vec_features(doc_1, doc_2, w2v)
    else:
        features = prepare_emb_features(doc_1_features, doc_2_features)
    # features = extract_features(doc_1, doc_2, w2v)
    features = np.expand_dims(features, axis=0)
    prob = classifier.predict_proba(features)[0][1]
    return prob


def classifier_prob_batch(lines, w2v, classifier):
    features = np.array([extract_sent2vec_features(doc_1, doc_2, w2v)
                         for doc_1, doc_2 in lines])
    prob = classifier.predict_proba(features).flatten()
    return prob


def classifier_prob_batch_features(features, classifier):
    return classifier.predict_proba(features).flatten()