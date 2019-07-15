"""
Main class for running the aligner.
"""

import argparse
import logging
import time

import coloredlogs
from annoy import AnnoyIndex
from gensim.models.keyedvectors import KeyedVectors

from align.global_align import GlobalAligner
from align.hierarchical_align import HierarchicalAligner
from tools.embeddings import EMBED_MODES
from tools.text_similarity import WORD_METRICS, METRICS_IMPLEMENTED

coloredlogs.install()

SUPPORTED_ALIGN_LEVELS = ['global', 'hierarchical']

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('-src', help='The source file', required=True)
parser.add_argument('-tgt', help='The target file', required=True)
parser.add_argument('-emb', help='The embedding type. Options: {}'.format(EMBED_MODES), default="sent2vec")
parser.add_argument('-level', default="hierarchical", type=str, help='The level of alignment. Options: {}'.format(
    SUPPORTED_ALIGN_LEVELS))
parser.add_argument('-refine', default=None, help='Optional, a similarity metric that will be used to refine the '
                                                  'embedding-based similarity scores. '
                                                  'Currently implemented metrics: {}'.format(METRICS_IMPLEMENTED))
parser.add_argument('-refine_all', default=False, help='If true, will refine all possible pairs. '
                                                       'If false, will only refine nearest_lim neighbours.')
parser.add_argument('-annoy_metric', default="angular", type=str, help='Annoy similarity metric that was used to build '
                                                                       'the index. See the Annoy docs for more info.')
parser.add_argument('-k_best', default=1, help='How many pairs to choose per source/target. '
                                               'More than 1 will extract multiple nearest neighbours.', type=int)
parser.add_argument('-nearest_lim', default=50, help='Number of nearest neighbours to search for with Annoy.', type=int)
parser.add_argument('-batch_size', default=2000, help='Batch the computation of the alignments. '
                                                      'Set to smaller values for limited amounts of RAM.', type=int)
parser.add_argument('-w2v', default='news', help='The Word2Vec model that will be used. Only relevant if youre using a'
                                                 ' Word2Vec-based embedding method.')
parser.add_argument('-vec_size', default=600, help='The embedding dimensionality of your index/embedding model.'
                                                   'A wrong dimensionality might result in weird results.',
                    type=int)
parser.add_argument('-lower_th', default=0.5, type=float,
                    help='The lower similarity threshold for accepting a pair.')
parser.add_argument('-upper_th', default=1.1, type=float,
                    help='The upper similarity threshold for accepting a pair.')
parser.add_argument('-lazy_target', default=False, help="If True, will not load target index in memory but will "
                                                        "compute the source embeddings on the fly.")
parser.add_argument('-lazy_source', default=False, help="If True, will not load source index in memory but will "
                                                        "compute the source embeddings on the fly.")
parser.add_argument('-negative', default=False, help="If True, will choose most dissimilar sentences. "
                                                     "Useful when the goal is to collect negative examples.")
args = parser.parse_args()

if args.refine == "None":
    args.refine = None
if args.batch_size == -1:
    args.batch_size = None

# Update according to your set-up
w2v_models = {
    "news": "~/data/raw/wiki/GoogleNews-vectors-negative300.bin",
    "wiki": "~/data/raw/word2vec/wiki.en/wiki.en.bin"
}

if __name__ == '__main__':
    if args.refine in WORD_METRICS: # Load the Word2Vec model.
        w2v_path = w2v_models[args.w2v]
        if w2v_path.endswith(".bin"):
            w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        else:
            w2v = KeyedVectors.load(w2v_path, mmap='r')
        logging.warning("Loaded Word2Vec model {} with embedding dim {} and vocab size {}".format(
            w2v_path, w2v.vector_size, len(w2v.syn0)))
        if args.vec_size is None:
            args.vec_size = w2v.vector_size
    else:
        w2v = None

    args_v = vars(args).copy()

    # Load the Annoy indices.
    if args.lazy_source:
        src_index = None
        logging.warning("Lazy mode: no src index")
    else:
        src_index = AnnoyIndex(args.vec_size, metric=args.annoy_metric)
        src_index_fpath = "{}.{}.ann".format(args.src, args.emb)
        src_index.load(src_index_fpath)
        logging.info("Loaded {}".format(src_index_fpath))

    if args.lazy_target:
        tgt_index = None
        logging.warning("Lazy mode: no tgt index")
    else:
        tgt_index = AnnoyIndex(args.vec_size, metric=args.annoy_metric)
        tgt_index_fpath = "{}.{}.ann".format(args.tgt, args.emb)
        tgt_index.load(tgt_index_fpath)
        logging.info("Loaded {}".format(tgt_index_fpath))

    args.src_index = src_index
    args.tgt_index = tgt_index
    args.w2v = w2v

    # Some checks of the size
    try:
        if src_index is not None:
            assert src_index.get_n_items() > 0
        if tgt_index is not None:
            assert tgt_index.get_n_items() > 0
    except AssertionError:
        logging.critical("Error loading Annoy indices. SRC size: {} TGT size: {}".format(
                           src_index.get_n_items(), tgt_index.get_n_items()))

    logging.warning("Starting alignment of {} source and {} target documents...".format(
        src_index.get_n_items() if src_index is not None else "LAZY", tgt_index.get_n_items() if tgt_index is not None else "LAZY"))
    logging.warning("Will use the {} metric for refinement.".format(args.refine))
    start_time = time.time()
    batch_time = time.time()

    args_dict = vars(args).copy()
    for key in ["emb", "level", "annoy_metric", "vec_size", "lazy_source", "lazy_target"]:
        del args_dict[key]

    aligner = None
    try:
        assert args.level in SUPPORTED_ALIGN_LEVELS
    except AssertionError as e:
        logging.critical("-level needs to be in {}".format(SUPPORTED_ALIGN_LEVELS))
        raise e

    if args.level == 'global':
        logging.info("Starting Global aligner.")
        aligner = GlobalAligner(**args_dict)
    elif args.level == "hierarchical":
        aligner = HierarchicalAligner(**args_dict, global_pairs=None)

    aligner.predict_write()

    end_time = time.time()
    time_diff = end_time - start_time
    logging.warning("Finished! Total time for alignment %d minutes, %f seconds" \
                    % (int(time_diff / 60), time_diff % 60))
