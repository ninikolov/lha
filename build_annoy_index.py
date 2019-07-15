"""Script for building an Annoy index of text embeddings."""

import argparse
import itertools
import logging
import sys
import time
from multiprocessing import Process, Manager, cpu_count

import coloredlogs
import numpy as np
from annoy import AnnoyIndex
from gensim.models.keyedvectors import KeyedVectors
from tqdm import *

from preprocessing.clean_text import clean_text
from preprocessing.file_tools import wc
from tools.embeddings import EMBED_MODES
from tools.embeddings import embed_text, get_word_vector_dic

coloredlogs.install()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Update this with your models
models = {
    "news": "/home/nikola/data/raw/wiki/GoogleNews-vectors-negative300.bin",
    "wiki": "/home/nikola/data/raw/word2vec/wiki.en/wiki.en.vec",
    "wiki_unigrams": "/home/nikola/data/raw/wiki/wiki_unigrams.bin"
}

parser = argparse.ArgumentParser()
parser.add_argument('-src_file', help='The text file to process and embed.', required=True)
parser.add_argument('-model', default="wiki_unigrams", help='The embedding model that will be used. '
                                                            'Available models: {}'.format(list(models.keys())))
parser.add_argument('-vec_size', default=600, type=int, help='Dimensionality of the embedding to produce.')
parser.add_argument('-precision', default=32, type=int, help='Floating point precision of embeddings')
parser.add_argument('-metric', default="angular", type=str, help='Annoy distance metric to use.')
parser.add_argument('-emb', default="sent2vec", type=str, help='Embedding approach. Options: {}'.format(EMBED_MODES))
parser.add_argument('-n_trees', default=50, type=int, help='Number of trees for Annoy (read Annoy docs)')
parser.add_argument('-n_threads', default=int(cpu_count() * 0.5), type=int, help='Number of threads to use.')
parser.add_argument('-chunk_size', default=100000, type=int, help='Process this many lines at once.'
                                                                  'You may have to adjust this depending on how much'
                                                                  'RAM you have.')
args = parser.parse_args()


def text_to_embedding(in_queue, out_list):
    """Compute a single embedding"""
    while True:
        input_id, input_text = in_queue.get()
        if input_text is None:  # exit signal
            return
        if args.emb == "sent2vec":
            clean_input = clean_text(input_text)
            input_emb = embed_text(
                clean_input, embedding_model, None,
                args.vec_size, mode=args.emb, clean=False
            ).astype(args.precision)
            input_emb = input_emb[0]
        elif args.emb == "bert":
            return embedding_model.encode([input_text])
        else:
            clean_input = clean_text(input_text, lower=True)
            embeddings = get_word_vector_dic(clean_input, embedding_model)
            input_emb = embed_text(
                clean_input, embedding_model, embeddings, args.vec_size, mode=args.emb, clean=False).astype(args.precision)

        out_list.append((input_id, input_emb))


def parallel_embed(chunk, chunk_id):
    """Compute embeddings for a chunk using multithreading."""
    # Set up simple multiprocessing
    manager = Manager()
    results = manager.list()
    work = manager.Queue(args.n_threads)

    pool = []
    for i in range(args.n_threads):
        p = Process(target=text_to_embedding, args=(work, results))
        p.start()
        pool.append(p)

    chunk_iter = itertools.chain(
        iter(chunk), enumerate((None,) * args.n_threads))

    with tqdm(total=len(chunk), desc="Process chunk {}".format(chunk_id)) as pbar2:
        for d_id, doc in chunk_iter:
            work.put((d_id, doc))
            if doc is not None:
                pbar2.update()

    for p in pool:
        p.join()

    assert len(results) == len(chunk)
    return results


assert args.precision in [8, 16, 32, 64]
args.precision = "float{}".format(args.precision)

logging.warning("Checking input file {}...".format(args.src_file))
total_documents = wc(args.src_file)
print_every = int(total_documents * 0.001) + 1

logging.warning("Starting to build Annoy index for {} documents...".format(total_documents))
logging.warning("Will use the {} distance metric..".format(args.metric))

model_path = models[args.model]

if args.emb == 'avg':
    model_path = models[args.model]
    if model_path.endswith(".bin"):
        embedding_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    elif model_path.endswith(".vec"):
        embedding_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    else:
        embedding_model = KeyedVectors.load(model_path, mmap='r')

    if args.vec_size is None:
        args.vec_size = embedding_model.vector_size

    logging.warning("Loaded Word2Vec model {} with embedding dim {} and vocab size {}".format(
        args.model, embedding_model.vector_size, len(embedding_model.wv.vocab)))
else:
    embedding_model = None

if args.emb == "avg":
    logging.warning("Will use AVG embedding approach")
elif args.emb == "sent2vec":
    import sent2vec
    embedding_model = sent2vec.Sent2vecModel()
    embedding_model.load_model(models["wiki_unigrams"])
    logging.info("Loaded sent2vec model")
elif args.emb == "bert":
    # Requires the bert client https://github.com/hanxiao/bert-as-service
    from bert_serving.client import BertClient
    embedding_model = BertClient(check_length=False)
else:
    logging.critical("Wrong embedding choice.")
    sys.exit(0)


if __name__ == '__main__':
    index = AnnoyIndex(args.vec_size, metric=args.metric)
    start_total = time.time()
    logging.warning("Starting to build index for {} using arguments: {}".format(args.src_file, vars(args)))

    with open(args.src_file) as src_file:
        file_iter = iter(src_file)
        chunk = []
        chunk_count = 0
        print()
        with tqdm(total=total_documents, desc="Total progress") as pbar:
            for document_id, document in enumerate(file_iter):
                # If we've reached the chunk limit, compute the embeddings,
                # otherwise store text in a list.
                if len(chunk) > args.chunk_size:
                    # Process chunk
                    if args.emb == "bert":
                        embs = embedding_model.encode([p[1] for p in chunk])
                        assert len(embs) == len(chunk)
                        results = zip([p[0] for p in chunk], embs)
                    else:
                        results = parallel_embed(chunk, chunk_count)

                    for d_id, document_emb in results:
                        index.add_item(d_id, document_emb)
                        pbar.update()

                    chunk = []
                    chunk_count += 1
                else:
                    chunk.append((document_id, document))

            # Check if last chunk has been processed.
            if len(chunk) > 0:
                if args.emb == "bert":
                    embs = embedding_model.encode([p[1] for p in chunk])
                    assert len(embs) == len(chunk)
                    results = zip([p[0] for p in chunk], embs)
                else:
                    results = parallel_embed(chunk, chunk_count)
                for d_id, document_emb in results:
                    index.add_item(d_id, document_emb)
                    pbar.update()

    end_total = time.time() - start_total
    logging.warning("Finished processing {} documents. Took {}s ({}s per doc on average)".format(
        total_documents, np.round(end_total, 2), np.round(end_total / float(total_documents), 4)))

    # Try to free up some memory...
    del embedding_model

    logging.warning("Starting to build Annoy index.")
    index.build(args.n_trees)

    fname = '{}.{}.ann'.format(args.src_file, args.emb)
    logging.warning("Saving Annoy index with {} items at {}.".format(index.get_n_items(), fname))
    index.save(fname)
    logging.warning("Finished!")
