"""Code for sentence tokenization of a text file"""

import argparse
import itertools
import logging
import pickle
import time
from multiprocessing import Process, Manager, cpu_count

import nltk
import numpy as np
from file_tools import wc
from tqdm import *

parser = argparse.ArgumentParser()
parser.add_argument('-src_file', help='The source file', required=True)
parser.add_argument('-tab_split', default=False, type=bool,
                    help='Sentence separator token to use in text. Default: use nltk.sent_tokenize(). '
                         'You can optionally specify a token to use for the splitting, e.g. space or tab "\\t"')
parser.add_argument('-label', default=None, type=int, help='Label to add to sentences (for classification)')
args = parser.parse_args()
total_documents = wc(args.src_file)


def split(in_queue, out_list):
    global total_documents, print_every
    while True:
        document_id, document = in_queue.get()
        if document is None:  # exit signal
            return
        if not args.tab_split:
            sentences = nltk.sent_tokenize(document)
        else:
            sentences = document.split("\t")
        result = (document_id, sentences)
        out_list.append(result)


if __name__ == '__main__':
    sent_file = open("{}.sent".format(args.src_file), "w")

    sent_doc_id = {}
    doc_sent_id = {}
    start_total = time.time()

    sentence_id = 0

    logging.warning("Starting split of {} lines...".format(total_documents))

    # Set up simple multiprocessing
    num_workers = int(cpu_count() / 2.)
    manager = Manager()
    results = manager.list()
    work = manager.Queue(num_workers)

    pool = []
    for i in range(num_workers):
        p = Process(target=split, args=(work, results))
        p.start()
        pool.append(p)

    with tqdm(desc="Sentence split", total=total_documents) as pbar:
        with open(args.src_file) as src_file:
            iters = itertools.chain(src_file, (None,) * num_workers)
            for doc_sent_ids in enumerate(iters):
                work.put(doc_sent_ids)
                pbar.update()

    for p in pool:
        p.join()

    for document_id, sentences in results:
        sent_ids = list(range(sentence_id, sentence_id + len(sentences)))
        sentence_id += len(sentences)
        doc_sent_id[document_id] = sent_ids
        for i, sent_id in enumerate(sent_ids):
            if args.label is not None:
                sent_file.write("{} __label__{}\n".format(sentences[i].strip(), args.label))
            else:
                sent_file.write("{}\n".format(sentences[i].strip()))
            sent_doc_id[sent_id] = document_id

    sent_file.close()
    pickle.dump(sent_doc_id, open('{}.sent.s2d.pkl'.format(
        args.src_file), "wb"))
    pickle.dump(doc_sent_id, open('{}.sent.d2s.pkl'.format(
        args.src_file), "wb"))

    end_total = time.time() - start_total
    logging.warning(
        "Finished processing {} documents, extracting {} sentences. Took {}s ({}s per doc on average)".format(
            total_documents, sent_id, np.round(end_total, 2), np.round(end_total / float(total_documents), 4)))
