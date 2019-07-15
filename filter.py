"""A script for filtering the extracted pairs post-alignment. """


import argparse
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from tools.text_similarity import copy_rate
import numpy as np
from tqdm import *
from preprocessing.clean_text import clean_text
import fastText

classifiers = {
    "wiki_simple": "/home/nikola/data/raw/alignment/local/wiki-simple/classifier/model.bin",
    "paper_press": "/home/nikola/data/raw/alignment/local/paper-press/classifier-train/model.bin"
}

parser = argparse.ArgumentParser()
parser.add_argument('-src', help='The source file', required=True)
parser.add_argument('-tgt', help='The target file', required=True)
parser.add_argument('-sim', help='The file containing the computed similarities between src/tgt lines.', required=True)
parser.add_argument('-low_sim_th', help='Low threshold for the similarity', default=0.63, type=float)
parser.add_argument('-up_sim_th', help='Up threshold for the similarity', default=1.1, type=float)
parser.add_argument('-low_copy_th', help='Low threshold for copy', default=0.3, type=float)
parser.add_argument('-up_copy_th', help='Up threshold for copy', default=1.1, type=float)
parser.add_argument('-print_n', type=int, help='Print debug lines', default=10)
args = parser.parse_args()

stat = {
    "too_short": 0,
    "sim_out_of_range": 0,
    "copy_out_of_range":0
}


def valid_pair(src, tgt):
    tok_src = clean_text(src, lower=True, remove_digits=False, remove_stop=False)
    tok_tgt = clean_text(tgt, lower=True, remove_digits=False, remove_stop=False)
    return len(tok_src) > 2 and len(tok_tgt) > 2 and len(src.split("et al")) < 3 and len(tgt.split("et al")) < 3


if __name__ == '__main__':
    src_file = open(args.src)
    tgt_file = open(args.tgt)
    sim_file = open(args.sim).readlines()

    if args.filter_classifier:
        logging.warning("Loading {} classifier from {}".format(args.classifier, classifiers[args.classifier]))
        ft_model = fastText.load_model(classifiers[args.classifier])

    if args.split_id is not None:
        src_id_file = open("{}.id".format(args.src)).readlines()
        tgt_id_file = open("{}.id".format(args.tgt)).readlines()
    else:
        src_id_file = [None for i in range(len(sim_file))]
        tgt_id_file = [None for i in range(len(sim_file))]

    printed = 0
    count = 0
    total = 0

    src_filter = open("{}.{}-filter".format(args.src, args.low_sim_th), "w")
    tgt_filter = open("{}.{}-filter".format(args.tgt, args.low_sim_th), "w")
    sim_filter = open("{}.{}-filter".format(args.sim, args.low_sim_th), "w")

    if args.split_id is not None:
        src_filter_test = open("{}.filter.test".format(args.src), "w")
        tgt_filter_test = open("{}.filter.test".format(args.tgt), "w")
        sim_filter_test = open("{}.filter.test".format(args.sim), "w")

    src_history = []
    tgt_history = []

    with tqdm(desc="Filter") as pbar:
        for i, (src_line, tgt_line) in enumerate(zip(src_file, tgt_file)):
            total += 1

            if args.split_id is not None:
                src_line_id = int(src_id_file[i].strip())
                tgt_line_id = int(tgt_id_file[i].strip())

            src_line = src_line.strip()
            tgt_line = tgt_line.strip()

            similarity = float(sim_file[i].strip())
            cr = copy_rate(src_line, tgt_line)
            sim_approve = args.up_copy_th >= similarity > args.low_sim_th
            if not sim_approve:
                stat["sim_out_of_range"] += 1
            copy_approve = args.up_copy_th >= cr > args.low_copy_th
            if not copy_approve:
                stat["copy_out_of_range"] += 1

            if sim_approve and copy_approve and valid_pair(src_line, tgt_line):
                if printed < args.print_n:
                    print(similarity)
                    print(cr)
                    print(src_line)
                    print(tgt_line)
                    print()
                    printed += 1
                count += 1

                if args.split_id is not None:
                    if src_line_id >= args.split_id:
                        src_filter_test.write("{}\n".format(src_line))
                        tgt_filter_test.write("{}\n".format(tgt_line))
                        sim_filter_test.write("{} {}\n".format(similarity, cr))
                    else:
                        src_filter.write("{}\n".format(src_line))
                        tgt_filter.write("{}\n".format(tgt_line))
                        sim_filter.write("{} {}\n".format(similarity, cr))
                else:
                    src_filter.write("{}\n".format(src_line))
                    tgt_filter.write("{}\n".format(tgt_line))
                    sim_filter.write("{} {}\n".format(similarity, cr))

                if len(src_history) > 50:
                    del src_history[0]
                    del tgt_history[0]
                src_history.append(src_line)
                tgt_history.append(tgt_line)
                src_filter.flush()
                tgt_filter.flush()
                sim_filter.flush()
            pbar.update()

    logging.info("Selected {} pairs ({}%)".format(count, np.round(100. * count / float(total), 2)))
    logging.info("Pairs excluded per category: {}".format(stat))

    src_file.close()
    tgt_file.close()
