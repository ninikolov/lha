import itertools
import os
from multiprocessing import Process, Manager, cpu_count

import matplotlib as mpl
import scipy as sp
from distance import nlevenshtein
from langdetect import detect
from linecache_light import LineCache
from tqdm import *

from preprocessing.file_tools import wc
from tools.text_similarity import *
from tools.embeddings import embed_text

mpl.use('Agg')
import coloredlogs

from gensim.summarization.bm25 import get_bm25_weights, BM25

coloredlogs.install()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
logging.getLogger("gensim").setLevel(logging.WARNING)

class AlignerBase:
    """Base class of the aligner."""

    def __init__(self, src, tgt, src_index, tgt_index, refine, lower_th, \
                 refine_all=False, upper_th=1., src_local_to_global=None, tgt_local_to_global=None, \
                 src_global_to_local=None, tgt_global_to_local=None, w2v=None,
                 tfidf_model=None, nearest_lim=2, max_target=True, batch_size=None, k_best=1,
                 load_in_memory=False, debug=False, negative=False,
                 merge=False, filter=False):
        self.src = src
        self.tgt = tgt

        self.w2v_model = w2v
        self.tfidf_model = tfidf_model
        self.nearest_lim = nearest_lim
        self.src_index = src_index
        self.tgt_index = tgt_index
        self.upper_th = upper_th
        self.max_target = max_target
        self.out_k_best = k_best
        self.batch_size = batch_size

        self.refine_all = refine_all

        # If true, will load all the text in memory. Otherwise, will use a cache.
        if load_in_memory:
            self.src_lines = open(self.src).readlines()
            self.tgt_lines = open(self.tgt).readlines()
        else:
            self.src_cache = LineCache(self.src, cache_suffix='.cache')
            self.tgt_cache = LineCache(self.tgt, cache_suffix='.cache')

        logging.info("Counting file lines...")
        self.src_total = wc(self.src)
        self.tgt_total = wc(self.tgt)
        # Some checks on the item counts.
        if self.src_index is not None:
            if self.src_total != self.src_index.get_n_items():
                logging.warning("Mismatch between src index size and file size: {} vs. {}".format(
                    self.src_total, self.src_index.get_n_items()))
        if self.tgt_index is not None:
            if self.tgt_total != self.tgt_index.get_n_items():
                logging.warning("Mismatch between src index size and file size: {} vs. {}".format(
                    self.tgt_total, self.tgt_index.get_n_items()))

        self.load_in_memory = load_in_memory
        self.lower_th = lower_th

        self.debug = debug

        assert refine in METRICS_IMPLEMENTED
        self.ref_similarity_metric = refine

        # For local alignment: the dictionaries that link a sentence id to a document id.
        self.src_local_to_global = src_local_to_global
        self.tgt_local_to_global = tgt_local_to_global
        self.src_global_to_local = src_global_to_local
        self.tgt_global_to_local = tgt_global_to_local

        self.negative = negative
        self.merge = merge
        self.merge_count = 0

        self.min_len = 6
        self.max_len = 100
        self.len_ratio_max = 100
        self.check_overlap = False

        self.search_k_multiply = 2

        # load sent2vec
        import sent2vec
        self.w2v = sent2vec.Sent2vecModel()
        self.w2v.load_model('/home/nikola/data/raw/wiki/wiki_unigrams.bin')
        logging.info("Loaded sent2vec model")

    def get_src_line(self, i):
        if self.load_in_memory:
            return self.src_lines[i]
        else:
            return self.src_cache[i].decode('latin1')

    def get_tgt_line(self, i):
        if self.load_in_memory:
            return self.tgt_lines[i]
        else:
            return self.tgt_cache[i].decode('latin1')

    def get_sent2vec_emb(self, str):
        str = clean_text(str, lower=True)
        embedding = embed_text(str, self.w2v, None, 600, mode="sent2vec", clean=False)
        return embedding[0]

    def get_tgt_emb(self, idx):
        if self.tgt_index is None:
            txt = self.get_tgt_line(idx)
            return self.get_sent2vec_emb(txt)
        else:
            try:
                return self.tgt_index.get_item_vector(idx)
            except IndexError as e:
                logging.critical("No index item with ID: {}. INDEX SIZE: {}".format(idx, self.get_tgt_count()))
                return np.random.rand(600)

    def get_src_emb(self, idx):
        if self.src_index is None:
            txt = self.get_src_line(idx)
            return self.get_sent2vec_emb(txt)
        else:
            try:
                return self.src_index.get_item_vector(idx)
            except IndexError as e:
                logging.critical("No index item with ID: {}. INDEX SIZE: {}".format(idx, self.get_src_count()))
                return np.random.rand(600)

    def get_src_count(self):
        return self.src_total

    def get_tgt_count(self):
        return self.tgt_total

    def cosine_sim(self, src_id, tgt_id):
        """Cosine similarity between two individual items."""
        return 1. / (1. + sp.spatial.distance.cosine(self.get_src_emb(src_id), self.get_tgt_emb(tgt_id)))

    def cosine_sim_matrix(self, src_indices, tgt_indices, src_emb=None, tgt_emb=None, return_embs=False):
        """Compute whole similarity matrix, fetching src and target indices."""
        # print(src_indices, tgt_indices)
        try:
            if src_emb is None or tgt_emb is None:
                src_emb = np.array([self.get_src_emb(idx) for idx in src_indices], dtype=float)
                tgt_emb = np.array([self.get_tgt_emb(idx) for idx in tgt_indices], dtype=float)
            src_emb = np.nan_to_num(src_emb)
            tgt_emb = np.nan_to_num(tgt_emb)
            similarity_matrix = cosine_similarity(src_emb, tgt_emb)
            # Remove NAN, Make sure simiarlity is between 0 and 1.
            similarity_matrix = np.nan_to_num(similarity_matrix)
            similarity_matrix = np.round(similarity_matrix, 5)
            similarity_matrix = similarity_matrix.clip(min=0., max=1.)
            if return_embs:
                src_embs = {idx: emb for idx, emb in zip(src_indices, src_emb)}
                tgt_embs = {idx: emb for idx, emb in zip(tgt_indices, tgt_emb)}
                return similarity_matrix, src_embs, tgt_embs
            else:
                return similarity_matrix
        except Exception as e:
            print(src_emb.shape)
            print(tgt_emb.shape)
            raise e

    def bm25_sim_matrix(self, src_indices, tgt_indices, index_source=True, monitor_progress=True):
        base_matrix = np.zeros((len(src_indices), len(tgt_indices)))
        if monitor_progress:
            pbar = tqdm(total=len(src_indices) if index_source is True else len(tgt_indices), desc="BM25 matrix")

        print(base_matrix.shape)

        src_lines = [clean_text(self.get_src_line(s)) for s in src_indices]
        tgt_lines = [clean_text(self.get_tgt_line(t)) for t in tgt_indices]
        if index_source:
            bm25 = BM25(src_lines)
        else:
            bm25 = BM25(tgt_lines)
        average_idf = sum(float(val) for val in bm25.idf.values()) / len(bm25.idf)

        for src_id, src_line in enumerate(src_lines):
            for tgt_id, tgt_line in enumerate(tgt_lines):
                if index_source:
                    bm25_res = bm25.get_score(tgt_line, src_id, average_idf)
                else:
                    bm25_res = bm25.get_score(src_line, tgt_id, average_idf)
                base_matrix[src_id, tgt_id] = bm25_res
            if monitor_progress:
                pbar.update()
        return base_matrix / np.max(base_matrix)

    def pair_similarity(self, src_id, tgt_id, src=None, tgt=None):
        """
        Compute a similarity between two texts: source and target.

        :param src: the src text
        :param tgt: the tgt text
        :param src_id: the id of the src text in the index
        :param tgt_id: the id of the tgt text in the index
        :return:
        """

        if self.ref_similarity_metric in WORD_METRICS:
            if src is None:
                src_line = self.get_src_line(src_id)
                src = clean_text(src_line)
            if tgt is None:
                tgt_line = self.get_tgt_line(tgt_id)
                tgt = clean_text(tgt_line)

        if self.ref_similarity_metric == "cosine":
            return self.cosine_sim(src_id, tgt_id)
        elif self.ref_similarity_metric == "wmd":
            return 1. / (1. + self.w2v_model.wmdistance(src, tgt))
        elif self.ref_similarity_metric == "rwmd":
            return relaxed_wmd_combined(src, tgt, self.w2v_model)
        elif self.ref_similarity_metric == "rwmd_left":
            return relaxed_wmd(src, tgt, self.w2v_model)
        elif self.ref_similarity_metric == "rwmd_right":
            return relaxed_wmd(tgt, src, self.w2v_model)
        elif self.ref_similarity_metric == "jaccard":
            return jaccard_similarity(src_line, tgt_line)
        elif self.ref_similarity_metric == "jaccard_stem":
            return jaccard_similarity(src_line, tgt_line, stem=True)
        elif self.ref_similarity_metric == "copy_rate":
            return copy_rate(src, tgt)
        elif self.ref_similarity_metric == "nlev":
            return 1. - nlevenshtein(src_line, tgt_line)

    def text_similarity(self, txt, src_id, refine_metric="rwmd"):
        tgt = clean_text(self.get_src_line(src_id))
        if refine_metric == "wmd":
            return 1. / (1. + self.w2v_model.wmdistance(txt, tgt))
        elif refine_metric == "rwmd":
            return relaxed_wmd_combined(txt, tgt, self.w2v_model)
        elif refine_metric == "rwmd_norm":
            return 1. - relaxed_wmd_combined(txt, tgt, self.w2v_model, normed=True)

    def parallel_pair_similarity(self, in_queue, out_list):
        while True:
            pair_id, new_pair = in_queue.get()
            if new_pair is None:
                return
            src_id, tgt_id = new_pair
            similarity = self.pair_similarity(src_id, tgt_id)
            result = (src_id, tgt_id, similarity)
            out_list.append(result)

    def parallel_refine(self, pairs_to_compute, progress_bar=True, num_workers=int(cpu_count() / 2)):
        """Compute refine function in parallel"""

        if progress_bar:
            pbar = tqdm(total=len(pairs_to_compute), desc="Refine {}".format(self.ref_similarity_metric))

        manager = Manager()
        results = manager.list()
        # num_workers = 1
        work = manager.Queue(num_workers)
        self.refine_count = len(pairs_to_compute)

        # start for workers
        pool = []
        for i in range(num_workers):
            p = Process(target=self.parallel_pair_similarity, args=(work, results))
            p.start()
            pool.append(p)

        iters = itertools.chain(pairs_to_compute, (None,) * num_workers)

        for id_pair in enumerate(iters):
            work.put(id_pair)
            if progress_bar:
                pbar.update()

        for p in pool:
            p.join()

        return results

    def most_similar_src(self, tgt_item, tgt_emb=None):
        try:
            if tgt_emb is None:
                tgt_emb = self.get_tgt_emb(tgt_item)
            return self.src_index.get_nns_by_vector(
                tgt_emb, n=self.nearest_lim,
                search_k=self.nearest_lim * self.search_k_multiply, include_distances=True)
        except IndexError as e:
            logging.error("IndexError for tgt item {}".format(tgt_item))
            return [], []

    def most_similar_tgt(self, src_item):
        src_emb = self.src_index.get_item_vector(src_item)
        return self.tgt_index.get_nns_by_vector(
            src_emb, n=self.nearest_lim, search_k=self.nearest_lim * self.search_k_multiply, include_distances=True)

    def most_similar_src_emb(self, embedding):
        return self.src_index.get_nns_by_vector(
            embedding, n=self.nearest_lim, search_k=self.nearest_lim * self.search_k_multiply, include_distances=True)

    def most_similar_tgt_emb(self, embedding):
        return self.tgt_index.get_nns_by_vector(
            embedding, n=self.nearest_lim, search_k=self.nearest_lim * self.search_k_multiply, include_distances=True)

    def predict(self):
        pass

    def predict_batch(self, silent=False):
        """Compute prediction in batch mode, e.g. for big datasets."""
        pass

    def predict_write(self):
        self.predict_batch()

    def similarity_to_prediction(self, sim_out, lower_th=None, upper_th=None):
        prediction = []
        if lower_th is None:
            lower_th = self.lower_th
        if upper_th is None:
            upper_th = self.upper_th

        for tgt_id in sim_out.keys():
            nearest_dict = sim_out[tgt_id]
            nearest_list = list(nearest_dict.keys())
            similarities = [nearest_dict[idx] for idx in nearest_list]
            if len(similarities) == 0:
                logging.warning("Empty similarity vector for {}".format(tgt_id))
                return prediction
            selected = 0
            # If this option is enabled, will choose dissimilar pairs.
            # Useful when searching for negative examples.
            try:
                if self.negative:
                    sorted_similarities = np.argsort(similarities)
                else:
                    sorted_similarities = np.argsort(similarities)[::-1]
            except Exception as e:
                print(e)
            for pred in sorted_similarities:

                if self.negative:  # choose negative examples
                    if selected < self.out_k_best:
                        prediction.append((nearest_list[pred], tgt_id))
                        selected += 1
                    else:
                        break
                else:  # find normal examples
                    try:
                        if upper_th >= similarities[pred] > lower_th:
                            if selected < self.out_k_best:
                                current_prediction = (nearest_list[pred], tgt_id)
                                prediction.append(current_prediction)
                                selected += 1
                            else:
                                break
                    except TypeError as er:
                        pass
        return prediction

    def get_nn(self, in_queue, out_list):
        while True:
            iter_id, new_input = in_queue.get()
            if new_input is None:  # exit signal
                return
            tgt_id, tgt_emb = new_input
            nearest_neighbours, similarities = self.most_similar_src(tgt_id, tgt_emb)
            similarities = 1. / (1. + np.array(similarities))
            nearest_dict = dict(zip(nearest_neighbours, similarities))
            out_list.append((tgt_id, nearest_dict))

    def global_similarity_struct_parallel(self, tgt_range=None, progress_monitor=False, skip_refine=False):
        """
        Find the nearest neighbours of each target item in the index. Return them in a dictionary structure, along
        with their similarity. Parallel implementation.

        :param tgt_range: if specified, compute the structure only for this range of target items.
        :param progress_monitor: if True, show a progress bar to the user
        :return: a dictionary where the keys are the target ids, and the values are the source ids and their
        similarities to the target.
        """
        src_count, tgt_count = self.get_src_count(), self.get_tgt_count()
        if tgt_range is None:
            tgt_range = (0, tgt_count - 1)

        sim_struct = {}
        # List of pairs for which to recompute the similarity
        refine_pairs = []
        if progress_monitor:
            pbar = tqdm(desc="Compute similarity struct", total=tgt_range[1] - tgt_range[0])

        # Setup basic multiprocessing
        num_workers = int(cpu_count() / 2.)
        manager = Manager()
        results = manager.list()
        work = manager.Queue(num_workers)
        pool = []
        for i in range(num_workers):
            p = Process(target=self.get_nn, args=(work, results))
            p.start()
            pool.append(p)

        # Prepare embeddings and tgt ids in advance
        tgt_range_iter = range(tgt_range[0], tgt_range[-1] + 1)
        tgt_embs = [self.get_tgt_emb(tgt_id) for tgt_id in tgt_range_iter]
        tgt_final_iter = zip(tgt_range_iter, tgt_embs)
        target_iterator = itertools.chain(tgt_final_iter, (None,) * num_workers)

        for ids in enumerate(target_iterator):
            work.put(ids)
            if progress_monitor:
                pbar.update()

        for p in pool:
            p.join()

        ## Apply refinement optionally
        # Add top k to the list of pairs to be refined
        for tgt_id, nearest_dict in results:
            if self.ref_similarity_metric is not None:
                for src_id in nearest_dict.keys():
                    refine_pairs.append((src_id, tgt_id))
            sim_struct[tgt_id] = nearest_dict

        if self.ref_similarity_metric is not None and not skip_refine:
            # Refine the similarity scores, update sim_struct
            refine_results = self.parallel_refine(refine_pairs)
            for src_id, tgt_id, similarity in refine_results:
                sim_struct[tgt_id][src_id] = similarity

        return sim_struct

    def final_check(self, src, tgt):
        """
        Final check for acceptance of valid pairs.

        :param src: the source text
        :param tgt: the target text
        :param min_len: minimum word length of source/target
        :param len_ratio_max: maximum ratio between the token lengths of src/tgt
        :return:
        """
        return True

    def write_output_lines(self, prediction, src_output=None, tgt_output=None, progress_monitor=True):
        """
        Write the predicted pairs to disk.

        :param prediction: a tuple of the format (source_id, target_id)
        """
        # logging.info("Writing {} output pairs to disk...".format(len(prediction)))
        src_name = self.src.split("/")[-1]
        tgt_name = self.tgt.split("/")[-1]
        dir_path = os.getcwd()
        close_file = False
        if src_output is None:
            close_file = True
            src_output_fname = "{}/{}.{}".format(dir_path, src_name, self.ref_similarity_metric)
            src_output = open(src_output_fname, "w")
        if tgt_output is None:
            close_file = True
            tgt_output_fname = "{}/{}.{}".format(dir_path, tgt_name, self.ref_similarity_metric)
            tgt_output = open(tgt_output_fname, "w")

        if progress_monitor:
            pbar = tqdm(desc="Write lines", total=len(prediction))

        # Merge all target sentences that point to the same source sentence.
        # Careful with that one when using low threshold.
        if self.merge:
            tgt_src_map = {}
            for src_id, tgt_id in prediction:
                if tgt_id in tgt_src_map.keys():
                    tgt_src_map[tgt_id].append(src_id)
                else:
                    tgt_src_map[tgt_id] = [src_id]
            for tgt_id in tgt_src_map.keys():
                tgt_line = self.get_tgt_line(tgt_id)
                src_line = " ".join([self.get_src_line(src_id).strip().replace("\n", "")
                                        for src_id in tgt_src_map[tgt_id]])
                src_line = "{}\n".format(src_line)
                if self.final_check(src_line, tgt_line):
                    src_output.write(src_line)
                    tgt_output.write(tgt_line)
                    if len(tgt_src_map[tgt_id]) > 1:
                        self.merge_count += 1
                if progress_monitor:
                    pbar.update()
        else:
            for i, j in prediction:
                src_line = self.get_src_line(i)
                tgt_line = self.get_tgt_line(j)
                if self.final_check(src_line, tgt_line):
                    src_output.write(src_line)
                    tgt_output.write(tgt_line)
                if progress_monitor:
                    pbar.update()

        if close_file:
            src_output.close()
            tgt_output.close()
