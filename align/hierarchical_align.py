import logging
from tqdm import *
import numpy as np
import os
import nltk
from multiprocessing import Process, Manager, cpu_count
import itertools

from align.align_base import AlignerBase


class HierarchicalAligner(AlignerBase):
    """Hierarchical Aligner
    Performs Global --> Local alignment."""

    def __init__(self, global_pairs, *args, **kw):
        super(self.__class__, self).__init__(*args, **kw)
        self.max_len = 50
        self.min_len = 4
        self.len_ratio_max = 10
        self.global_pairs = global_pairs
        if self.global_pairs is not None:
            logging.warning("Document pairs already provided. Skipping document alignment.")

        src_name = self.src.split("/")[-1]
        tgt_name = self.tgt.split("/")[-1]
        dir_path = os.getcwd()

        src_output_fname = "{}/{}.hier.{}".format(dir_path, src_name, self.ref_similarity_metric)
        tgt_output_fname = "{}/{}.hier.{}".format(dir_path, tgt_name, self.ref_similarity_metric)
        similarities_fname = "{}/{}.{}.sims.{}".format(dir_path, src_name, tgt_name, self.ref_similarity_metric)
        self.src_output_file = open(src_output_fname, "w")
        self.tgt_output_file = open(tgt_output_fname, "w")
        self.src_doc_ids = open("{}.id".format(src_output_fname), "w")
        self.tgt_doc_ids = open("{}.id".format(tgt_output_fname), "w")
        self.src_tgt_similarities = open(similarities_fname, "w")

    def predict_batch(self, silent=False, lower_document_sim_th=0.5):
        """Compute prediction in batch mode, for big datasets."""
        total_target = self.get_tgt_count()
        batch_size = int(total_target / self.batch_size)
        batches = np.array_split(np.arange(total_target), batch_size)
        if not silent:
            logging.warning("Batch size: {}".format(len(batches[0])))

        if not silent:
            pbar = tqdm(total=len(batches), desc='Global align batch')
        for i, batch in enumerate(batches):
            if self.global_pairs is not None:
                # We already have the global pairs available
                document_pair_prediction = [self.global_pairs[i] for i in batch]
            else:
                # Document (Global) alignment
                similarity_matrix = self.global_similarity_struct_parallel(
                    tgt_range=(batch[0], batch[-1]), progress_monitor=not silent, skip_refine=True)
                document_pair_prediction = self.similarity_to_prediction(
                    similarity_matrix, lower_th=lower_document_sim_th)

            # Sentence alignment
            sentence_pair_prediction = self.sentence_alignment(document_pair_prediction)
            # Write output sentence pairs
            self.write_output_lines(sentence_pair_prediction)
            if not silent:
                pbar.update()

        self.src_output_file.close()
        self.tgt_output_file.close()
        self.src_tgt_similarities.close()

    def write_output_lines(self, prediction, src_output=None, tgt_output=None,
                           progress_monitor=True, lower=True):
        sentences, sent_similarities, sentence_doc_ids = prediction
        for (src_sent, tgt_sent), sim, (src_doc_id, tgt_doc_id) \
                in zip(sentences, sent_similarities, sentence_doc_ids):
            self.src_output_file.write("{}\n".format(src_sent.lower()))
            self.tgt_output_file.write("{}\n".format(tgt_sent.lower()))
            self.src_tgt_similarities.write("{}\n".format(sim))
            self.src_doc_ids.write("{}\n".format(src_doc_id))
            self.tgt_doc_ids.write("{}\n".format(tgt_doc_id))

            self.src_output_file.flush()
            self.tgt_output_file.flush()
            self.src_tgt_similarities.flush()
            self.src_doc_ids.flush()
            self.tgt_doc_ids.flush()

    def split_embed_sentences(self, in_queue, out_list):
        """Compute a single embedding"""
        while True:
            input_id, input_text = in_queue.get()
            if input_text is None:  # exit signal
                return
            sentences = nltk.sent_tokenize(input_text)
            input_emb = [self.get_sent2vec_emb(txt) for txt in sentences]
            out_list.append((input_id, sentences, input_emb))

    def parallel_embed(self, chunk, size=None, txt="", n_threads=int(cpu_count() / 2)):
        """Compute embeddings for a chunk using multithreading."""
        # Set up simple multiprocessing
        manager = Manager()
        results = manager.list()
        work = manager.Queue(n_threads)

        pool = []
        for i in range(n_threads):
            p = Process(target=self.split_embed_sentences, args=(work, results))
            p.start()
            pool.append(p)

        chunk_iter = itertools.chain(iter(chunk), enumerate((None,) * n_threads))

        with tqdm(total=size, desc="Embed {}".format(txt)) as pbar2:
            for document_id, document in chunk_iter:
                work.put((document_id, document))
                if document is not None:
                    pbar2.update()

        for p in pool:
            p.join()

        if size is not None:
            assert len(results) == size
        return results

    def parallel_pair_similarity_sent_new(self, in_queue, out_list):
        while True:
            pair_id, new_pair = in_queue.get()
            if new_pair is None:
                return
            doc_ids, sentence_id_pairs, sentence_texts = new_pair
            # sentence_id_pairs is a list of lists, where each sublist is containing the ID of the pair
            # sentence_texts is a tuple containing two lists, each containing the text of the source/target sentences.
            for pair, src_sent, tgt_sent in zip(sentence_id_pairs, sentence_texts[0], sentence_texts[1]):
                similarity = self.pair_similarity(None, None, src_sent, tgt_sent)
                if isinstance(similarity, np.float64) or isinstance(similarity, float):
                    result = (doc_ids, pair, similarity)
                else:
                    # problem with similarity function..
                    logging.warning("Weird similarity output {}... skipping: {} {} {} {}".format(
                        type(similarity), pair, src_sent, tgt_sent, similarity))
                    result = (doc_ids, pair, 0.)
                out_list.append(result)

    def parallel_refine_sent_new(self, doc_prediction, sentence_ids_to_refine,
                                 sentences_to_refine, progress_bar=True,
                                 num_workers=int(cpu_count() / 2)):
        """Compute refine function in parallel"""
        manager = Manager()
        results = manager.list()
        work = manager.Queue(num_workers)

        # start for workers
        pool = []
        for i in range(num_workers):
            p = Process(target=self.parallel_pair_similarity_sent_new, args=(work, results))
            p.start()
            pool.append(p)

        # Pairs to refine is list of lists. Each sublist is containing the sentence IDs
        # Sentences_to_refine is a list of tuples. In each tuple, there are two lists that contain
        # the text of the source and target sentences.
        iters = itertools.chain(
            zip(doc_prediction, [sentence_ids_to_refine[p] for p in doc_prediction],
                [sentences_to_refine[p] for p in doc_prediction]), (None,) * num_workers)
        if progress_bar:
            pbar = tqdm(desc="Refine {}".format(self.ref_similarity_metric))

        for id_pair in enumerate(iters):
            work.put(id_pair)
            if progress_bar:
                pbar.update()

        for p in pool:
            p.join()

        return results

    def compute_all_sentence_embeddings_parallel(self, doc_prediction):
        src_all_lines = {}
        tgt_all_lines = {}
        src_all_embs = {}
        tgt_all_embs = {}

        # Load the source/target documents from disk
        src_idx = [p[0] for p in doc_prediction]
        tgt_idx = [p[1] for p in doc_prediction]

        # Embed all sentences in the current batch
        src_docs = zip(src_idx, [self.get_src_line(idx) for idx in src_idx])
        src_results = self.parallel_embed(src_docs, size=len(src_idx), txt="src")
        for d_id, sentences, sentence_embeddings in src_results:
            src_all_lines[d_id] = sentences
            src_all_embs[d_id] = sentence_embeddings

        tgt_docs = zip(tgt_idx, [self.get_tgt_line(idx) for idx in tgt_idx])
        tgt_results = self.parallel_embed(tgt_docs, size=len(tgt_idx), txt="tgt")
        for d_id, sentences, sentence_embeddings in tgt_results:
            tgt_all_lines[d_id] = sentences
            tgt_all_embs[d_id] = sentence_embeddings

        return src_all_lines, tgt_all_lines, src_all_embs, tgt_all_embs

    def all_sent_align_sim_matrices(self, doc_prediction, src_all_embs, tgt_all_embs,
                                    src_all_lines, tgt_all_lines):
        # Contains all sentence-level similarity matrices, for all aligned document pairs
        sim_matrices = {}

        # Perform the alignment
        sentence_ids_to_refine = {}
        sentence_texts_to_refine = {}
        with tqdm(total=len(doc_prediction), desc="Sent matrix") as sent_pbar1:
            for src_doc_id, tgt_doc_id in doc_prediction:
                # Base matrix (based on cosine similarity of sentence embeddings)
                base_matrix = self.cosine_sim_matrix(
                    None, None,
                    src_emb=src_all_embs[src_doc_id],
                    tgt_emb=tgt_all_embs[tgt_doc_id])
                # Optionally refine the similarities in the base matrix
                if self.ref_similarity_metric is not None:
                    pairs_to_refine = np.argwhere(base_matrix > (self.lower_th - 0.05))
                    sentence_ids_to_refine[(src_doc_id, tgt_doc_id)] = pairs_to_refine
                    # Get all source/target sentences that need to be refined, store them
                    src_refine_sents = [src_all_lines[src_doc_id][pair[0]] for pair in pairs_to_refine]
                    tgt_refine_sents = [tgt_all_lines[tgt_doc_id][pair[1]] for pair in pairs_to_refine]
                    sentence_texts_to_refine[(src_doc_id, tgt_doc_id)] = (src_refine_sents, tgt_refine_sents)
                    base_matrix = np.zeros_like(base_matrix)
                sim_matrices[(src_doc_id, tgt_doc_id)] = base_matrix
                sent_pbar1.update()

        # Perform refinement of the similarity matrix
        if self.ref_similarity_metric is not None:
            assert len(sim_matrices) == len(sentence_ids_to_refine) == len(sentence_texts_to_refine)
            for doc_id, sentence_pair, similarity in self.parallel_refine_sent_new(
                    doc_prediction, sentence_ids_to_refine, sentence_texts_to_refine):
                sim_matrices[doc_id][sentence_pair[0], sentence_pair[1]] = similarity

        return sim_matrices

    def sentence_prediction(self, src_all_lines, tgt_all_lines, doc_prediction, sim_matrices):
        """
        :param src_all_lines:
        :param tgt_all_lines:
        :param doc_prediction:
        :param sim_matrices:
        :return:
        """
        out_pairs = []
        all_similarities = []
        sentence_doc_ids = []

        with tqdm(total=len(doc_prediction), desc="Sent align") as sent_pbar:
            for src_doc_id, tgt_doc_id in doc_prediction:
                base_matrix = sim_matrices[(src_doc_id, tgt_doc_id)]
                # Align from target to source
                tgt_matches = {}
                for tgt_id in range(base_matrix.shape[1]):
                    similarity_slice = base_matrix[:, tgt_id]
                    # Sort and get the top k
                    nearest_local_sent = np.argsort(similarity_slice)[::-1][:self.out_k_best]
                    # Similarities
                    similarities = [similarity_slice[idx] for idx in nearest_local_sent]
                    for sent_id, sim in zip(nearest_local_sent, similarities):
                        if sim > self.lower_th:
                            if tgt_id in tgt_matches.keys():
                                tgt_matches[tgt_id].append(sent_id)
                            else:
                                tgt_matches[tgt_id] = [sent_id]

                # Align from source to target
                src_matches = {}
                for src_id in range(base_matrix.shape[0]):
                    similarity_slice = base_matrix[src_id, :]
                    nearest_local_sent = np.argsort(similarity_slice)[::-1][:self.out_k_best]
                    similarities = [similarity_slice[idx] for idx in nearest_local_sent]
                    for sent_id, sim in zip(nearest_local_sent, similarities):
                        if sim > self.lower_th:
                            if src_id in src_matches.keys():
                                src_matches[src_id].append(sent_id)
                            else:
                                src_matches[src_id] = [sent_id]

                # Merge the source/target alignments, and construct the final pairs.
                for tgt_id in tgt_matches.keys():
                    src_selection = tgt_matches[tgt_id]
                    tgt_selection = []
                    for a in src_selection:
                        for k in src_matches[a]:
                            tgt_selection.append(k)

                    src_selection = list(set(src_selection))
                    tgt_selection = list(set(tgt_selection))
                    src_selection.sort()
                    tgt_selection.sort()

                    # Get all the lines
                    src_lines = [src_all_lines[src_doc_id][k] for k in src_selection
                                 if self.max_len > len(src_all_lines[src_doc_id][k].split()) > self.min_len]
                    tgt_lines = [tgt_all_lines[tgt_doc_id][k] for k in tgt_selection
                                 if self.max_len > len(tgt_all_lines[tgt_doc_id][k].split()) > self.min_len]

                    sim = 0.
                    for s in src_selection:
                        t_sims = [base_matrix[s, t] for t in tgt_selection]
                        sim += np.max(t_sims)
                    sim /= len(src_selection)

                    src_line_str = " ".join(src_lines)
                    tgt_line_str = " ".join(tgt_lines)

                    # Check if the pair is a good one
                    if (src_line_str, tgt_line_str) not in out_pairs \
                            and self.final_check(src_line_str, tgt_line_str):
                        out_pairs.append((src_line_str, tgt_line_str))
                        all_similarities.append(sim)
                        sentence_doc_ids.append((src_doc_id, tgt_doc_id))

                sent_pbar.update()

        return out_pairs, all_similarities, sentence_doc_ids

    def sentence_alignment(self, doc_prediction, parallel_run=True):
        """
        Perform sentence alignment

        :param doc_prediction: list of document ID pairs, representing the documents to be aligned
        :param parallel_run: If true will perform computations in parallel
        :return:
        """

        # Embed the sentences in the selected documents
        src_all_lines, tgt_all_lines, src_all_embs, tgt_all_embs = \
            self.compute_all_sentence_embeddings_parallel(doc_prediction)
        # Compute similarity matrices for each document pair
        sim_matrices = self.all_sent_align_sim_matrices(
            doc_prediction, src_all_embs, tgt_all_embs, src_all_lines, tgt_all_lines)
        # Compose the final list of aligned sentence pairs.
        return self.sentence_prediction(
            src_all_lines, tgt_all_lines, doc_prediction, sim_matrices)

