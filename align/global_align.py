import logging
from tqdm import *
import numpy as np
import os

from align.align_base import AlignerBase


class GlobalAligner(AlignerBase):
    """Global aligner -- e.g. document level."""

    def __init__(self, *args, **kw):
        super(self.__class__, self).__init__(*args, **kw)

    def predict_batch(self, silent=False):
        """Compute prediction in batch mode, for big datasets."""
        total_target = self.get_tgt_count()
        batch_size = int(total_target / self.batch_size)
        batches = np.array_split(np.arange(total_target), batch_size)
        if not silent:
            logging.warning("Batch size: {}".format(len(batches[0])))
        global_prediction = set()

        src_name = self.src.split("/")[-1]
        dir_path = os.getcwd()

        src_name = self.src.split("/")[-1]
        tgt_name = self.tgt.split("/")[-1]
        dir_path = os.getcwd()

        src_output_fname = "{}/{}.{}".format(dir_path, src_name, self.ref_similarity_metric)
        tgt_output_fname = "{}/{}.{}".format(dir_path, tgt_name, self.ref_similarity_metric)
        src_output_file = open(src_output_fname, "w")
        tgt_output_file = open(tgt_output_fname, "w")

        if not silent:
            pbar = tqdm(total=len(batches), desc='Global align batch')
        for i, batch in enumerate(batches):
            similarity_matrix = self.global_similarity_struct_parallel(
                tgt_range=(batch[0], batch[-1]), progress_monitor=not silent)
            local_prediction = self.similarity_to_prediction(similarity_matrix)
            self.write_output_lines(local_prediction, src_output_file, tgt_output_file,
                progress_monitor=not silent)
            global_prediction.update(local_prediction)
            if not silent:
                pbar.update()

        src_output_file.close()
        if not self.filter:
            tgt_output_file.close()
        return list(global_prediction)

