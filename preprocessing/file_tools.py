""""""

import logging, os, random, subprocess
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def prepare_files(target_folder, force_reshuffle=True):
    """Collect file names and optionally shuffle."""
    xml_file_names = []
    journal_folders = [x[0] for x in os.walk(target_folder)]
    for journal in journal_folders:
        all_files = os.listdir(journal)
        for article_file_name in all_files:
            if article_file_name.endswith(".XML") or article_file_name.endswith(".nxml") \
                    or article_file_name.endswith(".xml") \
                    or article_file_name.endswith(".text-align"):
                xml_file_names.append(journal + "/" + article_file_name)
    logging.info("No of journals: {}\t No of text-align files: {}".format(len(journal_folders), len(xml_file_names)))
    if force_reshuffle:
        logging.info("Shuffling files...")
        random.shuffle(xml_file_names)
    return xml_file_names


def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def wc(f):
    logging.info("running \"wc -l {}\"...".format(f))
    if type(f) is list:
        return tuple([file_len(subf) for subf in f])
    elif type(f) is str:
        return file_len(f)
    else:
        raise Exception