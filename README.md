# Large-scale Hierarchical Alignment

![Large-scale Hierarchical Alignment](lha.png?raw=true | width=300)

This code implements large-scale hierarchical alignment from the paper
[Large-scale Hierarchical Alignment for Data-driven Text Rewriting](https://arxiv.org/abs/1810.08237),
presented at RANLP 2019.

The code constructs [Annoy](https://github.com/spotify/annoy) indices
using document/sentence embeddings of two datasets, following which
it performs nearest neighbour search across the datasets.
It first extracts similar documents (document
alignment), and then similar sentences (sentence alignment). See the paper
for more info.

## Setting up

Install all project dependencies:

```
pip install -r requirements.txt
```

You will also need the [linecache_light](https://github.com/Yelrose/linecache_light)
library (not available through pip, you can install it
from source). I had to change a line in the file
`linecache_light/linecache_light.py` to get it working in Python 3:
from `import cPickle as pkl` to `import pickle as pkl`.

## Running the aligner

### 1. Build document embedding indices

The first step is to build an index of document embeddings.
This is implemented in the file `build_annoy_index.py`, e.g. if you want
to align two files, *source* and *target*, that contain one document
per line, run:

```
python build_annoy_index.py -src_file source -emb sent2vec -vec_size 600
python build_annoy_index.py -src_file target -emb sent2vec -vec_size 600
```

to compute [sent2vec](https://github.com/epfml/sent2vec) embeddings.
You'd have to modify the script to point to the correct model file paths.

This will create two index files, *source.sent2vec.ann* and
*target.sent2vec.ann* for the above example. Run
`python build_annoy_index.py --help` for more info on all of the
available options. The [Annoy](https://github.com/spotify/annoy)
documentation also contains additional details.

### 2. Run the aligner

After you have the source and target indices prepared, you can run the aligner as:

```
python aligner.py -level hierarchical -src source -tgt target -emb sent2vec -vec_size 600 -batch_size 2000 -lower_th 0.65
```

run `python aligner.py --help` for more info on the available options.

This will first extract similar document pairs and then similar sentence pairs.
The final sentence pairs will be stored in two files:
*source.hier.None* and *target.hier.None*. Additionally, a file
*source.target.sims.None* will be created, which will contain the final
similarities of the sentence pairs.

After alignment, you can subsequently post-filter the extracted pairs,
e.g. using:

```
python filter.py -src source.hier.None -tgt target.hier.None -sim source.target.sims.None -low_sim_th 0.7
```

## Citation

```
@InProceedings{nikolov-alignment-ranlp19,
  author    = {Nikolov, Nikola  and  Hahnloser, Richard},
  title     = {Large-scale Hierarchical Alignment for Data-driven Text Rewriting},
  booktitle = {Proceedings of the International Conference Recent Advances in Natural Language Processing, RANLP 2019},
  year      = {2019}
}
```
