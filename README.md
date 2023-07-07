# PlasABC

**Plas**mid **A**ttention **B**ased **C**lassifier

Py-torch based module for identifying plasmids in a set of sequences, based on embeddings of protein sequences found in the sequences.

Depending on the training data, the model can be used to identify plasmids in draft genomes, or in metagenomic samples.

The model architecture was heavily inspired by the work done by Dan Liu *et al.* in [EvoMIL](https://github.com/liudan111/EvoMIL), as described in [this paper](https://www.biorxiv.org/content/10.1101/2023.04.07.536023v1)

## Installation

1. Clone the repository with `git clone https://github.com/Haelmorn/PlasABC.git`
2. Create a virtual environment with `python3 -m venv venv` and activate it with `source venv/bin/activate`
3. Install the requirements with `pip3 install -r requirements.txt`

## Usage

### Data

A small sample dataset is provided in the `Data/Sample` folder. The data is available in all formats used in training and classification (fasta, faa, hdf5, pickle). A detailed instruction on how to create the data is provided in the `Data` folder.

### Training

To check if the training process works properly, you can use the sample data provided with this repo. Bear in mind, that the sample data is not sufficient to train a model that can be used for classification. See `Data/README.md` for details on how to obtain full-size datasets.

1. Process the embeddings with 
```
python3 Model/parse_embeddings.py --input Data/Sample/Plasmid/plsdb_sample.h5 --output Data/Sample/Plasmid/plsdb_sample.pkl
``` 
and 
```
python3 Model/parse_embeddings.py --input Data/Sample/Chromosome/chromosome_sample.h5 --output Data/Sample/Chromosome/chromosome_sample.pkl
```

2. Train the model with 

```
python3 Model/train.py --plasmids plsdb.pkl --chromosomes bacteria.pkl --output model.pt
```

### Classification

To classify your sets of embeddings, simply run

```
python3 src/models/classify.py --plasmids plsdb_sample.h5 --model model.pt --output plsdb_sample_classified.h5
```

#### Output

The output file is a tab-delimited file with `contig_id` in one column, `classification_result` and `prediction`. The `prediction` is a boolean value, indicating whether the contig is classified as plasmid or not. The `classification_result` is the probability of the contig being a plasmid, as calculated by the model.
