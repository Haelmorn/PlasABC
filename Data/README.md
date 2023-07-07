# Data files used for training/testing/validation

# Data
Use dropbox/s3 to hold the files?
Things to deposit:
* Raw fastas [ ]
* Raw faa's [ ]
* Embeddings [ ]
* pickle files with embeddings [ ]
* Model weights [ ]

## Data descriptions
...

## Sample data

#### Plasmids

1. Obtain PLSDb database from [ccb-microbe.cs.uni-saarland.de](ccb-microbe.cs.uni-saarland.de)
2. Unzip the archive with `bzip2 -d plsdb.fna.bz2`
3. Subsample N plasmids with `seqkit sample -n N plsdb.fna`. Here I used 10 plasmids to make training fast
4. Get predicted CDS with `prodigal -i plsdb_sample.fna -a plsdb_sample.faa -p meta`
5. Obtain protein embeddings with ProtTrans model. See [ProtTrans](https://github.com/agemagician/ProtTrans) and ProtTrans section below for details

#### Chromosomes

1. Obtain RefSeq database from [ftp://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/](ftp://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/)
2. Unzip the archive with `bzip2 -d bacteria.*.genomic.fna.gz`
3. Subsample chromosomes. Here I used one to make training fast.
4. Get predicted CDS with `prodigal -i bacteria_sample.fna -a bacteria_sample.faa`
5. Obtain protein embeddings with ProtTrans model. See [ProtTrans](https://github.com/agemagician/ProtTrans) and ProtTrans section below for details