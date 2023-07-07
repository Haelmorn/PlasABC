import pickle
import numpy as np

def load_pickled_embeddings(plasmid_file, chromosome_file):
    """
    Load pickled embeddings from disk
    :param plasmid_file: Path to pickled plasmid protein embeddings
    :param chromosome_file: Path to pickled chromosome protein embeddings
    :return: data, labels
    """
    try:
        with open(chromosome_file, 'rb') as f:
            chromosomes = pickle.load(f)
            # Tuple to dict
            chromosomes = {k: v for k, v in chromosomes}

        with open(plasmid_file, 'rb') as f:
            plasmids = pickle.load(f)
            # Tuple to dict
            plasmids = {k: v for k, v in plasmids}

    except:
        print("Error while loading datasets - make sure you use parsed embeddings (see README: Data Preprocessing)")
        exit()

    # Convert to numpy arrays
    try:
        chromosomes = {k: np.array(v) for k, v in chromosomes.items()}
        plasmids = {k: np.array(v) for k, v in plasmids.items()}
    except:
        print("Error while converting datasets to numpy arrays - make sure your data has the correct format")
        exit()

    data = list(chromosomes.values()) + list(plasmids.values())
    labels = [0]*len(chromosomes) + [1]*len(plasmids)

    return data, labels