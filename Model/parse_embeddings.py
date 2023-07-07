import numpy as np
import h5py
import pickle
from tqdm import tqdm
import re
import argparse

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split('(\d+)', s)]

# Set up argument parser
parser = argparse.ArgumentParser(description="Process embeddings from HDF5 file and save as pickle")
parser.add_argument('--input', '-i', required=True, help="Input file path. Expects an HDF5 file.")
parser.add_argument('--output', '-o', required=True, help="Output file path. Outputs a pickle file.")
args = parser.parse_args()

# Placeholder for the zero array
zeros = np.zeros(1024,)

# Initialize all_embeds
all_embeds = []

# Open the input HDF5 file
with h5py.File(args.input, 'r') as embeddings_file:
    keys = sorted(embeddings_file.keys(), key=natural_sort_key)

    # Initialize an empty list for the first contig
    current_contig = keys[0].split(" ")[:-1][0]
    current_contig = "_".join(current_contig.split("_")[:-1])
    embeds = []

    for k in tqdm(keys):
        full_id = k.split(" ")[:-1][0]
        ctg_name, prot_id = "_".join(full_id.split("_")[:-1]), full_id.split("_")[-1]

        if ctg_name != current_contig:
            # A new contig is found, add the current contig and its embeddings to all_embeds
            all_embeds.append((current_contig, np.array(embeds)))
            embeds = []

        # Check if length of embeddings is 0 - if so, add a zero array
        # This means that the protein was probably too big to be embedded
        if len(embeddings_file[k][()]) == 0:
            embeds.append(zeros)
        else:
            embeds.append(embeddings_file[k][()])

        current_contig = ctg_name

    # Add the last contig's embeddings
    if embeds:
        all_embeds.append((current_contig, np.array(embeds)))

# Save all_embeds as a pickle file
with open(args.output, "wb") as f:
    pickle.dump(all_embeds, f)

