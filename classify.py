# parse arguments
import argparse
import pickle

import torch
from Model.model import PlasABC
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='PlasABC - attention based plasmid classifier')
parser.add_argument('-i', '--input', type=str, required=True,
                    help='Path to pickled protein embeddings')
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Path to trained plasABC model')
parser.add_argument('-o', '--output', type=str, required=True,
                    help='Path to output file')
parser.add_argument('-t', '--threshold', type=float, default=0.5,
                    help='Threshold for plasmid classification (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use GPU for classification')
parser.add_argument('-a', '--attention_output', type=str, default=None,
                    help='Path to output file for attention weights (default: None)')

args = parser.parse_args()

def load_model(path_to_model):
    """
    Load a pre-trained PlasABC model.
    
    :param path_to_model: Path to the trained PlasABC model
    :return: PlasABC model
    """
    model = PlasABC().float()  # instantiate model and convert to float
    model.load_state_dict(torch.load(path_to_model))  # load model weights

    # if cuda flag is set, move model to GPU
    if args.cuda:
        model = model.to('cuda:0')

    model.eval()  # set model to evaluation mode

    return model

def classify_proteins(embeddings_file, model):
    """
    Classify proteins using the PlasABC model.
    
    :param embeddings_file: Path to the pickled protein embeddings
    :param model: PlasABC model
    :return: A dictionary of plasmid predictions {sequence_id: score} 
    :return: A dictionary of attention weights {sequence_id: attention_weights}
    """
    # Load embeddings
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)

    embeddings_dict = {k: np.array(v) for k, v in embeddings}

    # dictionaries to store predictions and attention weights
    predictions_dict = {}
    attention_weights_dict = {}

    # Classify each protein sequence
    for contig_id, embedding_data in embeddings_dict.items():
        if args.cuda:
            embedding_data = torch.from_numpy(embedding_data).float().to('cuda:0')
        else:
            embedding_data = torch.from_numpy(embedding_data).float()

        # perform forward pass and get probabilities and attention weights
        with torch.no_grad():
            probabilities, _, attention_weights = model(embedding_data)

            attention_weights = torch.transpose(attention_weights, 1, 0).cpu().numpy()  # transpose attention weights
            max_weight_protein_index = np.argmax(attention_weights)  # get the index of the protein with max attention

            attention_weights_dict[contig_id] = max_weight_protein_index + 1  # store index of protein with max attention
            predictions_dict[contig_id] = probabilities.item()  # store prediction

    return predictions_dict, attention_weights_dict


def main():
    """
    Classify sequences as plasmid or chromosome using PlasABC model
    """
    # load pre-trained model
    model = load_model(args.model)

    # classify sequences
    predictions, attention_weights = classify_proteins(args.input, model)

    # create a dataframe with classification results and predictions
    df = pd.DataFrame.from_dict(predictions, orient='index', columns=['classification_results'])
    df['prediction'] = df['classification_results'].apply(lambda x: 1 if x > args.threshold else 0)

    # reset index and rename column to 'contig_id'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'contig_id'}, inplace=True)

    # write dataframe to file
    df.to_csv(args.output, sep='\t', index=False)

    # write attention weights to file
    if args.attention_output:
        with open(args.attention_output, 'w') as f:
            for contig, weight in attention_weights.items():
                f.write(f'{contig}\t{weight}\n')


if __name__ == '__main__':
    main()
