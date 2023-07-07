# parse arguments
import argparse
import pickle

import torch
from Model.model import PlasABC
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='PlasABC - attention based plasmid classifier')
parser.add_argument('--input', type=str, required=True,
                    help='Path to pickled protein embeddings')
parser.add_argument('--model', type=str, required=True,
                    help='Path to trained plasABC model')
parser.add_argument('--output', type=str, required=True,
                    help='Path to output file')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Threshold for plasmid classification (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use GPU for classification')
parser.add_argument('--attention_output', type=str, default=None,
                    help='Path to output file for attention weights (default: None)')

args = parser.parse_args()

def load_model(model_file):
    """
    Load pre-trained plasABC model
    :param model_file: Path to trained plasABC model
    :return: plasABC model
    """
    model = PlasABC().float()
    model.load_state_dict(torch.load(model_file))

    if args.cuda:
        model = model.to('cuda:0')

    model.eval()

    return model

def classify(embeddings_file, model_obj):
    """
    Use plasABC model to classify sets of protein embeddings
    :param embeddings_file: Path to pickled protein embeddings
    :param model_file: Path to trained plasABC model
    :return: dictionary of plasmid predictions {sequence_id: score}
    """
    # Load embeddings
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    embeddings_dict = {k: np.array(v) for k, v in embeddings}

    # Predict
    predictions_dict = {}
    attention_weights_dict = {}

    for contig_id, embedding_data in embeddings_dict.items():
        if args.cuda:
            embedding_data = torch.from_numpy(embedding_data).float().to('cuda:0')
        else:
            embedding_data = torch.from_numpy(embedding_data).float()

        with torch.no_grad():
            probabilities, _, attention_weights = model_obj(embedding_data)

            # Transpose the attention weights back to original dimension
            attention_weights = torch.transpose(attention_weights, 1, 0).cpu().numpy()

            # Find the index of protein with the maximum attention weight
            max_weight_protein_index = np.argmax(attention_weights)

            # Store the index of the protein with maximum attention
            attention_weights_dict[contig_id] = max_weight_protein_index + 1
            predictions_dict[contig_id] = probabilities.item()

    return predictions_dict, attention_weights_dict


def main():
    """
    Classify sequences as plasmid or chromosome using PlasABC model
    :return: None
    """
    # Load model
    pretrained_model = load_model(args.model)

    classification_results, attention_weights = classify(args.input, pretrained_model)

    # Create a df with contig_id, classification_results, prediction
    df = pd.DataFrame.from_dict(classification_results, orient='index', columns=['classification_results'])
    df['prediction'] = df['classification_results'].apply(lambda x: 1 if x > args.threshold else 0)

    # Remove index and rename to contig_id
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'contig_id'}, inplace=True)

    # Write df to file
    df.to_csv(args.output, sep='\t', index=False)

    # Write attention weights to file
    if args.attention_output:
        with open(args.attention_output, 'w') as f:
            for k, v in attention_weights.items():
                f.write(f'{k}\t{v}\n')


if __name__ == '__main__':
    main()