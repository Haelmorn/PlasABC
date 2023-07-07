import argparse
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from model import PlasABC, ProteinDataset
from utils import load_pickled_embeddings
from torch.utils.data import DataLoader

# Training settings,
parser = argparse.ArgumentParser(description='PlasABC - attention based plasmid classifier')
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-l', '--learning_rate', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('-s', '--random_seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('-p', '--plasmid_embeddings', type=str, required=True,
                    help='Path to pickled plasmid protein embeddings')
parser.add_argument('-c', '--chromosome_embeddings', type=str, required=True,
                    help='Path to pickled chromosome protein embeddings')
parser.add_argument('-m', '--model_path', type=str,
                    help='Path to save trained model')

# parse the arguments
args = parser.parse_args()

# set CUDA flag
args.cuda = not args.no_cuda and torch.cuda.is_available()


# set manual seed for torch
torch.manual_seed(args.random_seed)

# set seed for CUDA and print GPU info if CUDA is available
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)
    print(f'Using GPU: {torch.cuda.get_device_name()}')


def main():
    # load embeddings from pickled files
    print('Loading training and testing sets...')
    data, labels = load_pickled_embeddings(args.plasmid_embeddings, args.chromosome_embeddings)

    # split data into training and testing sets
    print("Splitting data into training and testing sets...")
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=args.random_seed)

    # create data loaders for training and testing datasets
    print("Creating DataLoaders...")
    train_dataset = ProteinDataset(data_train, labels_train)
    test_dataset = ProteinDataset(data_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # initialize PlasABC model
    print('Initializing PlasABC model...')
    model = PlasABC()
    if args.cuda:
        model.cuda()  # move model to GPU if CUDA is available

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # training loop
    print('Training PlasABC model...')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.
        train_acc=0.
        for data, label in train_loader:
            data = data.squeeze(0)
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = torch.autograd.Variable(data), torch.autograd.Variable(label)
            optimizer.zero_grad()
            loss, acc, _, _,_ = model.calculate_objective(data, label)
            train_loss += loss.item()
            train_acc += acc
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        # Validation
        model.eval()
        val_loss = 0.
        val_acc = 0.
        with torch.no_grad():
            for data, label in test_loader:
                data = data.squeeze(0)
                if args.cuda:
                    data, label = data.cuda(), label.cuda()
                data, label = torch.autograd.Variable(data), torch.autograd.Variable(label)
                loss, acc, _, _,_ = model.calculate_objective(data, label)
                val_loss += loss.item()
                val_acc += acc
        val_loss /= len(test_loader)
        val_acc /= len(test_loader)
        print(f'Epoch: {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    print("Finished Training")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # save the model if path is provided
    if args.model_path:
        print(f'Saving model to {args.model_path}')
        torch.save(model.state_dict(), args.model_path)

if __name__ == '__main__':
    main()
