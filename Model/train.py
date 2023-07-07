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
parser.add_argument('-l', '--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('-s', '--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('-p', '--plasmid', type=str, required=True,
                    help='Path to pickled plasmid protein embeddings')
parser.add_argument('-c', '--chromosomes', type=str, required=True,
                    help='Path to pickled chromosome protein embeddings')
parser.add_argument('-m', '--model_file', type=str,
                    help='Path to save trained model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if not args.no_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    print(f'Using GPU: {torch.cuda.get_device_name()}')


def main():
    print('Load Train and Test Set')
    data, labels = load_pickled_embeddings(args.plasmid_file, args.chr_file)

    print("Splitting into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=args.seed)

    print("Creating DataLoaders")
    train_dataset = ProteinDataset(X_train, y_train)
    test_dataset = ProteinDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print('Initialising PlasABC model')
    model = PlasABC()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Training PlasABC model')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.
        train_acc=0.
        for data, label in train_loader:
            data = data.squeeze(0)
            if torch.cuda.is_available():
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
                if torch.cuda.is_available():
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

    if args.model_file:
        print(f'Saving model to {args.model_file}')
        torch.save(model.state_dict(), args.model_file)

if __name__ == '__main__':
    main()