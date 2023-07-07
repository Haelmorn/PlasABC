import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class PlasABC(nn.Module):
    def __init__(self):
        super(PlasABC, self).__init__()
        self.L = 800
        self.D = 128
        self.K = 1
        self.extract_features = nn.Sequential(
            nn.Linear(1024, self.L),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        H = self.extract_features(x)
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob,Y_hat, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))
        train_acc = Y_hat.eq(Y).cpu().float().mean().item()
        return neg_log_likelihood, train_acc,A,Y_prob,Y_hat


class ProteinDataset(Dataset):
    def __init__(self, data, labels):
        self.data = [torch.from_numpy(d).float() for d in data]
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]