import torch
import torch.nn as nn
import numpy as np

from monai.networks.nets import DenseNet121

# DenseNet121 Model
def create_monai_densenet(num_classes, lr):
    # The 'spatial_dims' argument specifies the number of spatial dimensions, which is 2 for typical medical images
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_classes)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, optimizer

# Custom Dense Model
def CustomDenseModel(num_classes, lr, resize_shape):
    class CustomDenseNet(nn.Module):
        def __init__(self, num_classes):
            super(CustomDenseNet, self).__init__()
            # Define your custom architecture here
            # Example:
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                # Add more layers as needed...
            )
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
                nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    model = CustomDenseNet(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, optimizer


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased at epoch {epoch}: {self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
