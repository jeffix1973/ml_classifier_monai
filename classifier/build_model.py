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
        # Check for the number of GPUs
        if torch.cuda.device_count() > 1:
            # Save the original model when model is wrapped with nn.DataParallel
            model_to_save = model.module.state_dict()
        else:
            # Save the entire model directly
            model_to_save = model.state_dict()
        
        # Save the model
        torch.save(model_to_save, self.path)
        if self.verbose:
            print(f'Validation loss decreased at epoch {epoch}: {self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model...')
        self.val_loss_min = val_loss
