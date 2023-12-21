import os
import torch
import datetime

from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
from monai.config.deviceconfig import print_config

from tqdm import tqdm

# Framework imports
import classifier.utils as utils
import classifier.generators as generators
import classifier.preprocessing as preprocessing
import classifier.build_model as build_model
# import early_stopping class from classifier/build_model.py
from classifier.build_model import EarlyStopping


def run_training(args):
    
    print_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=0)
    
    # Load configuration file
    json_path = args.json_path
    data_root_path = args.data_root_path
    config, _ = utils.load_json(json_path)

    # Extract configurations
    root_path = config['root_path']
    output_dir = config['output_dir']
    resize_shape = config['size']
    lr = config['lr']
    patience = config['patience']
    batch_size = config['batch_size']
    epochs = config['epochs']
    label_names = config['label_names']
    nb_labels = len(label_names)
    
    print('>>> Label_names : ', label_names)
    label_mapping_json = {i: label for i, label in enumerate(label_names)}
    print('>>> Label mapping : ', label_mapping_json)
    print('>>> Nb Of Labels :', nb_labels)

    # Prepare output directories
    run_name = f"{config['run']}_{config['network']}_{config['runtype']}"
    train_output_dir = os.path.join(root_path, output_dir, "out", run_name, "train")
    utils.create_dir(train_output_dir)

    # Debug data augmentation
    running_csv = os.path.join(root_path, output_dir, "out", run_name, "split", "db.csv")
    print("Debugging data augmentation...")
    preprocessing.debug_dataset_images(data_root_path, running_csv, resize_shape, 20)
    print("Debugging done, see folder /tmp/debug_images/ for results.")

    # Create Data Loaders
    train_loader, val_loader = generators.create_dataloaders(
        data_root_path=data_root_path,
        csv_file=running_csv,
        resize_shape=resize_shape,
        batch_size=batch_size,
        num_classes=nb_labels
    )

    # Model Initialization
    model, optimizer = build_model.create_monai_densenet(nb_labels, lr)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create log directory for TensorBoard
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + config['run'])
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Instantiate EarlyStopping class
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(train_output_dir, 'best_model.pth'))

    # Training and Validation Loop
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}") as train_bar:
            for inputs, targets in train_bar:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                # Update the progress bar
                current_accuracy = 100 * total_correct / total_samples
                train_bar.set_postfix(loss=loss.item(), accuracy=f'{current_accuracy:.2f}%')
        
        average_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * total_correct / total_samples
        writer.add_scalar('Loss/Train', average_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_train_loss}, Accuracy: {train_accuracy:.2f}%")

        # Validation
        model.eval()
        val_loss, val_total_correct, val_total_samples = 0, 0, 0
        with tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}") as val_bar:
            with torch.no_grad():
                for inputs, targets in val_bar:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_total_correct += (predicted == targets).sum().item()
                    val_total_samples += targets.size(0)

                    current_val_accuracy = 100 * val_total_correct / val_total_samples
                    val_bar.set_postfix(loss=loss.item(), accuracy=f'{current_val_accuracy:.2f}%')
        
        average_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_total_correct / val_total_samples
        writer.add_scalar('Loss/Validation', average_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        print(f"Validation Loss: {average_val_loss}, Accuracy: {val_accuracy:.2f}%")
        
        # Early stopping and checkpoint saving
        early_stopping(average_val_loss, model, epoch + 1)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    writer.close()
    print(f"Best model saved to {os.path.join(train_output_dir, 'best_model.pth')}")

    pass
