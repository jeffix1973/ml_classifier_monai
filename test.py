import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Framework imports
import classifier.build_model as build_model
import classifier.generators as generators
import classifier.utils as utils

def run_testing(args):
    # Load configuration and model
    json_path = args.json_path
    data_root_path = args.data_root_path
    config, _ = utils.load_json(json_path)
    
    # Extract configurations
    root_path = config['root_path']
    output_dir = config['output_dir']
    resize_shape = config['size']
    lr = config['lr']
    batch_size = config['batch_size']
    label_names = config['label_names']
    nb_labels = len(label_names)
    label_mapping_json = {i: label for i, label in enumerate(label_names)}
    
    # Prepare output directories
    run_name = f"{config['run']}_{config['network']}_{config['runtype']}"
    train_output_dir = os.path.join(root_path, output_dir, "out", run_name, "train")
    test_output_dir = os.path.join(root_path, output_dir, "out", run_name, "test")
    utils.create_dir(test_output_dir)
    
    running_csv = os.path.join(root_path, output_dir, "out", run_name, "split_for_predict", "db.csv")

    # Assuming your model's name is 'DenseNet121' in build_model
    model, _ = build_model.create_monai_densenet(nb_labels, lr)
    model.load_state_dict(torch.load(os.path.join(train_output_dir, 'best_model.pth')))
    model.eval()

    # Setup DataLoader for test data
    test_loader = generators.create_test_dataloader(
        data_root_path,
        running_csv,
        resize_shape,
        batch_size,
        nb_labels
    )

    true_labels = []
    predicted_labels = []
    probabilities = []

    # Testing loop
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            probabilities.extend(F.softmax(outputs, dim=1).numpy())
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.numpy())
    
    true_labels = pd.read_csv(running_csv)['label'].to_numpy()

    # Extract the maximum probability for each prediction
    max_probs = np.max(probabilities, axis=1) * 100
    
    probabilities = np.vstack(probabilities)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.clf()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    
    # add accuracy, precision, recall, f1-score to title
    accuracy = np.round(100 * np.sum(true_labels == predicted_labels) / len(true_labels), 1)
    precision = np.round(100 * precision_score(true_labels, predicted_labels, average='weighted'), 1)
    recall = np.round(100 * recall_score(true_labels, predicted_labels, average='weighted'), 1)
    f1 = np.round(100 * f1_score(true_labels, predicted_labels, average='weighted'), 1)
    
    plt.title(f"{run_name} / {len(predicted_labels)} series\n" +
            f"Accuracy: {accuracy}% / " +
            f"Precision: {precision}% / " +
            f"Recall: {recall}% / " +
            f"F1-score: {f1}%")
    plt.ylabel('EXPECTED (gt)')
    plt.xlabel('PREDICTED')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(test_output_dir, 'confusion_matrix.png'), format='png', bbox_inches = 'tight')

    # Save predictions to CSV
    test_df = pd.read_csv(running_csv)
    results_df = pd.DataFrame({
        'Path': test_df['OLEA_INSTANCE_PATH'],
        'SeriesDescription': test_df['SeriesDescription'],
        'GT': [label_mapping_json[gt] for gt in true_labels],
        'Prediction': predicted_labels,
        'Max_Prob': np.round(max_probs, 1)
    })
    
    # Add probabilities for each class
    for i, class_name in enumerate(label_names):
        results_df[class_name] = np.round(probabilities[:, i] * 100, 1)

    results_df.to_csv(os.path.join(test_output_dir, 'results.csv'), index=False)

    print(f"Testing completed. Metrics: Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f"Results saved to {os.path.join(test_output_dir, 'predictions.csv')}")
    print(f"Confusion matrix saved to {os.path.join(test_output_dir, 'confusion_matrix.png')}")

    pass
