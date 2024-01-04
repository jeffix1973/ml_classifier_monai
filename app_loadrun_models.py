import json
import numpy as np
import sys
import torch
import torch.nn.functional as F  # For softmax

from monai.networks.nets import DenseNet121

import classifier.preprocessing as preprocessing

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model(m_path: str):

    print('>>>> Loading', m_path + '.json')
    # Opening settings JSON file
    try:
        f = open(m_path + '.json', "r")
    except Exception as e:
        print(e)
        sys.exit(0)

    # Collect JSON part for this framework
    VAR = json.load(f)

    # Dectection threshold
    DT = VAR['DT']
    network = VAR['network']
    resize_shape = VAR['resize_shape']
    labels = VAR['labels']

    m = DenseNet121(spatial_dims=2, in_channels=1, out_channels=len(labels)).to(device)
    
    state_dict = torch.load(m_path + '.pt', map_location=device)
    m.load_state_dict(state_dict)
    m.eval()

    return m, labels, DT, resize_shape


def run_model(model, labels, DT, resize_shape, path):
    
    # Print loaded model
    print('>>> Labels loaded: ', labels)
    print('>>> Detection threshold: ', DT)
    print('>>> Resize shape: ', resize_shape)
    print('>>> Inferred slice path: ', path)
    
    # Preprocess image
    _, processed, uids = preprocessing.preprocess_pipeline(path, resize_shape, 'test')

    # Predict and apply softmax
    prediction_probas = model(processed.unsqueeze(0).to(device))
    probabilities = F.softmax(prediction_probas, dim=1)[0]

    # Get the predicted class index using torch
    prediction = torch.argmax(probabilities).item()  # Keep it as tensor here for argmax

    # Now convert probabilities to numpy array for further processing
    probabilities = probabilities.detach().cpu().numpy()

    # Print the prediction probabilities
    print("Prediction probabilities:")
    for i, label in enumerate(labels):
        print("{}: {:.2f}".format(label, probabilities[i]))

    # Print the predicted class
    predicted_label = labels[prediction]
    predicted_probability = round(probabilities[prediction] * 100, 1)  # Ensure probabilities is not overwritten anywhere
    print("Predicted class   >>>> {} (@ {}%)".format(predicted_label, predicted_probability))

    # Write results
    parsed_results = {}
    score = probabilities.tolist()  # Convert numpy array back to list for JSON serialization
    parsed_results['scores'] = score
    if max(score) >= DT:  # Ensure DT is defined somewhere in your code as the decision threshold
        parsed_results['prediction'] = predicted_label
    else:
        parsed_results['prediction'] = 'OTHER'

    # Return the results and the UID information
    return parsed_results, uids
