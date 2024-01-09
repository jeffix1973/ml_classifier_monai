import os
import argparse
from classifier import utils

def main():
    
    parser = argparse.ArgumentParser(description="Copy models and settings to /models folder.")
    parser.add_argument("--json_path", required=True, help="Path to the 'common.json' configuration file.")
    
    args = parser.parse_args()
    
    json_path = args.json_path

    CONF = utils.load_json(json_path)
    config = CONF['bp_detection_inference_rest_api']
    config2 = CONF['report_publisher']
        
    # Extract configurations
    root_path = config2['root_path']
    output_dir = config2['output_dir']
    
    model_ax = config['model_ax'].split('.pt')[0]
    model_sag = config['model_sag'].split('.pt')[0]
    model_cor = config['model_cor'].split('.pt')[0]
    
    # Clean up /models folder
    clean_models()
    
    # Copy models and settings to /models folder
    ax_train_output_dir = os.path.join(root_path, output_dir, "out", model_ax, "train")
    copy_model(ax_train_output_dir, 'model_ax')
    sag_train_output_dir = os.path.join(root_path, output_dir, "out", model_sag, "train")
    copy_model(sag_train_output_dir, 'model_sag')
    cor_train_output_dir = os.path.join(root_path, output_dir, "out", model_cor, "train")
    copy_model(cor_train_output_dir, 'model_cor')
    
# Clean up /models folder
def clean_models():
    for file in os.listdir('models'):
        if file.endswith('.pth'):
            os.remove('models/' + file)

# Make a file copy of both best_model.pth and config.json from train_output_dir folder to current local /models folder
def copy_model(output_dir, model_name):
    for file in os.listdir(output_dir):
        if file.endswith('.pth'):
            os.system('cp ' + output_dir + '/' + file + ' models/' + model_name + '.pt')
        elif file.endswith('.json'):
            os.system('cp ' + output_dir + '/' + file + ' models/' + model_name + '.json')
            
if __name__ == "__main__":
    main()