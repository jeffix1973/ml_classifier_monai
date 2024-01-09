import os
import argparse
import datetime
import boto3

import classifier.utils as utils

def main():
    parser = argparse.ArgumentParser(description="Run training and/or testing.")
    parser.add_argument("--json_path", required=True, help="Path to the model_ax, sag or cor configuration file.")
    
    args = parser.parse_args()
    
    # collect s3 name in the environment variable
    s3_name = os.environ['S3_ARTIFACTS_NAME']
    
    # Load configuration file
    json_path = args.json_path
    CONF = utils.load_json(json_path)
    config = CONF['class_monai_dcm']

    # Extract configurations
    root_path = config['root_path']
    output_dir = config['output_dir']
    
    dir_to_copy = os.path.join(root_path, output_dir)
    
    # copy the directory and its content to the s3 bucket with the prefix 'public/models/' using boto3
    upload_directory(dir_to_copy, s3_name, 'public/models/')


def upload_directory(directory_path, bucket_name, prefix):
    """
    Upload a directory to an S3 bucket

    :param directory_path: Path of directory to upload
    :param bucket_name: Name of bucket to upload to
    :param destination: The destination prefix in the bucket
    """
    s3_client = boto3.client('s3')
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                s3_client.upload_fileobj(data, bucket_name, os.path.join(prefix, subdir.replace(directory_path, '').lstrip('/'), file))    


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    script_duration = datetime.datetime.now() - start_time
    print('Script duration: {}'.format(script_duration))