import boto3
import zipfile
import io
import pandas as pd
import os
import datetime
import argparse

s3 = boto3.client('s3')
bucket_name = 'dicomzipprocessingstack-datalakebucket8c2bd199-1m5uekvmq2uw'
csv_files = [
    'local_test/rh8-dev02-0.0.0_tf/out/rh8-dev02-0.0.0_tf_ax_DenseNet121_simple/split/db.csv', 
    'local_test/rh8-dev02-0.0.0_tf/out/rh8-dev02-0.0.0_tf_ax_DenseNet121_simple/split_for_predict/db.csv',
    'local_test/rh8-dev02-0.0.0_tf/out/rh8-dev02-0.0.0_tf_sag_DenseNet121_simple/split/db.csv', 
    'local_test/rh8-dev02-0.0.0_tf/out/rh8-dev02-0.0.0_tf_sag_DenseNet121_simple/split_for_predict/db.csv',
    'local_test/rh8-dev02-0.0.0_tf/out/rh8-dev02-0.0.0_tf_cor_DenseNet121_simple/split/db.csv', 
    'local_test/rh8-dev02-0.0.0_tf/out/rh8-dev02-0.0.0_tf_cor_DenseNet121_simple/split_for_predict/db.csv'
]

def main():
    
    parser = argparse.ArgumentParser(description="Dump DICOM files locally.")
    parser.add_argument("--dump_dir", required=True, help="Enter dump directory path.")
    
    args = parser.parse_args()
    
    local_directory = args.dump_dir
    
    def extract_and_save_file(zip_key, file_path_in_zip, local_directory):
        obj = s3.get_object(Bucket=bucket_name, Key=zip_key)
        with io.BytesIO(obj['Body'].read()) as tf:
            tf.seek(0)
            with zipfile.ZipFile(tf, mode='r') as zipf:
                if file_path_in_zip in zipf.namelist():
                    local_file_path = os.path.join(local_directory, file_path_in_zip)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    zipf.extract(file_path_in_zip, local_directory)

    for csv in csv_files:
        # Load CSV from disk
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print('Error: {}'.format(e))
            return
        # Process each row in the CSV
        for _, row in df.iterrows():
            zip_file_key = row['OLEA_ZIP_FILE_KEY']
            file_path_in_zip = "/".join(row['OLEA_INSTANCE_PATH'].split("/")[-3:])
            if not file_path_in_zip.endswith('.dcm'):
                file_path_in_zip += '.dcm'
            # Add dicom/ prefix to the zip file key
            zip_file_key = 'dicom/{}'.format(zip_file_key)
            # Check if file exists locally
            local_file_path = os.path.join(local_directory, file_path_in_zip)
            if os.path.isfile(local_file_path):
                print('File already exists: {}'.format(local_file_path))
                continue
            # Extract file from zip and save locally
            extract_and_save_file(zip_file_key, file_path_in_zip, local_directory)
            # Print progress dump compared to df.length in % rounded to 1 decimals
            # Print progress using tqdm

            print('Progress: {}%'.format(round(_ / df.shape[0] * 100, 1)))
        
        print('>>>> Finished dumping files from CSV: {}'.format(csv))
            

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    main()
    print('Duration: {}'.format(datetime.datetime.now() - start_time))
    