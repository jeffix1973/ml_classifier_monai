import sys
import os
import json
import pandas as pd
import numpy as np
import pydicom
from PIL.Image import fromarray


def export_image(path, export_dir):
    try:
        ds = pydicom.dcmread(path)
        StudyInstanceUID = ds.StudyInstanceUID
        SeriesInstanceUID = ds.SeriesInstanceUID
        test_and_create_dir(os.path.join(export_dir, StudyInstanceUID, SeriesInstanceUID))
        im = ds.pixel_array.astype(float)
        rescaled_image = (np.maximum(im,0)/im.max())*255
        final_image = np.uint8(rescaled_image)
        final_image = fromarray(final_image)
        final_image.save(os.path.join(export_dir, StudyInstanceUID, SeriesInstanceUID, 'middleImg.jpg'))
    except  Exception as e:
        print(e)


def get_paths_and_chunk(cfg):
    # Collect paths
    try:
        df = pd.read_csv(cfg.csv_path)
        paths = df['OLEA_INSTANCE_PATH'].tolist()
        gt = [round(x) for x in df['label'].tolist()]
    except Exception as e:
        print(e)
        sys.exit(0)

    # Chunk list / params.batch_size
    def chunk (list, x):
        return [list[i:i+x] for i in range(0, len(list), x)]

    divider = round(len(paths)//int(cfg.params.batch_size))

    if divider > 1:
        chunked_paths = chunk(paths, cfg.params.batch_size)
        chunked_gt = chunk(gt, cfg.params.batch_size)
    else:
        chunked_paths = [paths]
        chunked_gt = [gt]
    
    return chunked_paths, chunked_gt


def test_and_create_dir (outdir):
    # Check if outdir directory exists
    isExist = os.path.exists(outdir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(outdir)
        return 'New directory : ' + outdir + ' has been created'
    else:
        return outdir + ' allready exists...'


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(">>> Directory created: ", path)
    else:
        print(">>> Directory already exists: ", path)
        

def load_json(path):
    print('>>>> Loading', path)
    try:
        f = open(path, "r")
    except:
        print('File does not exist. The program has been stopped...')
        sys.exit(0)
    
    # STEP 2 : Collect JSON variables
    file = json.load(f)
    VAR = file['class_monai_dcm']
    VAR2 = file['bp_detection_inference_rest_api']
    print('>>>> Variables have been successfuly loaded...')
    
    return VAR, VAR2