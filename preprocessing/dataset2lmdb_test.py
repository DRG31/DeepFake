import os
import json
import cv2
import lmdb
import yaml
import argparse
from PIL import Image
import io
import numpy as np

def file_to_binary(file_path):
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        file_binary = data.tobytes()
    else:
        with open(file_path, 'rb') as f:
            file_binary = f.read()
    return file_binary

def lmdb_dataset(source_folder, lmdb_path, dataset_name, dataset_root_path, map_size):
    db = lmdb.open(lmdb_path, map_size=map_size)
    with db.begin(write=True) as txn:
        for root, dirs, files in os.walk(source_folder, followlinks=True):
            for file in files:
                image_path = os.path.join(root, file)
                # Get relative path from dataset root (datasets/rgb)
                relative_path = os.path.relpath(image_path, dataset_root_path)
                # Convert to POSIX path for consistency
                relative_path = relative_path.replace('\\', '/')
                key = relative_path.encode('utf-8')
                value = file_to_binary(image_path)
                txn.put(key, value)
    db.close()

def read_lmdb(lmdb_dir_path):
    env = lmdb.open(lmdb_dir_path)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            print(f"Key: {key.decode()}, Value length: {len(value)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size', type=int, default=25, required=True,
                        help='LMDB map size in GB')
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)['to_lmdb']

    dataset_name = config['dataset_name']['default']
    dataset_root_path = config['dataset_root_path']['default']
    output_lmdb_dir = config['output_lmdb_dir']['default']
    
    os.makedirs(output_lmdb_dir, exist_ok=True)
    
    dataset_dir_path = os.path.join(dataset_root_path, dataset_name)
    lmdb_path = os.path.join(output_lmdb_dir, f"{dataset_name}_lmdb")
    
    lmdb_dataset(
        source_folder=dataset_dir_path,
        lmdb_path=lmdb_path,
        dataset_name=dataset_name,
        dataset_root_path=dataset_root_path,  # Pass root path for relative keys
        map_size=args.dataset_size * 1024**3
    )
    
    # Validate the created LMDB
    read_lmdb(lmdb_path)