import os
import json
import cv2
import lmdb
import yaml
import numpy as np

def file_to_binary(file_path):
    """Convert image or numpy file to binary data."""
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        file_binary = data.tobytes()
    else:
        with open(file_path, 'rb') as f:
            file_binary = f.read()
    return file_binary

def create_lmdb_dataset(source_folder, lmdb_path, dataset_name, map_size):
    """Create LMDB dataset."""
    db = lmdb.open(lmdb_path, map_size=map_size)
    with db.begin(write=True) as txn:
        for root, dirs, files in os.walk(source_folder, followlinks=True):
            if 'video' in root:
                continue
            for file in files:
                image_path = os.path.join(root, file)
                relative_path = f"{dataset_name}/" + os.path.relpath(image_path, source_folder)
                key = relative_path.encode('utf-8')
                value = file_to_binary(image_path)
                txn.put(key, value)
                print(f"Written {key.decode('utf-8')} to LMDB")  # Log the written key
    db.close()
    print(f"LMDB dataset created at {lmdb_path}")

def read_lmdb(lmdb_dir_path):
    """Validate the key and value in the generated LMDB."""
    env = lmdb.open(lmdb_dir_path)
    with env.begin(write=False) as txn:
        # List all keys in the LMDB for debugging
        print("Listing all keys in LMDB:")
        cursor = txn.cursor()
        for key, _ in cursor:
            print(key.decode('utf-8'))  # Decode bytes to string

        # Example key for validation (update this key based on your dataset structure)
        key='npy_test\\000_003\\000.npy'
        binary = txn.get(key.encode())
        if binary is not None:
            data = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
            print("Data loaded successfully:", data)
        else:
            print("Key not found in LMDB.")

if __name__ == '__main__':
    # Load parameters from config.yaml
    yaml_path = 'config.yaml'
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)
        exit(1)

    config = config['to_lmdb']
    dataset_name = config['dataset_name']['default']
    dataset_root_path = config['dataset_root_path']['default']
    output_lmdb_dir = config['output_lmdb_dir']['default']
    os.makedirs(output_lmdb_dir, exist_ok=True)
    
    dataset_dir_path = f"{dataset_root_path}/{dataset_name}"
    lmdb_path = f"{output_lmdb_dir}/{dataset_name}_lmdb"
    
    # Create LMDB dataset
    create_lmdb_dataset(dataset_dir_path, lmdb_path, dataset_name, map_size=25 * 1024 * 1024 * 1024)
    
    # Optionally read and validate the LMDB
    read_lmdb(lmdb_path)
