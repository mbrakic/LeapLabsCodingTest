import numpy as np
import torch 
from torchvision import datasets, models, transforms
from torch.hub import download_url_to_file
import os 
import json 
import tarfile 

# imagenette is a nice subset of imagenet, it's not big and therefore fast to
# use. should be good for this 

# most of this code is adapted from previous code in knowing how to use the imagenette repo

# Normalization statistics for ImageNet-trained models
imagenette_mean = [0.485, 0.456, 0.406]
imagenette_std = [0.229, 0.224, 0.225]

imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
labels_url = "https://s3.amazonaws.com/deep-learning-models/image-models/image-models/imagenet_class_index.json"
data_dir = "imagenette_data"

# need to map from keys to nice readable names
imagenette_map = {
    'n01440764': 'tench', 'n02102040': 'English springer', 'n02979186': 'cassette player',
    'n03000684': 'chain saw', 'n03028079': 'church', 'n03394916': 'French horn',
    'n03417042': 'garbage truck', 'n03425413': 'gas pump', 'n03445777': 'golf ball',
    'n03888257': 'parachute'
}

# download the data
def setup_imagenette_data():
    os.makedirs(data_dir, exist_ok=True) 
    dataset_path = os.path.join(data_dir, "imagenette2-160")

    if os.path.exists(dataset_path):
        # no need to re-download
        return dataset_path

    print("Downloading dataset...")
    tgz_path = os.path.join(data_dir, "imagenette.tgz")
    download_url_to_file(imagenette_url, tgz_path, progress = True)
    
    print("Extracting dataset...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    os.remove(tgz_path)
    
    print(f"Dataset ready at '{dataset_path}'")
    return dataset_path
    

# there's an annoying feature of imagenette because it uses the indices 0-9 but
# these do not correspond with the ImageNet indices 0-9. we need to map them
# between each other. 
def create_label_mapping():

    dataset_root = setup_imagenette_data()

    # download the imagenet labels
    labels_path = os.path.join(dataset_root, "imagenet_class_index.json") 
    # check to see if it already exists and download if not
    # if not os.path.exists(labels_path):
    #     print("downloading imagenet labels...")
    #     download_url_to_file(labels_url, labels_path)

    with open(labels_path) as f:
        full_imagenet_labels = json.load(f) 

    # now we make the mapping between the imagenette index to the imagenet index

    # create a mapping from ImageNet ID ('n...') to its full 0-999 index
    id_to_full_idx = {v[0]: int(k) for k, v in full_imagenet_labels.items()}

    # we need to make a temp_dataset to get the 0-9 index.
    val_dir = os.path.join(dataset_root, 'val') 

    temp_dataset = datasets.ImageFolder(root=val_dir, transform=transforms.ToTensor())

    dataset_idx_to_full_idx = {
        dataset_idx: id_to_full_idx[nid]
        for nid, dataset_idx in temp_dataset.class_to_idx.items()
    }
    
    # create a reverse map for getting the name from the dataset index
    idx_to_nid = {v: k for k, v in temp_dataset.class_to_idx.items()}

    return dataset_idx_to_full_idx, full_imagenet_labels, idx_to_nid
    

# let's run it and check
if __name__=="__main__":
    dataset_idx_to_full_idx, _ , idx_to_nid = create_label_mapping()

    sample_idx = 3 
    sample_nid = idx_to_nid[sample_idx] 
    sample_full_idx = dataset_idx_to_full_idx[sample_idx]    
    sample_name = imagenette_map[sample_nid]

    print(f"Example: Dataset index {sample_idx} ({sample_name}) -> Full ImageNet index {sample_full_idx}")



    

