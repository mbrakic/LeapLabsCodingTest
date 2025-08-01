import torch
import numpy as np 
import os
from torchvision import datasets, models, transforms 
import random 
import imagenette_setup 
import attack 

# going to try to collect all these together in a few useful functions 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def attack_image(
    image_tensor, label_idx, model, preprocess_transforms, device, epsilon, alpha, plot = False
):

    dataset_idx_to_full_idx, full_imagenet_labels , idx_to_nid, imagenette_map = imagenette_setup.create_label_mapping()

    full_imagenet_idx = dataset_idx_to_full_idx[label_idx]
    label_tensor = torch.tensor([full_imagenet_idx]) 

    # 5. Verify original prediction
    true_label_name = imagenette_setup.imagenette_map[idx_to_nid[label_idx]]
    original_pred = attack.predict_tensor(model, image_tensor, device, full_imagenet_labels)
    
    print(f"Selected a random image: '{true_label_name}'")
    print(f"Model's Initial Prediction: '{original_pred[0]}' (Confidence: {original_pred[1]:.2%})\n")

    original_img = attack.denormalize(image_tensor).permute(1,2,0).numpy()
    
    epsilon = 8/255
    alpha = 2/255 
    num_iter = 40 

    adv_tensor = attack.attack(model, image_tensor, label_tensor, epsilon, alpha, num_iter, device)

    adv_pred = attack.predict_tensor(model, adv_tensor, device, full_imagenet_labels)
    adv_img = attack.denormalize(adv_tensor).permute(1,2,0).numpy() 
    noise = adv_img - original_img 

    if plot:
        attack.plot_results(original_img, noise, adv_img, original_pred, adv_pred, true_label_name)

    return int(adv_pred==true_label_name)

# go over as many as you like and see how well they work
def success_rate(num_images, model, preprocess_transforms, device, epsilon, alpha):

    # let's get the validation dataset
    dataset_root = imagenette_setup.setup_imagenette_data()
    val_dir = os.path.join(dataset_root, 'val')
    val_dataset = datasets.ImageFolder(root=val_dir, transform=preprocess_transforms)

    count = 0

    for i in range(num_images):
        idx = random.randint(0, len(val_dataset) - 1) 
        image_tensor, label_idx = val_dataset[idx] 

        success = attack_image(image_tensor, label_idx, model, preprocess_transforms, device, epsilon, alpha)

        count+=success

    print(f"Attack worked at a rate of {100*(1 - count/num_images):.2%} over {num_images} samples")


if __name__=="__main__":

    # get model and transform
    model, preprocess_transforms = attack.get_model()

    # let's get the validation dataset
    dataset_root = imagenette_setup.setup_imagenette_data()
    val_dir = os.path.join(dataset_root, 'val')
    val_dataset = datasets.ImageFolder(root=val_dir, transform=preprocess_transforms)

    # pick a random sample from imagenette to plot
    idx = random.randint(0, len(val_dataset) - 1)

    # n.b. these are the 0-9 index
    image_tensor, label_idx = val_dataset[idx] 

    # parameters
    epsilon = 8/255 
    alpha = 2/255

    # attack and plot
    attack_image(image_tensor, label_idx, model, preprocess_transforms, device, epsilon, alpha)

    # look at attack success_rate
    success_rate(5, model, preprocess_transforms, device, epsilon, alpha)

