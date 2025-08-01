import torch 
import torch.nn as nn 
from torchvision import datasets, models, transforms 
import numpy as np 
import matplotlib.pyplot as plt
import os 
import random
import imagenette_setup 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Normalization statistics for ImageNet-trained models
imagenette_mean = [0.485, 0.456, 0.406]
imagenette_std = [0.229, 0.224, 0.225]

# let's first download resnet
def get_model():
    weights = models.ResNet18_Weights.DEFAULT 
    model = models.resnet18(weights=weights)
    model.to(device)
    model.eval()
    return model, weights.transforms() 

# should also denormalize the tensor when we want to look at it 
def denormalize(tensor):
    inv_norm = transforms.Normalize(
        mean = [-m/s for m,s in zip(imagenette_mean, imagenette_std)], 
        std = [1/s for s in imagenette_std]
    )
    return inv_norm(tensor).clamp(0,1) 

# write a function that runs it through the model and returns some confidence
# too
def predict_tensor(model, tensor, device, labels):
    model.eval() 
    input_tensor = tensor.unsqueeze(0).to(device) 
    with torch.no_grad():
        output = model(input_tensor) 
    probabilities = torch.nn.functional.softmax(output[0], dim=0) 
    top_prob, top_catid = torch.topk(probabilities, 1) 
    predicted_idx_str = str(top_catid.item()) 
    _, predicted_name = labels.get(predicted_idx_str, ("Unknown", "Unknown")) 
    return predicted_name.replace('_', ' '), top_prob.item() 

def plot_results(
        original_img, noise, adv_img, 
        original_pred, adv_pred, true_label
    ):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(f"PGD Attack on Imagenette\nTrue Label: {true_label}", fontsize=16)
    ax1.imshow(original_img)
    ax1.set_title(f"Original Image\nPrediction: {original_pred[0]} ({original_pred[1]:.2%})")
    ax1.axis('off')
    scaled_noise = (noise - noise.min()) / (noise.max() - noise.min())
    ax2.imshow(scaled_noise)
    ax2.set_title("Adversarial Noise (Scaled)")
    ax2.axis('off')
    ax3.imshow(adv_img)
    ax3.set_title(f"Adversarial Image\nPrediction: {adv_pred[0]} ({adv_pred[1]:.2%})")
    ax3.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    

if __name__=="__main__":
    dataset_idx_to_full_idx, full_imagenet_labels , idx_to_nid, imagenette_map = imagenette_setup.create_label_mapping()

    sample_idx = 5
    sample_nid = idx_to_nid[sample_idx] 
    sample_full_idx = dataset_idx_to_full_idx[sample_idx]    
    sample_name = imagenette_map[sample_nid]

    print(f"Example: Dataset index {sample_idx} ({sample_name}) -> Full ImageNet index {sample_full_idx}")

    model, preprocess_transforms = get_model()

    # let's get the validation dataset
    dataset_root = imagenette_setup.setup_imagenette_data()
    val_dir = os.path.join(dataset_root, 'val')
    val_dataset = datasets.ImageFolder(root=val_dir, transform=preprocess_transforms)

    # pick a random sample from imagenette
    idx = random.randint(0, len(val_dataset) - 1)
    # n.b. these are the 0-9 index
    image_tensor, label_idx = val_dataset[idx] 

    full_imagenet_idx = dataset_idx_to_full_idx[label_idx]
    label_tensor = torch.tensor([full_imagenet_idx]) 

    # 5. Verify original prediction
    true_label_name = imagenette_setup.imagenette_map[idx_to_nid[label_idx]]
    original_pred = predict_tensor(model, image_tensor, device, full_imagenet_labels)
    
    print(f"Selected a random image: '{true_label_name}'")
    print(f"Model's Initial Prediction: '{original_pred[0]}' (Confidence: {original_pred[1]:.2%})\n")

    original_img = denormalize(image_tensor).permute(1,2,0).numpy()
    noise = np.zeros_like(original_img)

    plot_results(original_img, noise, original_img, original_pred, original_pred, label_tensor)