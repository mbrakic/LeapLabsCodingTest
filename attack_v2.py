import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.to(device)
    model.eval()
    return model, weights

def get_imagenet_label_map(weights):
    labels = weights.meta["categories"]
    return {label.lower(): i for i, label in enumerate(labels)}

def load_and_preprocess_image(image_path, preprocess_transforms):
    input_image = Image.open(image_path).convert("RGB")
    return preprocess_transforms(input_image)

def denormalize(tensor):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    inv_norm = transforms.Normalize(
        mean=[-m/s for m, s in zip(imagenet_mean, imagenet_std)],
        std=[1/s for s in imagenet_std]
    )
    return inv_norm(tensor).clamp(0, 1)

def predict_tensor(model, tensor, labels):
    input_tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    predicted_name = labels[top_catid.item()]
    return predicted_name, top_prob.item()

def plot_results(original_img, noise, adv_img, original_pred, adv_pred, target_label):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    title = f"Targeted Adversarial Attack\nTarget Label: {target_label}"
    fig.suptitle(title, fontsize=16)

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

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()

def attack(
        model, image_tensor, target_label_idx, epsilon, alpha, num_iter
        ):
    original_tensor = image_tensor.clone().detach().to(device)
    adv_tensor = original_tensor.clone().detach().requires_grad_(True)
    target_label_tensor = torch.tensor([target_label_idx]).to(device)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_iter):
        output = model(adv_tensor.unsqueeze(0))
        loss = loss_fn(output, target_label_tensor)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_sign = adv_tensor.grad.detach().sign()
            adv_tensor.sub_(grad_sign, alpha=alpha)
            perturbation = torch.clamp(adv_tensor - original_tensor, min=-epsilon, max=epsilon)
            adv_tensor = original_tensor + perturbation
        
        adv_tensor.requires_grad_(True)

    return adv_tensor.detach()

def generate_adversarial_image(
        image_path, target_label_name, epsilon=8/255, alpha=2/255, num_iter=50, plot=True
        ):
    """Main function to load an image, attack it, and show results."""

    # get weights and model 
    model, weights = get_model() 
    preprocess_transforms = weights.transforms()

    # get label index for requested target label
    full_imagenet_labels = weights.meta["categories"]
    label_to_idx_map = get_imagenet_label_map(weights)
    target_label_idx = label_to_idx_map[target_label_name.lower()]
    target_label_display = full_imagenet_labels[target_label_idx]

    # load in the image and make some initial prediction
    image_tensor = load_and_preprocess_image(image_path, preprocess_transforms)
    original_pred = predict_tensor(model, image_tensor, full_imagenet_labels)

    print(f"Attacking image: '{image_path}'")
    print(f"Targeting class: '{target_label_display}'")
    print(f"Model's Initial Prediction: '{original_pred[0]}' (Confidence: {original_pred[1]:.2%})\n")

    adv_tensor = attack(model, image_tensor, target_label_idx, epsilon, alpha, num_iter)

    adv_pred = predict_tensor(model, adv_tensor, full_imagenet_labels)
    print(f"Model's Prediction After Attack: '{adv_pred[0]}' (Confidence: {adv_pred[1]:.2%})")

    if plot:
        original_img = denormalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
        adv_img = denormalize(adv_tensor.cpu()).permute(1, 2, 0).numpy()
        noise = adv_img - original_img
        plot_results(original_img, noise, adv_img, original_pred, adv_pred, target_label_display)

    return adv_tensor