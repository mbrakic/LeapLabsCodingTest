import os 
import sys
from attack_v2 import generate_adversarial_image

# Please replace image path with your own image path
image_path = "n01440764_tench.JPEG"

# likewise with label
target_label = 'stingray'

generate_adversarial_image(image_path, target_label) 

