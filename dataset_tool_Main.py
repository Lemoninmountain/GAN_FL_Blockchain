import os
import json
from pathlib import Path
import shutil
from PIL import Image

data_dirs = r'C:\Users\lemon\Desktop\FL-GAN_COVID-main\8GAN\Kvasir\dyed-lifted-polyps\dyed-lifted-polyps',

output_dir = r'C:\Users\lemon\Desktop\FL-GAN_COVID-main\8GAN\dyed-lifted-polyps'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

labels = []

def preprocess_and_save_image(img_path, output_img_path):
    with Image.open(img_path) as img:
        # Ensure the image is in RGB mode
        img = img.convert('RGB')
        width, height = img.size
        min_side = min(width, height)
        left = (width - min_side) / 2
        top = (height - min_side) / 2
        right = (width + min_side) / 2
        bottom = (height + min_side) / 2
        img = img.crop((left, top, right, bottom))
        img = img.resize((512, 512), Image.LANCZOS)  # Resize to 32×32 using LANCZOS filter
        img.save(output_img_path)

for class_idx, class_dir in enumerate(data_dirs):
    class_name = os.path.basename(class_dir)
    print(f'Processing class: {class_name}')
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        if os.path.isfile(img_path):
            output_img_path = os.path.join(output_dir, f'{class_name}_{img_name}')
            preprocess_and_save_image(img_path, output_img_path)
            labels.append([f'{class_name}_{img_name}', class_idx])

# Save labels to dataset.json
with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
    json.dump({'labels': labels}, f)

print(f"Dataset prepared at {output_dir}")

# Now you can run the conversion script
os.system(f'python dataset_tool.py --source {output_dir} --dest {output_dir}.zip --transform=center-crop --width=512 --height=512')
