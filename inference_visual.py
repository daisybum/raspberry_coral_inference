import json
import os
import numpy as np
from PIL import Image
import time  # Time measurement module added

# GUI 없는 환경을 위한 matplotlib 설정
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경을 위한 백엔드 설정
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pycoral.utils import edgetpu
from pycoral.adapters import common, segment

# 0) Path setup
annotations_path = '/workspace/merged_all/_annotations.coco.json'
image_dir = '/workspace/merged_all'
model_path = 'model_quant_fixed_edgetpu.tflite'

# Folder to save segmentation visualization results (if needed)
output_dir = '/workspace/output_visual'
os.makedirs(output_dir, exist_ok=True)

# 1) Load TFLite model for Edge TPU and initialize interpreter
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

# 2) Model input size
input_width, input_height = common.input_size(interpreter)
print(f"Model input size: {input_width} x {input_height}")

# 3) Load COCO format JSON file
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

# 4) Define color palette and class names by class index with updated values
#  0: background, 1: dry, 2: humid, 3: slush, 4: snow, 5: wet
palette = np.array([
    [0, 0, 0],         # 0: background
    [113, 193, 255],   # 1: dry
    [255, 219, 158],   # 2: humid
    [125, 255, 238],   # 3: slush
    [235, 235, 235],   # 4: snow
    [255, 61, 61]      # 5: wet
], dtype=np.uint8)

class_names = ["background", "dry", "humid", "slush", "snow", "wet"]

# 5) Function for overlay
def blend_mask(original_np, mask_np, alpha=0.5):
    """
    original_np: (H, W, 3) uint8 original image array
    mask_np:     (H, W)     class index per pixel (0~5)
    alpha:       mask transparency (0~1)
    """
    overlay = original_np.copy()
    color_mask = palette[mask_np]  # (H, W, 3)
    # Can overlay only mask_np != 0 parts or all pixels according to desired rules
    # Here we only overlay mask_np != 0 (assuming 0 is background)
    mask_region = (mask_np != 0)
    overlay[mask_region] = (
        overlay[mask_region] * (1 - alpha) +
        color_mask[mask_region] * alpha
    ).astype(np.uint8)
    return overlay

# 6) Create legend patches (including label names for each patch)
legend_patches = []
for i, (r, g, b) in enumerate(palette):
    legend_color = (r/255, g/255, b/255)  # matplotlib uses 0~1 range colors
    # Use label parameter in mpatches.Patch to specify class name
    legend_patches.append(mpatches.Patch(color=legend_color, label=class_names[i]))

# 7) Iterate through images for inference and visualization
# Measure start time for entire loop
total_start_time = time.time()
total_images = len(coco_data['images'])
processed_images = 0

for image_info in coco_data['images']:
    # Start time measurement for each image
    img_start_time = time.time()
    
    file_name = image_info['file_name']
    orig_width = image_info['width']
    orig_height = image_info['height']

    img_path = os.path.join(image_dir, file_name)
    if not os.path.exists(img_path):
        print(f"[WARN] Image file not found: {img_path}")
        continue

    print(f"Processing: {img_path}")

    # (1) Load original image and resize
    img_pil = Image.open(img_path).convert('RGB')
    resized_img = img_pil.resize((input_width, input_height), resample=Image.LANCZOS)

    # (2) Segmentation inference
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    mask = segment.get_output(interpreter)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)

    # (3) Resize mask to original size
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    mask_pil = mask_pil.resize((orig_width, orig_height), resample=Image.NEAREST)
    mask_np = np.array(mask_pil)

    # (4) Create color mask
    color_mask_np = palette[mask_np]

    # (5) Create overlay image
    orig_np = np.array(img_pil)
    if (orig_np.shape[1] != orig_width) or (orig_np.shape[0] != orig_height):
        orig_np = np.array(img_pil.resize((orig_width, orig_height), resample=Image.LANCZOS))
    overlay_np = blend_mask(orig_np, mask_np, alpha=0.5)

    # (6) Matplotlib visualization (original, mask, overlay in 3 parts)
    # Set overall figure size wide to secure space for legend
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Segmentation Visualization - {file_name}", fontsize=16)

    # Subplot 1: Original image
    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # Subplot 2: Color mask
    axes[1].imshow(color_mask_np)
    axes[1].set_title("Segmentation Mask", fontsize=14)
    axes[1].axis("off")

    # Subplot 3: Overlay
    axes[2].imshow(overlay_np)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")

    # Place legend in center-right, each legend item displays label name
    legend = fig.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(0.92, 0.5),  # Secure right margin
        fontsize=12,
        title_fontsize=14
    )
    legend.get_frame().set_edgecolor("black")  # Set legend border to black (optional)

    # Adjust layout: Secure margins so subplots and legend don't overlap
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # plt.show()

    # (Optional) Save visualization results
    out_fig_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '_visual.png')
    fig.savefig(out_fig_path)
    plt.close(fig)  # Close figure to prevent memory leak
    
    # End time measurement for each image and output
    img_end_time = time.time()
    img_elapsed_time = img_end_time - img_start_time
    processed_images += 1
    
    print(f"Visualization saved: {out_fig_path}")
    print(f"Image processing time: {img_elapsed_time:.2f} seconds")
    print(f"Progress: {processed_images}/{total_images} ({processed_images/total_images*100:.1f}%)")
    print("-" * 50)

# End time measurement for entire loop and output
total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time
avg_time_per_image = total_elapsed_time / processed_images if processed_images > 0 else 0

print("\n" + "="*50)
print(f"Processing complete!")
print(f"Total images processed: {processed_images}")
print(f"Total processing time: {total_elapsed_time:.2f} seconds")
print(f"Average processing time per image: {avg_time_per_image:.2f} seconds")
print("="*50)
