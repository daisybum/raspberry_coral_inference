import json
import os
import time  # Added for timing
import numpy as np
from PIL import Image
from pycoral.utils import edgetpu
from pycoral.adapters import common, segment

# 0) Path settings
annotations_path = '/workspace/moveawheel_data/COCO/test_without_street.json'
image_dir = '/workspace/moveawheel_data/images'
model_path = '../models/model_quant_fixed_edgetpu.tflite'

# Load COCO format JSON file
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

# 1) Load TFLite model for Edge TPU and initialize interpreter
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

# 2) Check model input size
input_width, input_height = common.input_size(interpreter)
#print(f"Model input size: {input_width} x {input_height}")

# 3) Iterate through images in COCO data and perform inference
for image_info in coco_data['images']:
    start_time = time.time()  # Start timing
    file_name = image_info['file_name']
    orig_width = image_info['width']
    orig_height = image_info['height']
    img_path = os.path.join(image_dir, file_name)
    
    if not os.path.exists(img_path):
        print(f"[WARN] Image file not found: {img_path}")
        continue

    #print(f"Processing: {img_path}")
    
    # (1) Load and preprocess image (resize to model input size)
    img_pil = Image.open(img_path).convert('RGB')
    resized_img = img_pil.resize((input_width, input_height), resample=Image.LANCZOS)
    
    # (2) Perform inference with timing
    common.set_input(interpreter, resized_img)
    interpreter.invoke()
    mask = segment.get_output(interpreter)
    processing_time = time.time() - start_time  # Calculate inference time
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)
    
    # (3) Resize result mask to original image size (optional)
    # mask_pil = Image.fromarray(mask.astype(np.uint8)).resize((orig_width, orig_height), resample=Image.NEAREST)
    # mask_np = np.array(mask_pil)
    
    print(f"{file_name} inference completed in {processing_time:.4f} seconds")
