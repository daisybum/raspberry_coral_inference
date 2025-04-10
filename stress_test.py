import os
import time
import logging
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycoral.utils import edgetpu
from pycoral.adapters import common, segment

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Basic settings
model_path = 'model_quant_fixed_edgetpu.tflite'
capture_dir = '/media/pi/ESD-USB/captured_images'
output_dir = '/media/pi/ESD-USB/output_visual'
os.makedirs(capture_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Function to get CPU temperature
def get_cpu_temperature():
    try:
        temp_str = os.popen("vcgencmd measure_temp").readline()
        # Parsing string of form: temp=45.0'C
        temp_value = float(temp_str.replace("temp=", "").replace("'C", "").strip())
        return temp_value
    except Exception as e:
        logging.error("Error measuring CPU temperature: " + str(e))
        return None

# Model loading and Edge TPU interpreter initialization
try:
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_width, input_height = common.input_size(interpreter)
except Exception as e:
    logging.error("Error loading model or initializing interpreter: " + str(e))
    raise

# Color palette for segmentation visualization
palette = np.array([
    [0, 0, 0],
    [113, 193, 255],
    [255, 219, 158],
    [125, 255, 238],
    [235, 235, 235],
    [255, 61, 61]
], dtype=np.uint8)

# Function for visualization and saving the output
def visualize_and_save(img_pil, mask, img_filename):
    try:
        orig_np = np.array(img_pil)
        mask_np = np.array(Image.fromarray(mask.astype(np.uint8)).resize(img_pil.size, resample=Image.NEAREST))
        color_mask_np = palette[mask_np]

        overlay = orig_np.copy()
        alpha = 0.5
        mask_region = (mask_np != 0)
        overlay[mask_region] = (overlay[mask_region] * (1 - alpha) + color_mask_np[mask_region] * alpha).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(orig_np)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(color_mask_np)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        visual_path = os.path.join(output_dir, f'{img_filename}_visual.png')
        plt.savefig(visual_path)
        plt.close()
        logging.info(f"Visualization saved: {visual_path}")
    except Exception as e:
        logging.error("Visualization and saving error: " + str(e))

# Image capture and inference loop (for stress testing)
while True:
    try:
        # Log CPU temperature
        cpu_temp = get_cpu_temperature()
        if cpu_temp is not None:
            logging.info(f"Current CPU temperature: {cpu_temp:.2f}°C")
        else:
            logging.warning("Unable to measure CPU temperature.")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"capture_{timestamp}"
        img_path = os.path.join(capture_dir, f"{img_filename}.jpg")

        capture_cmd = f"libcamera-still -n -o {img_path} --width 1640 --height 1232"
        logging.info(f"Starting image capture: {img_filename}")
        os.system(capture_cmd)

        if os.path.exists(img_path):
            start_time = time.time()
            
            # Load and resize image
            try:
                img_pil = Image.open(img_path).convert('RGB').resize((input_width, input_height), resample=Image.LANCZOS)
            except Exception as e:
                logging.error("Error loading image: " + str(e))
                continue

            common.set_input(interpreter, img_pil)

            try:
                interpreter.invoke()
            except Exception as e:
                logging.error("Error invoking model inference: " + str(e))
                continue

            try:
                mask = segment.get_output(interpreter)
                if mask.ndim == 3:
                    mask = np.argmax(mask, axis=-1)
                mask = np.array(mask.tolist(), dtype=np.uint8)
            except Exception as e:
                logging.error("Error processing inference output: " + str(e))
                continue

            visualize_and_save(img_pil.resize((1640, 1232), Image.LANCZOS), mask, img_filename)

            elapsed_time = time.time() - start_time
            logging.info(f"{img_filename} processing completed - Elapsed time: {elapsed_time:.2f}s")
        else:
            logging.warning(f"{img_filename} capture failed")

    except Exception as e:
        logging.error("System error occurred: " + str(e))

    logging.info("Waiting 30 seconds before next capture...")
    time.sleep(30)
