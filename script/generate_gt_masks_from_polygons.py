"""
Generate ground truth pixel masks from JSON polygon annotations.

This script reads polygon annotations (in JSON format) corresponding to RGB images,
converts them into raster masks, and saves them as both .npy arrays and visualized .png images.

- Input:  RGB images (.jpg) and polygon annotations (.json)
- Output: Ground truth masks (.npy) and preview images (.png)


"""

import os
import json
from shapely.geometry import Polygon
from rasterio.features import rasterize
from PIL import Image
import numpy as np

def batch_generate_gt_masks(image_dir, json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.endswith(".jpg"):
            continue

        basename = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, filename)
        json_path = os.path.join(json_dir, f"{basename}.json")

        if not os.path.exists(json_path):
            print(f"JSON file not found: {basename}.json — skipping.")
            continue

        # Load image to get dimensions
        img = Image.open(image_path)
        width, height = img.size

        # Load polygon annotations from JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        polygons = [Polygon(p) for p in raw_data if len(p) >= 3]
        polygons_with_ids = [(poly, i + 1) for i, poly in enumerate(polygons)]

        # Rasterize the polygons into a pixel mask
        gt_mask = rasterize(
            polygons_with_ids,
            out_shape=(height, width),
            fill=0,
            dtype='uint8',
            all_touched=True
        )

        # Save as .npy and .png (for visualization)
        npy_path = os.path.join(output_dir, f"{basename}.npy")
        png_path = os.path.join(output_dir, f"{basename}.png")

        np.save(npy_path, gt_mask)
        Image.fromarray(gt_mask * 50).save(png_path)  # Multiply for visual separation of classes

        print(f"Processed: {basename}")

    print(f"\n All ground truth masks saved to: {output_dir}")


# Change these paths to match your local setup
image_dir = r"E:\2025_May_Data_Dblue\cities\test\rgb"
json_dir = r"E:\2025_May_Data_Dblue\cities\test\JSON"
output_dir = r"E:\2025_May_Data_After_Dblue_MRF\RoofVec\gt_pixel_mask"

batch_generate_gt_masks(image_dir, json_dir, output_dir)
