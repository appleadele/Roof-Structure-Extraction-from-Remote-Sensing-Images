# prepare_for_mrf.py

import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import geopandas as gpd
from rasterio.features import rasterize
import networkx as nx
from tqdm import tqdm
import random
import time
import pickle

def load_model(model_path, device=None, num_classes=2):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

# ------------------ GCO------------------

def generate_instance_level_pixel_prob_map(soft_masks, scores):
    N, H, W = soft_masks.shape
    weighted_soft_masks = soft_masks * scores[:, np.newaxis, np.newaxis]
    all_pixel_probs = np.moveaxis(weighted_soft_masks, 0, -1)
    bg_prob = 1 - np.max(all_pixel_probs, axis=-1)
    return np.concatenate([bg_prob[..., np.newaxis], all_pixel_probs], axis=-1)

def compute_polygon_probabilities(gpkg_path, pixel_prob_map):
    gdf = gpd.read_file(gpkg_path)
    H, W, C = pixel_prob_map.shape
    polygon_probs = np.zeros((len(gdf), C))
    for i, poly in enumerate(gdf.geometry):
        mask = rasterize([(poly, 1)], out_shape=(H, W), fill=0, all_touched=True)
        for j in range(C):
            polygon_probs[i, j] = np.mean(pixel_prob_map[..., j][mask == 1])
    return gdf, polygon_probs

def build_adjacency_graph(gdf, method="edge", weight_mode="constant", smoothness_config=None):
    neighbor_check = lambda a, b: a.intersection(b).geom_type == "LineString" if method == "edge" else a.touches(b)
    G = nx.Graph()
    for i in range(len(gdf)):
        G.add_node(i)
    sindex = gdf.sindex
    for i, geom in enumerate(gdf.geometry):
        for j in sindex.intersection(geom.bounds):
            if i >= j:
                continue
            if neighbor_check(geom, gdf.geometry.iloc[j]):
                inter = geom.intersection(gdf.geometry.iloc[j])
                if inter.is_valid and not inter.is_empty:
                    true_length = inter.length
                    weight = calculate_smoothness_weight(true_length, **smoothness_config) if weight_mode == "length" else 1
                    G.add_edge(i, j, weight=weight)
    return G

def calculate_smoothness_weight(true_length, max_length=80, method="sqrt", scale=5, offset=1):
    if method == "sqrt":
        normalized = np.sqrt(true_length) / np.sqrt(max_length)
    elif method == "log":
        normalized = np.log1p(true_length) / np.log1p(max_length)
    elif method == "linear":
        normalized = true_length / max_length
    else:
        raise ValueError(f"Unsupported method: {method}")
    return normalized * scale + offset

def prepare_unary(polygon_probs, scale=10):
    costs = (1.0 - polygon_probs) * scale
    return costs


if __name__ == "__main__":
    # Please update the path below to match your local file system
    img_dir = r"/scratch/hycheng/Thesis/2025_May_Data_Dblue/RoofVec/test/rgb"
    gpkg_dir = r"/scratch/hycheng/Thesis/2025_May_Data_After_Dblue_MRF/RoofVec/Gao_GT/polygon_geopackages"
    model_path = r"/scratch/hycheng/Thesis/2025_May_Data_Dblue/RoofVec/slurm/RoofVec_resnet50/best_model_epoch59_iou0.9415_RoofVec.pth"
    out_dir = r"/scratch/hycheng/Thesis/2025_May_Data_After_Dblue_MRF/RoofVec/precomputed"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    images = sorted(images)  


    model = load_model(model_path, device=device, num_classes=2)

    config = {
        "device": device,
        "unary_scale": 10,
        "adj_method": "edge",
        "weight_mode": "length",
        "smoothness_config": {
            "max_length": 80,
            "method": "sqrt",
            "scale": 5,
            "offset": 1
        }
    }

    for image_name in tqdm(images, desc="Precomputing"):
        basename = os.path.splitext(image_name)[0]
        image_path = os.path.join(img_dir, image_name)
        gpkg_path = os.path.join(gpkg_dir, f"{basename}_rooflines_polygon.gpkg")

        if not (os.path.exists(image_path) and os.path.exists(gpkg_path)):
            print(f"Missing files, skipping {basename}")
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)[0]

        soft_masks = output["masks"].squeeze(1).cpu().numpy()
        if soft_masks is None or len(soft_masks) == 0:
            print(f"No detection results, skipping {basename}")
            continue

        scores = output["scores"].cpu().numpy()
        pixel_prob_map = generate_instance_level_pixel_prob_map(soft_masks, scores)

        gdf, polygon_probs = compute_polygon_probabilities(gpkg_path, pixel_prob_map)
        unary = prepare_unary(polygon_probs, scale=config["unary_scale"])

        graph = build_adjacency_graph(
            gdf,
            method=config["adj_method"],
            weight_mode=config["weight_mode"],
            smoothness_config=config["smoothness_config"]
        )

        # save unary & graph
        np.save(os.path.join(out_dir, f"{basename}_unary.npy"), unary)
        with open(os.path.join(out_dir, f"{basename}_graph.pkl"), "wb") as f:
            pickle.dump((graph, gdf, image.size), f)

