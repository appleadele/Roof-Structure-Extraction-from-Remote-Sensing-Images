# run_mrf_and_evaluate.py

import os
import pickle
import numpy as np
from PIL import Image
import geopandas as gpd
from rasterio.features import rasterize
import networkx as nx
import gco
from tqdm import tqdm
import random
import time

def run_gco_labeling(unary_terms, graph, smoothness_cost=1.0):
    num_nodes, num_labels = unary_terms.shape
    unary = np.ascontiguousarray(unary_terms.astype(np.float32))
    pairwise = smoothness_cost * (np.ones((num_labels, num_labels)) - np.eye(num_labels))

    edges = []
    edge_weights = []
    for u, v, data in graph.edges(data=True):
        edges.append((u, v))
        edge_weights.append(float(data.get("weight", 1.0)))

    edges = np.array(edges, dtype=np.int32)
    edge_weights = np.array(edge_weights)

    return gco.cut_general_graph(edges, edge_weights, unary, pairwise)

def rasterize_prediction(gdf, labels, height, width):
    return rasterize(
        [(geom, int(label)) for geom, label in zip(gdf.geometry, labels)],
        out_shape=(height, width), fill=0, dtype='uint8', all_touched=True
    )

def evaluate_iou_metrics(gt_mask, pred_mask, iou_threshold=0.5):
    gt_ids = sorted(set(np.unique(gt_mask)) - {0})
    pred_ids = sorted(set(np.unique(pred_mask)) - {0})
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))

    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return {
            "TP": 0, "FP": len(pred_ids), "FN": len(gt_ids),
            "Precision": 0.0, "Recall": 0.0, "Mean_IoU": 0.0
        }

    for i, gt_id in enumerate(gt_ids):
        gt_bin = (gt_mask == gt_id)
        for j, pred_id in enumerate(pred_ids):
            pred_bin = (pred_mask == pred_id)
            inter = np.logical_and(gt_bin, pred_bin).sum()
            union = np.logical_or(gt_bin, pred_bin).sum()
            iou_matrix[i, j] = inter / union if union > 0 else 0

    matched_gt, matched_pred, iou_list = set(), set(), []
    for i, gt_id in enumerate(gt_ids):
        j = np.argmax(iou_matrix[i])
        if iou_matrix[i, j] >= iou_threshold:
            pred_id = pred_ids[j]
            if pred_id not in matched_pred:
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)
                iou_list.append(iou_matrix[i, j])

    TP, FP, FN = len(iou_list), len(pred_ids) - len(iou_list), len(gt_ids) - len(iou_list)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    mean_iou = np.mean(iou_list) if iou_list else 0

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": precision, "Recall": recall,
        "Mean_IoU": mean_iou
    }

def fix_unary_with_neighbors(unary, graph):

    """
    Fix NaN or invalid unary values by averaging valid neighboring polygon values.

    This is necessary because some polygons may be too small to contain any rasterized pixels,
    which results in NaN or invalid values in the unary array. To avoid errors during optimization,
    we replace those invalid rows with the average of their valid neighbors' unaries.
    If no valid neighbors exist, we assign uniform values.

    """

    unary_fixed = unary.copy()
    for i in range(unary.shape[0]):
        if not np.isfinite(unary[i]).all():
            neighbors = list(graph.neighbors(i))
            valid_neighbors = [n for n in neighbors if np.isfinite(unary[n]).all()]
            if valid_neighbors:
                unary_fixed[i] = np.mean(unary[valid_neighbors], axis=0)
            else:
                unary_fixed[i] = np.ones(unary.shape[1]) * 1.0  
    return unary_fixed

def compute_smoothness_statistics(graph, labels, smoothness_cost=1):
    violated_edges = 0
    total_edges = graph.number_of_edges()
    for u, v in graph.edges():
        if labels[u] != labels[v]:
            violated_edges += 1
    violation_ratio = violated_edges / total_edges if total_edges > 0 else 0
    return {
        "Violation_Ratio": violation_ratio
    }


if __name__ == "__main__":

    # Please update the paths below to match your local file system
    precomputed_dir = r"E:\2025_May_Data_After_Dblue_MRF\RoofVec\precomputed_unary10"
    gt_dir = r"E:\2025_May_Data_After_Dblue_MRF\RoofVec\Gao_GT\gt_pixel_mask"
    output_txt = os.path.join(precomputed_dir, "mrf_evaluation_summary_RoofVec.txt")

    # Smoothness cost values to evaluate (from coarse to fine)
    smoothness_costs = [0.001, 0.1, 1, 10, 100]
    unary_scale = 10  # Keep consistent with earlier pipeline steps

    results = []

    # Get all precomputed unary files
    files = [f for f in os.listdir(precomputed_dir) if f.endswith("_unary.npy")]
    
    for cost in smoothness_costs:
        print(f"Running MRF with smoothness cost = {cost}")
        for unary_file in tqdm(files, desc=f"Label Cost {cost}"):
            basename = unary_file.replace("_unary.npy", "")
            unary_path = os.path.join(precomputed_dir, unary_file)
            graph_path = os.path.join(precomputed_dir, f"{basename}_graph.pkl")
            gt_path = os.path.join(gt_dir, f"{basename}.npy")

            # Skip if necessary files are missing
            if not os.path.exists(graph_path) or not os.path.exists(gt_path):
                continue

            # Load unary and graph
            unary = np.load(unary_path)
            with open(graph_path, "rb") as f:
                graph, gdf, (width, height) = pickle.load(f)

            # Fix polygons with invalid unary values (e.g., too small to rasterize)
            unary = fix_unary_with_neighbors(unary, graph)

            # Run MRF label optimization
            labels = run_gco_labeling(unary, graph, smoothness_cost=cost)

            # Rasterize predicted labels into a pixel mask
            pred_mask = rasterize_prediction(gdf, labels, height, width)

            # Load ground truth mask
            gt_mask = np.load(gt_path)

            # Compute evaluation metrics (IoU, Precision, Recall)
            metrics = evaluate_iou_metrics(gt_mask, pred_mask)

            # Evaluate label consistency (smoothness violation ratio)
            smoothness_stats = compute_smoothness_statistics(graph, labels, smoothness_cost=cost)

            # Store all results
            results.append({
                "image_name": basename + ".jpg",
                "unary_scale": unary_scale,
                "smoothness_cost": cost,
                **metrics,
                **smoothness_stats
            })

    # Save results to a text file
    with open(output_txt, "w") as f:
        f.write("image_name\tunary_scale\tsmoothness_cost\tTP\tFP\tFN\tPrecision\tRecall\tMean_IoU\tViolation_Ratio\n")
        for r in results:
            f.write(f"{r['image_name']}\t{r['unary_scale']}\t{r['smoothness_cost']}\t{r['TP']}\t{r['FP']}\t{r['FN']}\t{r['Precision']:.4f}\t{r['Recall']:.4f}\t{r['Mean_IoU']:.4f}\t{r['Violation_Ratio']:.4f}\n")

