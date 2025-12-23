# Roof-Structure-Extraction-from-Remote-Sensing-Images

This repository contains code from my master's thesis, including line-to-polygon conversion, model training, mask generation, and MRF-based evaluation.

If you use this repository or build upon this work, please cite the following:

> **Cheng, H.-Y.** (2025, June). *Roof structure extraction from remote sensing images* [Master’s thesis presentation and full thesis]. TU Delft MSc Thesis Final Presentation, Delft University of Technology.  
> [https://repository.tudelft.nl/record/uuid:4a31d2f4-1615-4afb-a2d8-a9cbfd1530b2](https://repository.tudelft.nl/record/uuid:4a31d2f4-1615-4afb-a2d8-a9cbfd1530b2)


## Datasets Used

This project uses two publicly available rooftop datasets for training and evaluation.  
Only the datasets were used — **the methods or models proposed in the original papers were not applied**.

Please cite the original dataset papers if you use this project or the datasets in your research.

---

### 1. Vectorization-Roof-Data-Set (RoofVec Dataset)
[https://github.com/SimonHensel/Vectorization-Roof-Data-Set](https://github.com/SimonHensel/Vectorization-Roof-Data-Set)

```bibtex
@Article{isprs-archives-XLVI-4-W4-2021-85-2021,
  AUTHOR = {Hensel, S. and Goebbels, S. and Kada, M.},
  TITLE = {BUILDING ROOF VECTORIZATION WITH PPGNET},
  JOURNAL = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  VOLUME = {XLVI-4/W4-2021},
  YEAR = {2021},
  PAGES = {85--90},
  URL = {https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLVI-4-W4-2021/85/2021/},
  DOI = {10.5194/isprs-archives-XLVI-4-W4-2021-85-2021}
}
```

### 2. Vectorizing World Buildings (Cities Dataset)

[ennauata/buildings2vec](https://github.com/ennauata/buildings2vec)  

```bibtex
@inproceedings{nauata2020vectorizing,
  title={Vectorizing World Buildings: Planar Graph Reconstruction by Primitive Detection and Relationship Inference},
  author={Nauata, Nelson and Furukawa, Yasutaka},
  booktitle={European Conference on Computer Vision},
  pages={711--726},
  year={2020},
  organization={Springer}
}

```


##  Data Preprocessing

The original annotations (e.g., point-edge graphs) cannot be directly used for training instance segmentation models like Mask R-CNN, which require closed polygon annotations. We converted the raw data into JSON-formatted closed polygons.

## Mask R-CNN Training

The script [`resnet50_RoofVec.py`](./resnet50_RoofVec.py) provides a complete training pipeline for rooftop instance segmentation using **Mask R-CNN**.

### Input & Output

- **Input:** RGB images and corresponding JSON files with closed polygon annotations  
- **Output:** Best model checkpoint (e.g., `best_model_epoch59_iou0.9415_RoofVec.pth`) and basic training visualizations in `.png` files (loss curve, learning rate schedule, validation IoU)





## Roofline Extraction from RGB Orthophotos

To extract structured rooflines from RGB orthophotos, this project uses the method from:

**[Unsupervised Roofline Extraction from True Orthophotos](https://github.com/tudelft3d/Roofline-extraction-from-orthophotos)**  
*Gao, Peters, and Stoter, 3D GeoInfo 2023*

This method detects line segments corresponding to rooftop structures and saves the output as `.gpkg` (GeoPackage) files. These line features are later used for polygon proposal generation and further post-processing in this project.

#### Citation

```bibtex
@article{sum2021,
author = {Weixiao Gao, Ravi Peters, and Jantien Stoter},
title = {Unsupervised Roofline Extraction from True Orthophotos for LoD2 Building Model Reconstruction},
journal = {Lecture Notes in Geoinformation and Cartography (LNG&C) series},
year={2023},
publisher = {Springer},
}

```

## Line-to-Polygon Conversion

The script [`lines_to_polygons.py`](./lines_to_polygons.py) converts roofline segments (stored as `.gpkg` LineStrings) into closed polygon geometries using `shapely.polygonize`.

This is a post-processing step that takes the line-based output from the *Unsupervised Roofline Extraction* method and turns it into usable polygonal proposals for later stages (e.g., MRF optimization).

## MRF Unary and Graph Preparation

The script [`MRF_RoofVec.py`](./MRF_RoofVec.py) prepares all necessary inputs for MRF optimization, including:

- Soft masks and scores from a trained **Mask R-CNN model**
- Closed polygons from previous `lines_to_polygons.py` output
- Computed **unary costs** per polygon (based on predicted instance masks)
- Built **adjacency graphs** using polygon edge adjacency
- All outputs saved for later optimization
- 

## Ground Truth Mask Generation

The script [`generate_gt_masks_from_polygons.py`](./generate_gt_masks_from_polygons.py) converts polygon annotations into pixel-level **ground truth masks**.

This is useful for:
- Quantitative evaluation (e.g., IoU, pixel accuracy)
- Visual comparison with predicted masks

### Workflow

- Input:
  - RGB images (`.jpg`)
  - Polygon annotations (`.json`) — format: list of closed polygons
- Output:
  - Binary masks (`.npy`) — 2D NumPy array (uint8)
  - Visual previews (`.png`) — each instance shaded for visibility


 
## MRF Inference and Evaluation

The script [`run_mrf_and_evaluate.py`](./run_mrf_and_evaluate.py) performs **final MRF label assignment** and **evaluates the results** against ground truth masks.

It uses Graph Cuts Optimization (via `gco`) to infer the optimal label configuration based on:
- Precomputed **unary potentials** (from model predictions)
- Polygon **adjacency graph**
- Varying **smoothness costs**

### Inputs

- Precomputed from previous stages:
  - `*_unary.npy`, `*_graph.pkl` (from MRF preparation)
  - `*.npy` GT masks (from polygon → mask script)

### Output

- `mrf_evaluation_summary_*.txt`: a tab-separated table summarizing:
  - TP, FP, FN
  - Mean IoU
  - Precision, Recall
  - Smoothness violation ratio

### Example Smoothness Settings

```python
smoothness_costs = [0.001, 0.1, 1, 10, 100]


## Citation

If you use this code or data processing pipeline in your research, please cite:

```bibtex
@misc{cheng2024roof,
  author = {Cheng, H.-Y.},
  title = {Roof structure extraction from remote sensing images},
  year = {2024},
  howpublished = {\url{https://repository.tudelft.nl/record/uuid:4a31d2f4-1615-4afb-a2d8-a9cbfd1530b2}},
  note = {Master’s thesis, Delft University of Technology}
}

