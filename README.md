# Roof-Structure-Extraction-from-Remote-Sensing-Images

This repository contains code from my master's thesis, including line-to-polygon conversion, model training, mask generation, and MRF-based evaluation.

## Datasets Used

This project uses two publicly available rooftop datasets for training and evaluation.  
**Note:** *Only the datasets were used; methods or models proposed in the original papers were not applied.*

### 1. Vectorization-Roof-Data-Set

A dataset designed for rooftop structure vectorization, based on aerial imagery from Detmold, Germany. It includes polygonal annotations and line-based representations of roofs.

- **Dataset:** [SimonHensel/Vectorization-Roof-Data-Set](https://github.com/SimonHensel/Vectorization-Roof-Data-Set)  
- **Paper:** *BUILDING ROOF VECTORIZATION WITH PPGNET*

#### Citation

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

This dataset contains annotations for 2,001 buildings across Los Angeles, Las Vegas, and Paris, using cropped RGB satellite images from the SpaceNet corpus.  
It was introduced in the ECCV 2020 paper *Vectorizing World Buildings*, and provides building rooftop structures in the form of planar graphs.

- **Dataset & Code:** [ennauata/buildings2vec](https://github.com/ennauata/buildings2vec)  
- **Paper:** *Vectorizing World Buildings: Planar Graph Reconstruction by Primitive Detection and Relationship Inference (ECCV 2020)*

#### Citation

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

The original annotations (e.g., point-edge graphs) cannot be directly used for training instance segmentation models like Mask R-CNN. We first converted them into closed polygons and rasterized binary masks for training.

##  Mask R-CNN Training

The file [`resnet50_RoofVec.py`](./resnet50_RoofVec.py) provides a full training pipeline for instance segmentation using **Mask R-CNN with a ResNet-50 backbone**, built on top of `torchvision`.

This script was primarily used to train a rooftop instance segmentation model on the **RoofVec** dataset, but it can also be used with the **Cities Dataset** (from the *Vectorizing World Buildings* paper) with minimal changes.

Both datasets follow the same preprocessed format: RGB images with corresponding JSON polygon annotations.

### Training Outputs

After training completes, the script will automatically save:

- The best model checkpoint:  
  e.g., `best_model_epoch59_iou0.9415_RoofVec.pth`

- Training visualizations:
  - `learning_rate_schedule_RoofVec.png` — the LR decay over epochs
  - `training_loss_curve_RoofVec.png` — loss trend during training
  - `validation_mean_iou_curve_RoofVec.png` — validation IoU per epoch

These outputs can be used to evaluate training dynamics or reload the best-performing model.


## 🧱 Roofline Extraction from RGB Orthophotos

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
