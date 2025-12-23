# Roof-Structure-Extraction-from-Remote-Sensing-Images

This repository contains code from my master's thesis, including line-to-polygon conversion, model training, mask generation, and MRF-based evaluation.

## Datasets Used

This project uses two publicly available rooftop datasets for training and evaluation.  
**Note:** *Only the datasets were used; methods or models proposed in the original papers were not applied.*

---

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

