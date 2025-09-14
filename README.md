# YOLO11 Optimization for Lightweight and Accurate Plant Detection in UAV Aerial Imagery

## 1. Overview

This repository contains the code supporting the article **"YOLO11 Optimization for Lightweight and Accurate Plant Detection in UAV Aerial Imagery"**, which is currently under peer review.

YOLO configuration files supporting the experiments in the paper can be found in the `cfgs` folder.

The `cfgs` folder contains two subfolders:

- **`Ablation Study`**: Includes configuration files for all models used in the Ablation Study section of the paper.
- **`SOTA Comparison`**: Configuration files for all models used in the paper’s **Comparison with State-of-the-Art Methods** and **Transfer Learning Study** sections.

The paper introduces three optimization strategies. The P2AR strategy can be located in the YAML model configuration files under the `cfgs` folder. The integration and use of the CBAM module and Shape-IoU Loss require modifications to the code in the `ultralytics/nn/modules` directory within the YOLO11 project. Detailed instructions can be found in the **User Guide** section.

## 2. Dataset

The four original single-class datasets referenced in the paper can be downloaded from the authors’ repositories:

- **CBDA (Cotton Boll Detection Augmented)** and **WEDU (Wheat Ears Detection Update)** — [Ye-Sk / Plant-dataset](https://github.com/Ye-Sk/Plant-dataset)
- **MTDC-UAV (Maize Tassel Detection and Counting)** — [git.io/MTDC](https://git.io/MTDC)
- **RFRB (Rape Flower Rectangular Box Labeling)** — [CV-Wang / RapeNet](https://github.com/CV-Wang/RapeNet)





The annotation files for the merged Plant4 dataset can be found in the `dataset` folder of this repository.

## 3. User Guide

### 3.1 Integrating the CBAM Module into YOLO11

To integrate the CBAM module into the YOLO11 framework, follow these steps:

1. In the `ultralytics/nn/tasks.py` file of the YOLO11 project, add the following import statement:
   
   ```python
   from ultralytics.nn.modules.conv import CBAM
   ```

2. modify the `parse_model` function in `tasks.py` by adding the following code under the `if m in ...` branch:
   
   ```python
   elif m is CBAM:
       c1 = ch[f]
       args = [c1, *args[1:]]
   ```
   
   This ensures proper handling and integration of the `CBAM` module within the YOLO11 model architecture.

3. Update the YOLO11 model's YAML configuration file to include the `CBAM` module. Example configurations can be found in the `cfgs` directory of this repository.

**Notes:**

### 3.2 Using Shape-IoU Loss

In this study, we replaced the original CIoU loss function in YOLO11 with the Shape-IoU loss [2] function to enhance the accuracy of bounding box regression for small plant targets. To implement Shape-IoU in YOLO11, follow the instructions below.

- Import `shape_iou` (the `shape_iou_loss.py` file can be found in the `code` directory of this repository) into `ultralytics/utils/loss.py`.

- Modify the `forward` function in the `BboxLoss` class located in `ultralytics/utils/loss.py` to allow selecting different IoU Loss functions using a control variable (e.g., `config.bbox_loss`). Add the following code snippet as an example:
  
  ```python
  if config.bbox_loss == 'Shape-IoU':
      iou = shape_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
      loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
  else:   # CIoU
      iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
      loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
  ```

## References

[1] Lu, Dunlu, et al. "Plant detection and counting: Enhancing precision agriculture in UAV and general scenes." *IEEE Access* (2023).

[2] Zhang, Hao, and Shuaijie Zhang. "Shape-iou: More accurate metric considering bounding box shape and scale." *arXiv preprint arXiv:2312.17663* (2023).