# 10th Place Solution: YOLOv8n with Pseudo Ensembling and 3LC Label Cleaning

## Overview

This solution finished **10th Place** by focusing on robust model tuning and ensembling strategies within the strict YOLOv8n architecture constraints. The main idea was to create model diversity through hyperparameter search and then combine predictions using **Weighted Boxes Fusion (WBF)** to build a stronger detector than any single model.

Alongside model tuning, the key driver of performance was **systematic label cleaning with 3LC**, followed by long YOLOv8n trainings on the corrected labels and a final pseudo-ensemble at inference time.

---

## Label Cleaning with 3LC

Before training the final models, I cleaned the CottonWeedDet3 labels using the **3LC Dashboard**. I first trained a plain YOLOv8n model on the original labels and uploaded the run to 3LC, following the official hard-hat tutorial ([3LC Guide](https://docs.3lc.ai/3lc/latest/tutorials/hard-hat.html)).

In the Dashboard, I used IoU and confidence filters to surface:
- **High-confidence, low-IoU predictions** → likely **missing boxes**  
- **Mid-IoU predictions** → **misaligned or poorly sized boxes** that needed resizing or repositioning  

These were batch-added or replaced as new ground truth.

For class errors, I focused on confusion between **Carpetweed**, **Morningglory**, and **Palmer Amaranth** and compared doubtful plants against external weed-identification resources:
- [Missouri WeedID – Carpetweed](https://weedid.missouri.edu/weedinfo.cfm?weed_id=179)  
- [Ontario Carpetweed guide](https://www.ontario.ca/document/weed-identification-guide-ontario-crops/carpetweed)  
- [Cornell Morningglories](https://cals.cornell.edu/weed-science/weed-profiles/morningglories)  
- [UMD key for Palmer Amaranth vs Waterhemp](https://extension.umd.edu/resource/keys-identifying-palmer-amaranth-and-waterhemp-fs-2023-0653/)  
- [Cornell Palmer Amaranth](https://cals.cornell.edu/weed-science/weed-identification/pigweed-identification/palmer-amaranth)  
- [Purdue WS-51 guide](https://www.extension.purdue.edu/extmedia/ws/ws-51-w.pdf)

Palmer Amaranth was particularly difficult to identify at the seedling stage because it often looks like waterhemp, so I only relabeled plants when the traits clearly matched and otherwise left them unchanged or removed them from training.

After several passes through these FP/FN and confusion views, I exported the corrected labels from 3LC as CSV files, converted them back to YOLO TXT format, and trained all YOLOv8n models **only on this revised label set**.

---

## Training Strategy & Model Diversity

To make the most of the lightweight YOLOv8n model, I trained multiple model variations to introduce diversity in the ensemble. I used the `ultralytics` library and performed hyperparameter searching with `model.tune()`, varying:

- **Optimizers:** `AdamW` vs `SGD`  
- **Training durations:** roughly **150–900 epochs**, depending on the configuration  
- **Augmentation intensities:** different strengths of Mosaic, MixUp, and Copy-Paste, plus smaller tweaks in other augmentations  

Each configuration produced a slightly different detector. I selected runs that:
- Performed well on the validation set, and  
- Showed **complementary error patterns** (e.g., some better on small weeds, others more robust under heavier augmentation).

These diverse YOLOv8n checkpoints became the inputs to the final pseudo-ensemble.

---

## Inference & Pseudo Ensembling (WBF)

Instead of standard Non-Maximum Suppression (NMS), I used **Weighted Boxes Fusion (WBF)** based on *[Weighted boxes fusion: Ensembling boxes from different object detection models](https://arxiv.org/abs/1910.13302)*. This made it possible to combine predictions from several models in a more consistent way.

From the competition rules, pseudo-ensembling was explicitly allowed:

> **“✅ Permitted: Pseudo-ensembling multiple YOLOv8n models via weighted averaging of predictions”**

WBF fits exactly into this category: it takes the boxes and scores from multiple YOLOv8n models and produces fused boxes via **weighted averaging** of their coordinates and confidences. No larger architectures are introduced – the ensemble remains fully compliant with the YOLOv8n-only constraint.

- **Ensemble Structure:** Predictions from the different YOLOv8n variants were merged into a single set of boxes.  
- **WBF Configuration:**
  - `IoU Threshold`: 0.45  
  - `Skip Box Threshold`: 0.01  
  - `Weights`: Higher weights were given to models that performed more reliably on the validation set.

This approach produced cleaner final boxes and improved mAP compared to using any of the individual models alone, while staying strictly within the competition rules.

---

## Results

- **Public Score:** 0.87430  
- **Private Score:** 0.82953  

The gap between public and private scores was small, suggesting that the label-cleaning and ensembling strategy generalized reasonably well across the hidden test images.

---

## Resources

- **Solution Notebook:** <https://www.kaggle.com/code/nishantvalvi1504/the-3lc-cotton-weed-detection-challenge>  
- **Trained YOLOv8n Models:** <https://www.kaggle.com/models/nishantvalvi1504/3cl-cotton-weed-det-yolov8n-models>  
- **Corrected Labels Dataset:** <https://www.kaggle.com/datasets/nishantvalvi1504/3cl-cotton-weed-det-final-labels>  

These resources contain the exact code, weights, and labels used to reproduce the leaderboard scores.

---

## Acknowledgements

I’m grateful to the **3LC team** and competition organizers for designing a challenge that encourages working on label quality and practical edge deployment instead of just scaling model size. The competition setup made it a very useful exercise in combining annotation tools, model training, and evaluation.

Thanks as well to the broader **YOLOv8 and 3LC community** for the tools, examples, and documentation that made it easier to experiment with different training and labeling workflows.

---

## References

- **3LC Guide:**  
  - [Improving models by fixing object detection labels (hard-hat tutorial)](https://docs.3lc.ai/3lc/latest/tutorials/hard-hat.html)

- **Weed Identification:**  
  - [Missouri WeedID – Carpetweed](https://weedid.missouri.edu/weedinfo.cfm?weed_id=179)  
  - [Ontario – Carpetweed, Weed Identification Guide](https://www.ontario.ca/document/weed-identification-guide-ontario-crops/carpetweed)  
  - [Cornell – Morningglories](https://cals.cornell.edu/weed-science/weed-profiles/morningglories)  
  - [UMD Extension – Keys to Identifying Palmer Amaranth and Waterhemp](https://extension.umd.edu/resource/keys-identifying-palmer-amaranth-and-waterhemp-fs-2023-0653/)  
  - [Cornell – Palmer Amaranth (Pigweed Identification)](https://cals.cornell.edu/weed-science/weed-identification/pigweed-identification/palmer-amaranth)  
  - [Purdue – WS-51 Weed Identification Guide](https://www.extension.purdue.edu/extmedia/ws/ws-51-w.pdf)

- **Ensembling:**  
  - [Weighted Boxes Fusion: Ensembling boxes from different object detection models](https://arxiv.org/abs/1910.13302)
