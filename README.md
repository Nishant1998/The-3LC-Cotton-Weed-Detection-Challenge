# 3LC Cotton Weed Detection – YOLOv8n Label-Centric Solution

This repository contains my solution for the **3LC Cotton Weed Detection Challenge**. The workflow is:

1. **Phase 1 – Label Cleaning with 3LC**
   Run `cotton_weed_starter_notebook.ipynb` locally to train a short YOLOv8n baseline, send runs to 3LC, inspect errors in the Dashboard, and export corrected labels.
2. **Phase 2 – Final Training on Kaggle**
   Upload the corrected labels as a Kaggle dataset and run `Kaggle_training_code.ipynb` to train longer YOLOv8n models and generate the competition submission.

The final solution uses only **YOLOv8n** models and improves performance mainly through **better labels** and **ensembling**.

---

## Repository Structure

```text
.
├── 3LC dashboard screenshots/      # Example screenshots for reference (optional)
├── kaggle_cotton_weed_detection.zip # Zipped project (for sharing/submission)
├── cotton_weed_starter_notebook.ipynb  # Phase 1: 3LC label-cleaning notebook
├── Kaggle_training_code.ipynb          # Phase 2: final training & inference
├── README.md                           # This file
└── .gitignore / .git                   # Git metadata
```

---

## 1. Environment Setup (3LC + Ultralytics)

> This section is adapted from the official 3LC starter notebook so you can reproduce the setup directly from this README.

### Before You Begin

This project uses **3LC (Three Lines of Code)** for data-centric AI workflows. Follow these steps to set up your environment.

#### Video Guide

Watch the full setup walkthrough: [3LC Quickstart Video](https://www.youtube.com/watch?v=zdIq1QpeSI8&list=PLFOZfHCPrAhDbgmxYcu9Qq5UUVMf7YLFy)

---

### Step 1: Create a 3LC Account

1. Go to [https://account.3lc.ai](https://account.3lc.ai)
2. Create your account

   * **Note:** A default workspace is automatically created for you (visible after login)
   * Your workspace name is what others on your team will see when collaborating
3. Get your API key from [https://account.3lc.ai/api-key](https://account.3lc.ai/api-key) and save it for the next step

---

### Step 2: Set Up Python Environment (Recommended)

#### Create a Virtual Environment

**Windows:**

```bash
python -m venv cotton-weed-env
cotton-weed-env\Scripts\activate
```

**Linux/MacOS:**

```bash
python -m venv cotton-weed-env
source cotton-weed-env/bin/activate
```

> You can skip this if you prefer to use your current Python environment.

---

### Step 3: Install 3LC and Dependencies

#### ⚠️ Important: PyTorch GPU Setup (For GPU Training)

The following command will install PyTorch, but **by default it installs the CPU version**.

**If you have a GPU and want to use it for training:**

1. First, install 3LC:

   ```bash
   pip install 3lc-ultralytics
   ```
2. Then, **reinstall PyTorch with CUDA support:**

   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

   Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to find the correct command for your CUDA version.

**If you're fine with CPU training (slower but works):**

```bash
pip install 3lc-ultralytics
```

**What gets installed:**

* `3lc-ultralytics` – 3LC integration with Ultralytics YOLO
* `3lc` – Core 3LC library
* `ultralytics`, `torch`, `pandas`, `numpy`, `pillow`, `pycocotools`, and other dependencies

**System Requirements:**

* Python 3.8+
* Windows 10+, Linux, or macOS
* GPU with CUDA (optional but recommended for faster training)

---

### Step 4: Login to 3LC

Once finished with installation, in the **same terminal** login using an API key. [Create one if not present](https://account.3lc.ai/api-key).
Replace `<your_api_key>` with your actual API key:

```bash
3lc login <your_api_key>
```

This saves your API key locally. **For future sessions, you don't need to run this again.**

---

### Step 5: Start the 3LC Service (for Dashboard visualization)

**Important Clarification:**

* The 3LC service is **NOT required for training** – you can train models without it
* It **IS required** if you want to use the 3LC Dashboard to visualize and analyze your data/runs
* Since the Dashboard is a key part of the data-centric workflow, we recommend starting it

**To start the service, run in the terminal:**

```bash
3lc service
```

**What happens:**

* Starts the local service for Dashboard connectivity
* Loads some preloaded example projects to help you learn 3LC
* These examples are useful for getting familiar with the Dashboard features

**Keep this terminal open while using the Dashboard.** To stop: press `Q` or `Ctrl+C`.

---

### Step 6: Open 3LC Dashboard (for visualization)

Once the service is running, open [https://dashboard.3lc.ai](https://dashboard.3lc.ai) in your browser.

**Browser Requirements:**

* Chrome (recommended), Firefox, or Edge (latest versions)
* Hardware acceleration enabled (GPU) for smoother experience
  (see the [3LC GPU Acceleration Guide](https://docs.3lc.ai/3lc/latest/user-guide/dashboard/gpu-acceleration.html))

**Tip:** Explore the preloaded example projects to learn Dashboard features before working on competition data.

---

### Verification Checklist

Before running any notebook, ensure:

1. ✅ 3LC account created and API key obtained
2. ✅ Python environment activated (if using venv)
3. ✅ `3lc-ultralytics` installed
4. ✅ PyTorch GPU version installed (if you have a GPU)
5. ✅ Logged in to 3LC (`3lc login <api_key>`)
6. ✅ 3LC service running and Dashboard open (optional but recommended)

**For future sessions:** Only steps 2 and 6 are needed (activate environment and optionally start service/Dashboard).

---

### Additional Resources

* [3LC Documentation](https://docs.3lc.ai/)
* [3LC Example Notebooks](https://github.com/3lc-ai/3lc-examples?tab=readme-ov-file)
* [Getting Started Video Playlist](https://www.youtube.com/watch?v=zdIq1QpeSI8&list=PLFOZfHCPrAhDbgmxYcu9Qq5UUVMf7YLFy)

---

### Troubleshooting

* **"3lc: command not found"** → Activate your Python environment or reinstall `3lc`
* **"API key invalid"** → Check your API key at [https://account.3lc.ai/api-key](https://account.3lc.ai/api-key)
* **"No GPU detected" during training** → Reinstall PyTorch with CUDA support (see Step 3)
* **"Cannot connect to Dashboard"** → Make sure `3lc service` is running in a terminal

---

## 2. Dataset Configuration

The competition dataset is organized in YOLO format with `train`/`val`/`test` splits already prepared.

### Dataset Structure

```text
cotton_weed_dataset/
├── train/images/        # 542 training images
├── train/labels/        # 542 YOLO label files
├── val/images/          # 133 validation images
├── val/labels/          # 133 YOLO label files
├── test/images/         # 170 test images (no labels provided)
└── dataset.yaml         # YOLO dataset configuration
```

### Test Set Information

The test set contains **170 images** split as follows:

* **Public leaderboard**: 85 images (50%)
* **Private leaderboard**: 85 images (50%)

> All 170 test images are included in the download for convenience. Only the public/private split determines which images are used for each leaderboard score.

### YOLO Label Format

Each `.txt` file contains bounding boxes in the form:

```text
class_id x_center y_center width height
```

All coordinates are normalized to `[0, 1]`.

### ⚠️ Data Quality Note

This dataset includes labeling imperfections. The whole point of this project is to use 3LC and a **train–fix–retrain** loop to identify and fix these issues before running long trainings.

---

## 3. How to Run – Phase 1: 3LC Label Cleaning

1. **Download the competition dataset** and arrange it as described in [Dataset Configuration](#2-dataset-configuration).
2. Ensure your 3LC environment is set up and the Dashboard is running.
3. Open `cotton_weed_starter_notebook.ipynb` in Jupyter / VS Code / your IDE.
4. Configure the dataset path inside the notebook (point it to `cotton_weed_dataset/`).
5. Run the notebook:

   * Train a short YOLOv8n model (few epochs) with 3LC tracking enabled.
   * Sync the run to 3LC and open it in the Dashboard.
   * Use the Dashboard to find missing boxes, misaligned boxes, and class confusions (especially Carpetweed / Morningglory / Palmer Amaranth).
6. After several passes of corrections, export the corrected labels from 3LC (CSV) and convert them back to YOLO TXT format (the notebook includes the conversion code).
7. Save the cleaned dataset (e.g. as `cotton_weed_dataset_cleaned/`) – this is what will be used for Phase 2.

---

## 4. How to Run – Phase 2: Final Training on Kaggle

1. Create a **Kaggle Dataset** from the cleaned labels (mirroring the original YOLO folder structure).
2. Open `Kaggle_training_code.ipynb` on Kaggle (or locally, if you prefer) and attach the cleaned dataset.
3. Configure paths (input dataset, working directory) in the first cells.
4. Run the notebook to:

   * Train multiple YOLOv8n variants with different hyperparameters and augmentations.
   * Evaluate them on the validation split.
   * Run inference on the test set.
   * Optionally ensemble predictions using Weighted Boxes Fusion (WBF) and create a submission ZIP.
5. Submit the ZIP on Kaggle to obtain leaderboard scores.

> In my final solution, all models were trained on the **3LC-corrected labels only**, and the ensemble of several YOLOv8n runs outperformed any individual model.

---

## 5. Dependencies Summary

* Python 3.8+
* [`3lc-ultralytics`](https://pypi.org/project/3lc-ultralytics/)
* `torch`, `torchvision` (GPU build recommended)
* `ultralytics`
* `numpy`, `pandas`, `tqdm`, `matplotlib`, `pillow`, `pycocotools`
* (Optional, for ensembling) `ensemble-boxes`

All of these are installed automatically when you follow the commands in the [Environment Setup](#1-environment-setup-3lc--ultralytics) section, except `ensemble-boxes`, which you can add via:

```bash
pip install ensemble-boxes
```

---

## 6. Notes & Tips

* Keep Phase 1 training **short** – its purpose is to surface label issues, not to be the final model.
* When in doubt about weed species (e.g. Palmer Amaranth vs Waterhemp), prefer **conservative edits** rather than overconfident relabeling.
* Back up your original labels before overwriting them with the cleaned versions.
* Save intermediate 3LC exports so you can compare label quality across iterations.

---

## 7. Optional: Kaggle Resources

If you want to see the full Kaggle pipeline and artifacts, you can refer to:

* **Solution Notebook:** [https://www.kaggle.com/code/nishantvalvi1504/the-3lc-cotton-weed-detection-challenge](https://www.kaggle.com/code/nishantvalvi1504/the-3lc-cotton-weed-detection-challenge)
* **Trained YOLOv8n Models:** [https://www.kaggle.com/models/nishantvalvi1504/3cl-cotton-weed-det-yolov8n-models](https://www.kaggle.com/models/nishantvalvi1504/3cl-cotton-weed-det-yolov8n-models)
* **Corrected Labels Dataset:** [https://www.kaggle.com/datasets/nishantvalvi1504/3cl-cotton-weed-det-final-labels](https://www.kaggle.com/datasets/nishantvalvi1504/3cl-cotton-weed-det-final-labels)
