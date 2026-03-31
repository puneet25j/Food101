# Food101: Surpassing Research Benchmarks with EfficientNetB0

This project implements a high-performance deep learning pipeline to classify 101 different food categories using the Food101 dataset. The primary objective was to outperform the **DeepFood (2016)** research benchmark of 77.4% accuracy.

## 🚀 Results
* **Final Accuracy:** 86.15% (Top-1)
* **Benchmark Comparison:** +8.75% improvement over 2016 SOTA.
* **Training Time:** Optimized from ~48 hours (original paper) to **35 minutes** via transfer learning and pipeline parallelization.

## 🛠️ Tech Stack & Optimizations
* **Architecture:** EfficientNetB0 (Transfer Learning)
* **Framework:** TensorFlow / Keras
* **Hardware:** Apple Silicon (M4) & Google Colab (T4 GPU)
* **Pipeline:** Utilized `tf.data` with `AUTOTUNE`, parallel mapping, and prefetching to eliminate CPU bottlenecks.
* **Regularization:** Global Fine-Tuning, Dropout (0.3), and Custom Image Augmentation (Crops, Flips, Brightness).

## 📈 Project Highlights
* **Hardware-Aware ML:** Engineered the data pipeline specifically to handle the Unified Memory architecture of the M4 and the constraints of Colab.
* **Deep Error Analysis:** Conducted a post-training analysis of the "Top 10 Most Confused Classes" to identify visual similarities in dataset labels (e.g., Steak vs. Pork Chops).
* **Production-Ready Code:** Fully documented helper functions for preprocessing, checkpointing, and inference.

## 📂 Project Structure
- `Food101.ipynb`: Main experimental notebook.
- `helper_functions.py`: Modular utilities for data prep and visualization.
- `models/`: Saved weights for the 86.15% accuracy champion.