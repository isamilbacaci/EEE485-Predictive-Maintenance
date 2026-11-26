# Predictive Maintenance using From-Scratch NumPy Implementations

An end-to-end statistical learning and predictive maintenance pipeline built entirely from scratch using **NumPy** (zero external ML libraries such as `scikit-learn`, `TensorFlow`, or `PyTorch`). Developed for **EEE 485: Statistical Learning and Data Analytics** at İhsan Doğramacı Bilkent University.

## Key Engineering Highlights
- **Zero External ML Libraries:** Implemented Gaussian Discriminant Analysis (GDA), L2-Regularized Logistic Regression, and a 1-Hidden Layer Multi-Layer Perceptron (MLP) purely with NumPy linear algebra.
- **Cost-Sensitive Learning (Class Weighting):** Addressed severe class imbalance (96.6% normal vs. 3.4% failure) and the **Accuracy Paradox** by integrating a 20x False-Negative penalty directly into the gradient updates.
- **Custom Backpropagation:** Derived analytical gradients using the chain rule and Sigmoid derivatives with manual matrix transpositions.
- **Strict Data Leakage Prevention:** Excluded failure-mode indicator columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and fitted Z-score normalization ($\mu, \sigma$) strictly on the training split.

## Experimental Results (20% Unseen Test Set, $N=2000$)

| Model | Accuracy | TP | TN | FP | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GDA (Baseline)** | 96.50% | 14 | 1916 | 23 | 47 | 0.3784 | 0.2295 | 0.2857 |
| **Weighted L2 Logistic Regression** | 86.40% | 50 | 1678 | 261 | 11 | 0.1608 | 0.8197 | 0.2688 |
| **Weighted MLP (1-Hidden Layer)** | **92.90%** | **51** | **1807** | **132** | **10** | **0.2787** | **0.8361** | **0.4180** |

### Industrial Cost-Benefit & Feature Importance
- **Labor Reduction:** Transitioning from linear Weighted LR to non-linear Weighted MLP cut False Positives nearly in half (261 $\rightarrow$ 132) while increasing Recall to **83.61%**, saving **32.5 technician inspection hours** per 2,000 operational cycles (assuming 15 min/inspection).
- **Physical Interpretability:** Extracted Logistic Regression weight magnitudes ($|W|$) confirmed **Torque (2.4691)** and **Rotational Speed (1.7884)** as the dominant failure predictors, followed by Air Temperature (1.7565) and Process Temperature (1.1741).

## Repository Structure
- `main.py`: Custom train-test split, Z-score standardization, GDA, Weighted Ridge LR, Weighted MLP, evaluation metrics, and feature importance extraction.
- `eda.py`: Exploratory data analysis script generating class distribution, Torque vs. RPM scatter, correlation heatmap, and Torque boxplot figures.

## Quickstart
```bash
pip install numpy pandas matplotlib seaborn
python eda.py
python main.py