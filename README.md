# COVID-19 Positive Rate Prediction (NTU ML Assignment Reproduction)

This project reproduces an older NTU Machine Learning assignment originally hosted on Kaggle.  
Because the Kaggle competition has been closed, this repository focuses on **learning**,  
**experimentation**, and **model evaluation** rather than leaderboard submission.

---

##  Project Overview

Goal:  
Predict `tested_positive.2` (COVID-19 positive rate) using demographic, behavioral, psychological,  
and epidemiological survey features collected across U.S. states.

Dataset:
- ~2700 training samples
- ~93 numerical features (state one-hot, symptoms, behavior patterns, mental state indicators)
- Separate test set provided (without labels)

Task type:
- **Supervised regression**

---

##  Techniques Used

### Data Processing
- Standardization (`StandardScaler`)
- Train/validation split
- Feature/target construction

### Models
| Model | Description |
|-------|-------------|
| Baseline | Predict mean of training labels |
| Linear Regression | Ridge / closed-form solution |
| Custom Linear GD | Hand-written gradient descent with PyTorch autograd |
| MLP (PyTorch) | 93 → 64 → 1 neural network |

### Optimization
- Adam optimizer
- Learning rate tuning
- Early stopping
- L2 regularization (weight decay)

---

##  Results (Validation)

| Model | RMSE |
|--------|-------|
| Baseline (mean predictor) | **7.48** |
| Linear Regression | **7.6** |
| MLP (64 hidden, Adam lr=0.01) | **7.48** |

Interpretation:
- Dataset is largely linear
- MLP slightly improves over linear, but improvement limited due to noise
- Strong regularization/dropout leads to underfitting  
- Larger MLPs quickly overfit (RMSE ↑)

---

##  Training Curves

Training and validation curves are stored in the `plots/` folder, including:
- Linear regression loss curve
- MLP loss curve
- Model comparison curves

---

##  File Structure

