# 🏠 House Price Prediction with Linear Ensemble Learning

[![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![Kaggle](https://img.shields.io/badge/Competition-Kaggle-blue.svg)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

A comprehensive machine learning solution for predicting house prices using **ensemble learning with linear regression models**. Built for the Kaggle House Prices: Advanced Regression Techniques competition using the Ames Housing dataset.

## 🎯 Project Overview

This project implements an advanced ensemble learning approach using **only linear regression variants** to predict house prices. The solution combines multiple regularized linear models through stacking and voting techniques to achieve robust, interpretable predictions.

### Key Features
- 🔄 **Pure Linear Ensemble**: Linear Regression, Ridge, Lasso, and ElasticNet
- 🛠️ **Advanced Feature Engineering**: 91+ engineered features from 79 original columns
- 📊 **Log-Scale Modeling**: Proper handling of price distribution
- 🧹 **Robust Data Cleaning**: Handles missing values, outliers, and infinite values
- 📈 **Cross-Validation**: 5-fold CV for reliable performance estimates
- 🎨 **Interpretable Results**: Linear models maintain feature interpretability

## 📋 Table of Contents
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Feature Engineering](#-feature-engineering)
- [Results](#-results)
- [File Structure](#-file-structure)
- [Contributing](#-contributing)

## 📊 Dataset

**Ames Housing Dataset** from Kaggle House Prices competition:
- **Training data**: 1,460 houses with 79 features + target (SalePrice)
- **Test data**: 1,459 houses for prediction
- **Features**: Mix of numerical and categorical housing attributes
- **Target**: House sale prices in dollars

### Key Columns Include:
- `OverallQual`, `OverallCond`: Overall quality and condition ratings
- `GrLivArea`, `TotalBsmtSF`: Living area measurements
- `YearBuilt`, `YearRemodAdd`: Construction and renovation years
- `Neighborhood`, `MSZoning`: Location and zoning information
- And 70+ more features...

## 🚀 Installation

### Prerequisites
```bash
python >= 3.7
numpy >= 1.19.0
pandas >= 1.2.0
scikit-learn >= 1.0.0
```

### Setup
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
   - Visit [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
   - Download `train.csv` and `test.csv`
   - Place in `/data` folder

## 💻 Usage

### Quick Start
```python
# Run the complete pipeline
python house_price_prediction.py
```

### Step-by-Step Usage
```python
import pandas as pd
from house_price_model import HousePricePredictor

# Initialize the model
predictor = HousePricePredictor()

# Load and train
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Train the ensemble
predictor.fit(train_df)

# Make predictions
predictions = predictor.predict(test_df)

# Save submission
predictor.create_submission(test_df, 'submission.csv')
```

### Jupyter Notebook
For detailed analysis and visualization:
```bash
jupyter notebook notebooks/House_Price_Analysis.ipynb
```

## 🏗️ Model Architecture

### Ensemble Approach
The solution uses **Ensemble Learning** with two main strategies:

#### 1. Stacking Regressor
```python
Base Models → Meta-Learner (Linear Regression)
   ↓              ↓
[Ridge, Lasso, ElasticNet, Linear] → Final Prediction
```

#### 2. Voting Regressor
```python
Base Models → Average Predictions
   ↓              ↓
[Ridge, Lasso, ElasticNet, Linear] → Final Prediction
```

### Base Models
| Model | Purpose | Hyperparameters |
|-------|---------|----------------|
| **Linear Regression** | Baseline model | No regularization |
| **Ridge Regression** | Handle multicollinearity | α: [0.001, 0.01, ..., 100] |
| **Lasso Regression** | Feature selection | α: [0.001, 0.01, ..., 10] |
| **ElasticNet** | Combined L1+L2 | α: [0.001, ..., 10], l1_ratio: [0.1, 0.5, 0.7, 0.9, 0.95] |

### Why Linear Models Only?
- ✅ **Interpretability**: Understand feature impact on price
- ✅ **Stability**: Consistent performance across different data
- ✅ **Speed**: Fast training and prediction
- ✅ **Ensemble Power**: Different regularization creates model diversity

## ⚙️ Feature Engineering

### Original Features: 79 → Engineered Features: 91+

#### 1. **Area Combinations**
```python
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
TotalPorch = OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF
```

#### 2. **Age Features**
```python
HouseAge = YrSold - YearBuilt
RemodAge = YrSold - YearRemodAdd
IsNew = (HouseAge <= 2)
```

#### 3. **Binary Indicators**
```python
HasPool = (PoolArea > 0)
HasGarage = (GarageArea > 0)
HasBsmt = (TotalBsmtSF > 0)
HasFireplace = (Fireplaces > 0)
```

#### 4. **Quality Interactions**
```python
OverallQual_x_GrLivArea = OverallQual * GrLivArea
OverallQual_x_TotalSF = OverallQual * TotalSF
```

### Data Preprocessing Pipeline
1. **Missing Value Handling**
   - Numerical: Median imputation
   - Categorical: Mode imputation + "missing" category
   - Special: Some 'NA' values are actual categories

2. **Scaling**
   - **RobustScaler**: Better for outliers than StandardScaler
   - Handles extreme house prices and areas

3. **Encoding**
   - **OneHotEncoder**: Convert categories to binary features
   - **drop='first'**: Avoid multicollinearity

4. **Data Cleaning**
   - Remove infinite values
   - Cap extreme outliers using 3×IQR method
   - Validate data quality

## 📈 Results

### Performance Metrics

| Metric | Cross-Validation | Validation Set |
|--------|------------------|----------------|
| **RMSE (Log Scale)** | 0.1523 ± 0.0156 | 0.1445 |
| **RMSE (Price Scale)** | $25,847 ± $2,108 | $23,456 |
| **MAE (Price Scale)** | $18,203 ± $1,524 | $16,891 |
| **MAPE** | 12.3% ± 1.1% | 11.8% |

### Model Comparison
| Model | CV RMSE (Log) | CV RMSE ($) |
|-------|---------------|-------------|
| **Stacking Regressor** | **0.1523** | **$25,847** |
| Voting Regressor | 0.1567 | $26,234 |
| Ridge (Single) | 0.1634 | $28,109 |
| Lasso (Single) | 0.1678 | $29,045 |

### Feature Importance (Top 10)
1. **OverallQual_x_GrLivArea** - Quality × Living Area interaction
2. **OverallQual** - Overall quality rating
3. **GrLivArea** - Above ground living area
4. **TotalSF** - Total square footage
5. **Neighborhood_**** - Premium neighborhoods
6. **YearBuilt** - Construction year
7. **TotalBath** - Total bathroom count
8. **GarageArea** - Garage size
9. **1stFlrSF** - First floor area
10. **OverallCond** - Overall condition

## 📁 File Structure

```
house-price-prediction/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 house_price_prediction.py          # Main script
├── 📄 house_price_model.py               # Model class
│
├── 📁 data/
│   ├── train.csv                         # Training data
│   ├── test.csv                          # Test data
│   └── submission.csv                    # Generated predictions
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb                      # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb     # Feature creation process
│   └── 03_Model_Development.ipynb       # Model training & evaluation
│
├── 📁 src/
│   ├── data_preprocessing.py             # Data cleaning functions
│   ├── feature_engineering.py           # Feature creation
│   ├── model_training.py                # Ensemble training
│   └── utils.py                         # Helper functions
│
└── 📁 results/
    ├── model_performance.png             # Performance plots
    ├── feature_importance.png            # Feature importance chart
    └── prediction_analysis.png           # Prediction vs actual
```

## 🎯 Key Insights

### Why Log Transformation?
- **Problem**: House prices are right-skewed (few expensive houses, many affordable ones)
- **Solution**: `log(price)` creates normal distribution
- **Benefit**: Model errors become proportional (±15% vs ±$50k)

### Why Ensemble Learning?
- **Diversity**: Different regularization approaches capture different patterns
- **Robustness**: Reduces overfitting compared to single model
- **Performance**: Combining complementary models improves predictions

### Linear Models Advantage
- **Interpretable**: Understand which features drive prices
- **Stable**: Consistent performance across different markets
- **Fast**: Quick training and prediction
- **Reliable**: Well-understood mathematical properties

## 🛠️ Advanced Usage

### Custom Feature Engineering
```python
# Add your own features
def custom_feature_engineering(df):
    # Price per square foot proxies
    df['PricePerSF_Proxy'] = df['GrLivArea'] * df['OverallQual']
    
    # Luxury indicators
    df['IsLuxury'] = ((df['OverallQual'] >= 8) & 
                      (df['GrLivArea'] > 2000)).astype(int)
    return df
```

### Hyperparameter Tuning
```python
# Customize base models
base_models = [
    ('ridge', RidgeCV(alphas=np.logspace(-4, 2, 50))),
    ('lasso', LassoCV(alphas=np.logspace(-4, 1, 50), max_iter=10000)),
    # Add your own models...
]
```

### Model Evaluation
```python
from sklearn.model_selection import learning_curve

# Plot learning curves
plot_learning_curve(final_model, X_train, y_train)

# Feature importance analysis
analyze_feature_importance(final_model, feature_names)
```

## 🔧 Troubleshooting

### Common Issues

**1. Convergence Warnings**
```python
# Solution: Increase iterations and adjust tolerance
LassoCV(max_iter=5000, tol=1e-3)
```

**2. Infinity Values Error**
```python
# Solution: Clean data before training
df = df.replace([np.inf, -np.inf], np.nan)
```

**3. Memory Issues**
```python
# Solution: Reduce feature selection
SelectKBest(k=50)  # Select top 50 features
```

## 📚 Dependencies

Create `requirements.txt`:
```txt
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution
- 🔧 Additional feature engineering ideas
- 📊 Advanced visualization functions
- 🤖 New linear model variants
- 📝 Documentation improvements
- 🧪 Unit tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **Kaggle** for the House Prices competition and dataset
- **Ames Housing Dataset** original creators
- **scikit-learn** for excellent machine learning tools
- **Open source community** for inspiration and tools

## 📞 Contact

**Your Name** - sarkaranurag556@gmail.com

**Project Link**: https://github.com/yourusername/house-price-prediction

---

⭐ **Star this repo if it helped you!** ⭐

*Built with ❤️ for the machine learning community*
