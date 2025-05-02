# Pattern Recognition - Computational Assignment

**University:** University of Piraeus
**Department:** Informatics
**Academic Year:** 2022–2023
**Semester:** 5th

**Course:** Pattern Recognition
**Assignment:** Computational Course Assignment

---

## 📖 Introduction

This project was developed in **Python** using **Visual Studio Code** and several key libraries that greatly supported our work:

* pandas
* matplotlib
* scikit-learn
* keras
* seaborn
* numpy

These libraries were installed through the terminal using standard pip commands (see bibliography in the report).

---

## 🏗️ Data Preprocessing

1. **Data Loading:**
   We begin by loading the dataset and verifying its structure using the `.head()` function.

2. **Feature Separation:**
   The target feature is `median_house_value`.

   * `X`: all features except `median_house_value`
   * `z`: only `median_house_value`

3. **Numerical vs. Categorical Features:**
   Using `.info()`, we identify data types:

   * `ocean_proximity`: categorical
   * All others: numerical

4. **Scaling Data:**
   We apply **Min-Max Scaling** to bring numerical features into the 0–1 range:

   ```python
   scaler = MinMaxScaler()
   X_scaled = pd.DataFrame(scaler.fit_transform(X[numerical]), columns=numerical)
   X_temp = X.drop(numerical, axis=1)
   X = pd.concat([X_temp, X_scaled], axis=1)
   ```

   Similar scaling is applied to `z`.

5. **One-Hot Encoding:**
   We one-hot encode the categorical feature:

   ```python
   encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
   X_enc = pd.DataFrame(encoder.fit_transform(X[categorical]), columns=oc_prox)
   X_temp = X.drop(categorical, axis=1)
   X = pd.concat([X_temp, X_enc], axis=1)
   ```

6. **Handling Missing Values:**
   Using `SimpleImputer(strategy='median')`, we fill missing values (e.g., in `total_bedrooms`) with the median.

---

## 📊 Data Visualization

* **Histograms:**
  With Seaborn, we visualize distributions:

  ```python
  sns.histplot(data[column], bins=50, kde=True, lw=2)
  ```

* **Scatter Plots:**
  Using Pandas and Seaborn:

  ```python
  dataset.plot(kind='scatter', x='longitude', y='median_house_value')
  sns.scatterplot(x=data['median_income'], y=data['median_house_value'], hue=data['NEAR OCEAN'])
  ```

These plots reveal correlations (e.g., between `median_income` and `median_house_value`) and geographic patterns.

---

## 🔧 Regression Models

### Least Squares Regression

We implemented two core functions:

```python
def least_squares_train(X, y):
    mul1 = X.T.dot(X)
    inv1 = np.linalg.pinv(mul1)
    mul2 = X.T.dot(y)
    weight = np.matmul(inv1, mul2)
    return weight

def least_squares_predict(X, w):
    return np.matmul(X, w)
```

* **Evaluation:**
  We applied **10-fold cross-validation** using scikit-learn’s `KFold`, calculating:

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)

---

## 🤖 Multilayer Neural Network

We used **Keras Sequential** to build a neural network with:

* Four dense layers
* `relu` and `softmax` activations
* Optimizer: `adam`
* Loss: `mean_squared_error`
* Metrics: `mae`

We applied **K-Fold Cross-Validation** for performance evaluation.

---

## 📎 Project Files

* `main.py`: Main script for data processing and model execution
* `report.pdf`: Detailed report with screenshots, visualizations, and analysis
* `requirements.txt`: List of required Python libraries

---

## ✅ How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:

   ```bash
   python main.py
   ```

---

If you want, I can also generate the full `README.md` file in markdown format ready for copy-paste. Would you like me to prepare that for you?
