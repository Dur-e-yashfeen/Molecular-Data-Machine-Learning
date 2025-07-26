# 🧪 Molecular Data Machine Learning

> Predicting **molecular photostability (T80)** using advanced ensemble machine learning techniques on small, high-dimensional datasets.

**Author**: Dur-e Yashfeen
📊 [**Kaggle Profile**](https://www.kaggle.com/dureyashfeen)

---

## 📌 Problem Statement

In experimental chemistry, deriving conclusions from small datasets is a persistent challenge. This project aims to **predict the photostability (T80)** of molecules based on calculated descriptors. With only **42 training samples** and **\~144 features**, this problem is a real-world example of high-dimensional, low-sample-size (HDLSS) regression.

---

## 📁 Dataset

* `train.csv` – Contains features for 42 molecules with corresponding T80 target.
* `test.csv` – Contains features for unseen molecules (without T80).
* `sample_submission.csv` – Template for submission.

---

## 🧪 Project Pipeline

### ✅ Step 1: Importing Libraries

Includes `numpy`, `pandas`, `matplotlib`, `seaborn`, and ML libraries like `XGBoost`, `LightGBM`, and `Scikit-learn`.

### 📂 Step 2: Load the Data

```python
df_tr = pd.read_csv("/kaggle/input/molecular-machine-learning/train.csv")
df_ts = pd.read_csv("/kaggle/input/molecular-machine-learning/test.csv")
```

### 🔍 Step 3: Basic EDA

* Checked types, nulls, and stats.
* Found no missing values.
* `T80` distribution plotted.
* Top 30 features correlated with `T80` visualized using a heatmap.

### 🧼 Step 4: Outlier Handling

Outliers handled using **IQR-based median imputation** for each numeric feature (excluding the target `T80`).

### 📊 Step 5: Data Preprocessing

* Dropped non-numeric identifiers (`Batch_ID`, `Smiles`)
* Scaled features using `StandardScaler`
* Split into `X_train` / `X_val`
* Applied same transformation to test set

---

## 🤖 Model Training

Used a **Stacked Regressor** consisting of:

* **Base Learners**:

  * `XGBRegressor`
  * `LGBMRegressor`
* **Meta Learner**:

  * `RandomForestRegressor`

```python
stacked_model = StackingRegressor(
    estimators=[('xgb', xgb), ('lgb', lgb)],
    final_estimator=rf,
    n_jobs=-1
)
stacked_model.fit(X_train, y_train)
```

### 🧪 Model Evaluation

```python
Validation RMSE: 22.3121  
Validation R² Score: -0.1803
```

⚠️ **Note**: Negative R² suggests overfitting or insufficient generalization due to small sample size. This invites further exploration like:

* Feature selection
* Dimensionality reduction
* Simulated data augmentation
* Bayesian regression

---

## 📤 Submission

```python
submission_file["T80"] = stacked_model.predict(X_test_scaled)
submission_file.to_csv("submission.csv", index=False)
```

---

## 📈 Visual Insights

* **Target Distribution**
  ![T80 Histogram](https://i.imgur.com/n9kZGqU.png) *(example placeholder)*

* **Top Correlations**
  ![Correlation Heatmap](https://i.imgur.com/43k9CzW.png) *(example placeholder)*

---

## 🛠️ Future Improvements

* Apply PCA or Lasso-based dimensionality reduction
* Experiment with SVR, ElasticNet, and Bayesian Ridge
* Explore domain-specific molecular embeddings
* Use graph-based representations with GNNs

---

## 👩‍🔬 Domain Impact

This model can help:

* Discover novel photostable compounds
* Reduce experimental costs via virtual screening
* Reveal non-intuitive molecular patterns overlooked by chemists

---

## 🙋‍♀️ Author

**Dur-e Yashfeen**
🔗 [Kaggle Profile](https://www.kaggle.com/dureyashfeen)
