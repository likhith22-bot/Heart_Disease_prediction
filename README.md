# ❤️ Heart Disease Prediction Using Machine Learning

A machine learning-based web application that predicts heart disease using selected clinical features. The project performs data preprocessing, feature transformation, outlier handling, feature selection, class balancing, scaling, hyperparameter tuning, and prediction using a **Gaussian Naive Bayes** classifier.

The trained model is integrated with a **Flask web application** that provides a user-friendly interface for entering patient information and generating predictions.

---

## 📌 Project Overview

Heart disease is one of the major health-related problems worldwide. Machine learning can be used to analyze clinical data and identify patterns that can support prediction tasks.

This project develops an end-to-end machine learning pipeline:

```text
Heart Disease Dataset
        ↓
Data Cleaning
        ↓
Train/Test Split
        ↓
Yeo-Johnson Transformation
        ↓
Outlier Handling
        ↓
Feature Selection
        ↓
SMOTE
        ↓
StandardScaler
        ↓
GridSearchCV
        ↓
Gaussian Naive Bayes
        ↓
Model Evaluation
        ↓
Flask Web Application
        ↓
Prediction
```

---

## 🚀 Features

* Data preprocessing and cleaning
* Missing-value analysis
* Train-test splitting
* Yeo-Johnson transformation
* Outlier handling using trimming
* Feature selection
* Class balancing using SMOTE
* Feature scaling using StandardScaler
* Gaussian Naive Bayes classification
* Hyperparameter tuning using GridSearchCV
* Model serialization using Pickle
* Flask-based web application
* Interactive HTML frontend
* Render deployment support

---

## 🧠 Machine Learning Model

### Gaussian Naive Bayes

The project uses **Gaussian Naive Bayes (GaussianNB)** as the final classification algorithm.

Gaussian Naive Bayes is suitable for classification problems where the features can be modeled using Gaussian distributions.

The model estimates the probability of each class based on the input features and selects the class with the highest posterior probability.

---

## 🎯 Selected Features

After preprocessing and feature selection, the model uses seven independent features:

| Feature   | Description                             |
| --------- | --------------------------------------- |
| `age`     | Patient's age                           |
| `sex`     | Patient's sex encoded numerically       |
| `cp`      | Chest pain type                         |
| `thalach` | Maximum heart rate achieved             |
| `oldpeak` | Exercise-induced ST depression          |
| `slope`   | Slope of peak exercise ST segment       |
| `thal`    | Encoded thalassemia-related measurement |

The prediction input follows this exact order:

```python
[
    age,
    sex,
    cp,
    thalach,
    oldpeak,
    slope,
    thal
]
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Imbalanced-learn
* NumPy
* Pandas

### Data Visualization

* Matplotlib
* Seaborn

### Web Development

* Flask
* HTML
* CSS

### Model Persistence

* Pickle

### Deployment

* Render

---

## 📂 Project Structure

```text
Heart-Disease/
│
├── app.py
├── main.py
├── heart.csv
│
├── final_model.pkl
├── scaling.pkl
│
├── varibale_transformation.py
├── training_models.py
├── fs.py
├── hyperparameter_tuning.py
├── loge_code.py
│
├── templates/
│   └── index.html
│
├── logs/
│   └── main.log
│
├── requirements.txt
│
└── README.md
```

---

## 🔄 Data Processing Pipeline

### 1. Load Dataset

```python
import pandas as pd

df = pd.read_csv("heart.csv")

print(df.shape)
print(df.isnull().sum())
```

---

### 2. Separate Features and Target

```python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```

---

### 3. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 4. Yeo-Johnson Transformation

Yeo-Johnson transformation is applied during preprocessing to transform feature distributions.

```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method="yeo-johnson")

X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)
```

The transformation is fitted using the training data and then applied to the test data.

---

### 5. Handle Class Imbalance

SMOTE is used to balance the training data.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)
```

---

### 6. Feature Scaling

StandardScaler is used to standardize the features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler is saved for later use during web-app prediction:

```python
import pickle

with open("scaling.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

---

## 🔍 Hyperparameter Tuning

GridSearchCV is used to find suitable parameters for Gaussian Naive Bayes.

Example:

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

model = GaussianNB()

param_grid = {
    "var_smoothing": [
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2
    ]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
```

The best estimator can then be saved:

```python
best_model = grid_search.best_estimator_

with open("final_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
```

---

## 🌐 Flask Web Application

The trained model and scaler are loaded into the Flask application.

```python
from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

with open("scaling.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)
```

The application receives the seven values from the HTML form:

```python
age = float(request.form["age"])
sex = float(request.form["sex"])
cp = float(request.form["cp"])
thalach = float(request.form["thalach"])
oldpeak = float(request.form["oldpeak"])
slope = float(request.form["slope"])
thal = float(request.form["thal"])
```

The values are arranged in the same order used during model training:

```python
values = np.array([[
    age,
    sex,
    cp,
    thalach,
    oldpeak,
    slope,
    thal
]])
```

The scaler and model are then used:

```python
scaled_values = scaler.transform(values)

prediction = model.predict(scaled_values)[0]
```

---

## 🖥️ Web Interface

The Flask application provides a web interface where users can enter:

```text
Age
Sex
Chest Pain Type
Maximum Heart Rate
Oldpeak
ST Segment Slope
Thal
```

After clicking:

```text
Predict Heart Disease
```

the trained machine learning model processes the input and displays the prediction.

---

## 📊 Model Evaluation

The model can be evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example:

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)
```

---

## 💾 Saved Model Files

The project saves two important files:

### `scaling.pkl`

Contains the fitted `StandardScaler`.

It is used to transform new input data before sending it to the model.

### `final_model.pkl`

Contains the final trained Gaussian Naive Bayes model selected through hyperparameter tuning.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project

```bash
cd Heart-Disease
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Required Libraries

Your `requirements.txt` can contain:

```text
Flask
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

If you are deploying on Render, make sure the required packages are included in `requirements.txt`.

---

## ▶️ Run the Application Locally

Start the Flask application:

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000/
```

Open the address in your browser and enter the patient information.

---

## ☁️ Deployment

This project can be deployed as a Flask web application using **Render**.

### Deployment Steps

1. Push the project to GitHub.
2. Create a new Web Service on Render.
3. Connect your GitHub repository.
4. Install dependencies using:

```bash
pip install -r requirements.txt
```

5. Configure the start command according to your Flask deployment setup.
6. Deploy the application.
7. Add the generated Render URL below.

---

# 🔗 Project Links

### 🚀 Live Demo

**Render Deployment:**
👉 **[ADD YOUR RENDER DEPLOYMENT LINK HERE]**

```text
https://your-project-name.onrender.com
```

### 💻 GitHub

**GitHub Repository:**
👉 **[https://github.com/likhith22-bot/Heart_Disease_prediction]**

---

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes**. The machine learning prediction should not be considered a medical diagnosis or a substitute for professional medical advice.

---

## 👨‍💻 Author

**Likhith Naga Sai Tadikonda**

Computer Science & Engineering | Java Full Stack & Machine Learning Enthusiast

LinkedIn: **[https://www.linkedin.com/in/likhith-naga-sai-tadikonda-a25a22316/]**

