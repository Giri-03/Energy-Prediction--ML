# Energy-Prediction--ML

## CO₂ Emission Estimation using Machine Learning

### **AICTE – Edunet Internship | Energy Domain (Prediction)**

---

### 👨‍💻 **Author:**

**GIRI V**
AICTE – Edunet Internship 2025
Theme: **Energy (Prediction)**

---

### 🎯 **Objective**

To develop a **machine learning model** that estimates **carbon dioxide (CO₂) emissions** based on energy consumption data, enabling industries and policymakers to monitor and reduce greenhouse gas emissions.

---

### 💡 **Problem Statement**

Carbon emissions are one of the primary causes of **global warming**. Accurate estimation of CO₂ emissions based on energy data such as fuel type, GDP, and electricity production helps in **sustainable energy management** and environmental planning.

---

### 🧩 **Project Workflow**

1️⃣ **Data Collection**

* Sources: [Kaggle CO₂ Emission Dataset](https://www.kaggle.com/datasets/yoannboyere/co2-ghg-emissionsdata)
* [World Bank Energy Data](https://data.worldbank.org/topic/energy-and-mining)

2️⃣ **Data Preprocessing**

* Handle missing values
* Encode categorical variables
* Normalize numerical data

3️⃣ **Model Building**

* Algorithms: Linear Regression, Random Forest, XGBoost
* Metrics: MAE, RMSE, R² Score

4️⃣ **Visualization**

* Correlation heatmap
* Actual vs Predicted CO₂ graph
* Feature importance

5️⃣ **Deployment (Optional)**

* Streamlit/Flask-based web app for real-time CO₂ emission prediction

---

### 🧰 **Tools & Libraries**

| Purpose          | Library               |
| ---------------- | --------------------- |
| Data Handling    | Pandas, NumPy         |
| Visualization    | Matplotlib, Seaborn   |
| Machine Learning | Scikit-learn, XGBoost |
| App (optional)   | Flask / Streamlit     |
| Report           | Word / PDF            |

---

### 📁 **Repository Structure**

```
CO2_Emission_Estimation/
│
├── 📁 data/
│   ├── raw/                     # Original datasets (energy, emissions, etc.)
│   ├── processed/               # Cleaned & transformed data for model training 
│   └── README.md
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_visualization.ipynb
│
├── 📁 models/
│   ├── co2_model.pkl
│   ├── scaler.pkl
│   └── model_summary.txt
│
├── 📁 flask_app/
│   ├── app.py                   # Flask backend entry point
│   ├── templates/               # HTML templates (dashboard, upload page, results)
│   │   ├── index.html
│   │   ├── dashboard.html
│   │   └── prediction.html
│   ├── models/
│   |   ├── co2_model.pkl
│   |   ├── scaler.pkl
│   |   └── model_summary.txt
│   └── requirements.txt
│
├── 📁 reports/
│   ├── week_1_report.md
|   ├── week_2_report.md
|   ├── finalweek_report.md
│   ├── project_report.docx
│   ├── final_presentation.pptx
│
├── .gitignore
├── README.md
└── LICENSE
```

---

### 🗓️ **Weekly Progress**

#### 📄 Week 1 – Project Setup & Data Collection

* Finalized project title, objective, and workflow
* Collected and cleaned open-source datasets
* Performed basic exploratory data analysis (EDA)
* Saved cleaned dataset for future processing

#### 🖥️ Week 2

* Data preprocessing & feature engineering
* Correlation visualization

#### 🧠 Week 3

* Model training & evaluation

#### 🧾 Week 4

* Report, PPT, and optional deployment

---

### 📊 **Expected Output**

**Input Example:**
Energy_Consumption = 3500 PJ
GDP = 4.5 Trillion
Electricity_Production = 2200 TWh
Population = 1.4 Billion

**Predicted CO₂ Emission:** 2600 kilotons

---

### 🏁 **Final Outcome**

A predictive ML model that **estimates CO₂ emissions** using energy-related features — contributing to **sustainability, clean energy**, and **climate change awareness**.

---
