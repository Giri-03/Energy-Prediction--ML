Week 1 – Project Initialization & Data Collection
🎯 Objective

To develop a machine learning model that estimates CO₂ emissions based on energy consumption data, helping industries and policymakers monitor and reduce greenhouse gas emissions.

| Task No | Task Description                                                                                                                       | Status      |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| 1       | Finalized project title and objective                                                                                                  | ✅ Completed |
| 2       | Studied the problem statement and workflow                                                                                             | ✅ Completed |
| 3       | Researched and selected open datasets from **Kaggle** and **World Bank**                                                               | ✅ Completed |
| 4       | Collected datasets containing: `Country`, `Year`, `Energy_Consumption`, `GDP`, `Population`, `Electricity_Production`, `CO2_Emissions` | ✅ Completed |
| 5       | Imported dataset into Jupyter Notebook using **Pandas**                                                                                | ✅ Completed |
| 6       | Performed **basic data exploration** (`.head()`, `.info()`, `.describe()`)                                                             | ✅ Completed |
| 7       | Handled **missing values** and removed **duplicates**                                                                                  | ✅ Completed |
| 8       | Saved cleaned dataset as `co2_energy_data.csv`                                                                                         | ✅ Completed |

| Column                 | Description                           |
| ---------------------- | ------------------------------------- |
| Country                | Name of the country                   |
| Year                   | Year of record                        |
| Energy_Consumption     | Total energy consumption (PJ)         |
| GDP                    | Gross Domestic Product (Trillion USD) |
| Population             | Total population (Billions)           |
| Electricity_Production | Total electricity generated (TWh)     |
| CO2_Emissions          | Recorded CO₂ emissions (kilotons)     |

🔍 Exploratory Data Analysis (EDA)

Checked data shape and column details
Verified data types and handled missing values
Analyzed summary statistics to understand value distribution
Identified potential relationships between energy consumption, GDP, and CO₂ emissions

🧠 Insights from Week 1

Energy consumption and GDP appear strongly correlated with CO₂ emissions
Some countries have missing or inconsistent energy data; handled via mean imputation
Dataset ready for feature engineering and model training in Week 2

🧰 Tools & Libraries Used
Python 3.10+
Pandas, NumPy – Data Handling
Matplotlib, Seaborn – Data Visualization
Jupyter Notebook – Analysis Environment

🚀 Next Week (Week 2) Goals
Perform data preprocessing (encoding categorical features, normalization)
Split data into training & testing sets
Generate correlation heatmap for feature analysis
