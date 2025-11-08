Week 2 – Data Preprocessing & Feature Engineering

🎯 Objective

To preprocess the dataset, encode categorical variables, normalize numerical features, and prepare data for model training and evaluation.

| **Task No** | **Task Description**                                                                                                      | **Status**  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------- | ----------- |
| 1           | Loaded the cleaned dataset (`co2_energy_data.csv`) for preprocessing                                                      | ✅ Completed |
| 2           | Checked for categorical and numerical features                                                                            | ✅ Completed |
| 3           | Encoded categorical variables (`Country`) using **Label Encoding**                                                        | ✅ Completed |
| 4           | Normalized numerical columns (`Energy_Consumption`, `GDP`, `Population`, `Electricity_Production`) using **MinMaxScaler** | ✅ Completed |
| 5           | Split dataset into **training (80%)** and **testing (20%)** sets using `train_test_split()`                               | ✅ Completed |
| 6           | Generated **correlation heatmap** using Seaborn to analyze feature relationships                                          | ✅ Completed |
| 7           | Saved processed dataset as `co2_energy_preprocessed.csv`                                                                  | ✅ Completed |

| **Feature Name**       | **Transformation Applied**          |
| ---------------------- | ----------------------------------- |
| Country                | Label Encoded                       |
| Energy_Consumption     | Normalized                          |
| GDP                    | Normalized                          |
| Population             | Normalized                          |
| Electricity_Production | Normalized                          |
| CO2_Emissions          | Target Variable (No transformation) |

🔍 Key Findings from Week 2

Strong positive correlation observed between Energy_Consumption and CO₂_Emissions

GDP also shows moderate correlation, indicating economic activity’s impact on emissions

Feature normalization improved uniformity and reduced scale bias

Dataset is now clean, consistent, and ready for model building in Week 3

🧰 Tools & Libraries Used

Python 3.10+

Pandas, NumPy – Data Preprocessing

Scikit-learn – Label Encoding, Normalization, Train-Test Split

Matplotlib, Seaborn – Heatmap Visualization

Jupyter Notebook – Implementation & Documentation

🚀 Next Week (Week 3) Goals

Train machine learning regression models (Linear, Random Forest, XGBoost)

Evaluate performance using MAE, RMSE, and R² metrics

Visualize model comparison results
