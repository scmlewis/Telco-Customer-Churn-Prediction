![image](https://github.com/user-attachments/assets/0bffb9d4-a14a-4da0-b756-e3775a0bb658)


# Telco Customer Churn Prediction

##  Overview

This project, developed for the ISOM5535 High Dimensional Statistics with Business Applications course at The Hong Kong University of Science and Technology (Spring 2025), analyzes customer churn in the telecommunications industry using the Telco Customer Churn dataset. The objective is to predict churn and identify key drivers to inform retention strategies, leveraging high-dimensional statistical methods. The analysis employs Principal Component Analysis (PCA) for dimension reduction and classification techniques (K-Nearest Neighbors, Random Forest, Logistic Regression) to build predictive models.

##  Key Findings
Exploratory Data Analysis (EDA): The dataset includes 7,043 customers with a 26.54% churn rate. High churn is observed for month-to-month contracts (42.71%), fiber optic users (41.89%), electronic check payers (45.29%), and senior citizens (41.68%).
PCA: PC1 explains 39.21% of variance, driven by MonthlyCharges, TotalCharges, and tenure; PC2 explains 18.68% (cumulative 57.89%), influenced by tenure and Contract_Two year.
Classification: Logistic Regression achieved the highest accuracy (0.82), followed by Random Forest (0.79) and KNN (0.80). Random Forest identifies TotalCharges, tenure, and MonthlyCharges as top predictors.
Business Implications: Recommendations include loyalty programs for short-tenure customers, incentives for longer-term contracts, fiber optic service improvements, and streamlined payment options to reduce churn.

## Results and Business Implications
The analysis provides actionable insights for reducing churn:

High-Risk Groups:
Month-to-month contracts: 42.71% churn vs. 11.27% (one-year) and 2.83% (two-year).
Fiber optic users: 41.89% churn vs. 18.96% (DSL).
Electronic check payers: 45.29% churn vs. 15.24% (credit card).
Senior citizens: 41.68% churn vs. 23.61% (non-seniors).


Predictive Models:
Logistic Regression: 0.82 accuracy, 0.60 sensitivity, 0.75 AUC.
Random Forest: 0.79 accuracy, key predictors: TotalCharges, tenure, MonthlyCharges.
KNN (k=10): 0.80 accuracy.


Recommendations:
Implement loyalty programs for customers with <12 months tenure.
Offer incentives for one/two-year contracts (e.g., reduced rates).
Enhance fiber optic service quality or provide bundled discounts.
Promote alternative payment methods to replace electronic checks.


