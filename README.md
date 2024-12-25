# Detecting Fraudulent Healthcare Providers Using Machine Learning

**Data 606 Capstone Project in Data Science**  
Under the guidance of Zeynep Kacar  
By
**Amrita Chandrasekar (NH53017)**  
Date: 12/11/2024  

## Table of Contents
- [Executive Summary](#executive-summary)
- [Introduction](#introduction)
- [Research Questions](#research-questions)
- [Background and Context](#background-and-context)
- [Objectives](#objectives)
- [Data Collection and Description](#data-collection-and-description)
- [Data Cleaning](#data-cleaning)
- [Challenges](#challenges)
- [Methodology](#methodology)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Modeling and Analysis](#modeling-and-analysis)
  - [Models Comparison](#models-comparison)
- [Streamlit Dashboard](#streamlit-dashboard)
  - [Key Features](#key-features)
  - [Building the Dashboard](#building-the-dashboard)
  - [Deployment](#deployment)
- [Discussion](#discussion)
- [Conclusions and Recommendations](#conclusions-and-recommendations)
- [Future Work](#future-work)
- [References](#references)
- [Project links](#project-links)

---

## Executive Summary
The project Detecting Fraudulent Healthcare Providers Using Machine Learning seeks to address the pervasive issue of fraudulent activities within Medicare claims data using advanced machine learning techniques.Healthcare fraud, including practices like inflated billing and falsified services, is a major concern that leads to financial losses and undermines the integrity of the healthcare system. By applying various machine learning models such as Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and LightGBM, the project identifies patterns indicative of fraud in the claims data. Additionally, the class imbalance in the dataset is mitigated using SMOTE (Synthetic Minority Over-sampling Technique), which enhances the model’s ability to accurately detect fraudulent claims. The project also features an interactive Streamlit dashboard, which allows healthcare administrators to conduct visualize key insights, predict fraud/non-fraud facilitating more informed decision-making and improving overall fraud prevention efforts.

### Key Findings
- **LightGBM** achieved 98% accuracy, demonstrating its strong ability to detect fraudulent providers with minimal bias.
- Important features such as **ProviderFraudRate** and **ClaimsPerProvider** were identified as key indicators of fraud.
- **SMOTE** significantly enhanced model performance by addressing class imbalance in the dataset.
- The **Streamlit dashboard** provided an intuitive interface for fraud prediction and decision-making.

---

## Introduction

Healthcare fraud, especially within Medicare, leads to significant financial losses and affects the quality of care. Traditional fraud detection methods are ineffective due to the high volume and complexity of claims data. This project applies **machine learning algorithms** to automate fraud detection, offering a scalable and efficient solution. An interactive **Streamlit dashboard** was built to visualize fraud patterns and allow  predictions, enabling healthcare administrators to make informed decisions swiftly.

---

## Problem Statement
In the U.S. healthcare system, fraud costs taxpayers billions of dollars every year. A real-world example is the case of a healthcare provider who submits fraudulent Medicare claims for services that were never provided, such as billing for unnecessary medical tests or overcharging for treatments. In 2019, it was reported that healthcare fraud led to losses of over $60 billion in the U.S. alone. This fraudulent behavior not only strains the financial stability of healthcare systems but also puts patient safety at risk, as resources are diverted away from legitimate care.

Traditional fraud detection methods, which often rely on manual audits and rule-based checks, struggle to keep up with the sheer volume and complexity of claims data, making it difficult to identify subtle patterns of fraud. This project aims to harness machine learning to automate the process, providing a more efficient, scalable, and accurate solution for detecting fraudulent healthcare providers, ultimately reducing financial losses and ensuring the quality and integrity of care provided to patients.

---

## Research Questions

- **How effectively can machine learning algorithms identify potentially fraudulent Medicare providers based on claims data?**
- **Which features in the dataset are most indicative of fraudulent behavior?**
- **How can we address the class imbalance in the dataset, where fraudulent claims are far fewer than legitimate ones?**

---

## Background and Context

Healthcare fraud is a pervasive problem that drains financial resources and jeopardizes patient care. The challenge lies in identifying subtle fraud patterns in massive datasets. By applying machine learning, particularly **supervised learning models**, this project aims to create a system that can efficiently predict fraudulent claims. The dataset used includes **inpatient**, **outpatient**, and **beneficiary** data, each containing features related to healthcare providers, services rendered, and claims.

---

## Objectives
- **Detect fraudulent healthcare providers** using machine learning algorithms.
- **Build an interactive Streamlit dashboard** for predictions and data visualizations.

---

## Data Collection and Description

The dataset was sourced from **Kaggle** and includes a range of claims data:

- **Inpatient Data**: Claims for patients admitted to hospitals, including admission/discharge dates and diagnosis codes.
- **Outpatient Data**: Claims for patients who received outpatient services.
- **Beneficiary Data**: Demographic and health condition information for each Medicare beneficiary.
- **Fraud Labels**: Labeled data indicating whether a provider is fraudulent or not.

Dataset link: [Healthcare Provider Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data)

## Dataset
The dataset for this project is too large to be uploaded directly to GitHub. You can access the dataset via the following links:
- [Google Drive Link to Dataset] (https://drive.google.com/drive/folders/1MenvqMRqx9gnJM-hmXxuovUQJZ4HSkFx?usp=sharing))

---

## Data Cleaning

- Dropping columns that have more than 50% null values for inpatient, outpatient and beneficiary datasets individually.
- Merging “inpatient” and “outpatient” as “merged_data” with the help of outer join function. Then merging “merged_data” with beneficiary dataset on BeneID column with outer join function as “final merged” dataset.
- Shape of Final Merged Dataset: (558211, 46)
- Now combining “final merged” dataset with “train dataset” on provider column with the help of inner join as “df_final”
- Shape of df_final dataset: (558211, 47).
- Removing columns from df_final that have more than 50%null values as they are not part of analysis.
- Shape of df_final dataset:(558211, 34)
- Checking for duplicate values
- Checking for null values: AttendingPhysician 1508, DeductibleAmtPaid 899, ClmDiagnosisCode_1 10453, ClmDiagnosisCode_2 195606
- To handle null values, “unknown” is used in place of null values for the column “AttendingPhysician” while “median” and “mode” is used for columns “DeductibleAmtPaid”,”ClmDiagnosisCode_1” and “ClmDiagnosisCode_2” respectively.
- Converting date to datetime format
- Performed label encoding to convert categorical features to numerical features for making data ready for visualizations

---

## Challenges

**Class Imbalance**
Fraudulent claims are a minority class, leading to biased models.

<img width="274" alt="image" src="https://github.com/user-attachments/assets/ba7e6c0a-d05b-4314-9cdd-cc3a1af1dc35">

SMOTE is used to handle class imbalance. SMOTE (Synthetic Minority Over-sampling Technique) is a technique used to address class imbalance by generating synthetic samples for the minority class. It works by selecting a sample from the minority class, finding its nearest neighbors, and then creating new samples by interpolating between the original sample and its neighbors. This helps balance the dataset, preventing the model from being biased toward the majority class.After SMOTE, combined features (X_resampled) and target (y_resampled) into a single data frame as “balanced_df”  having shape (690830, 32).

<img width="206" alt="image" src="https://github.com/user-attachments/assets/c6fb50fe-1dbc-4f94-8a27-cf7928c99f6f">



---

## Methodology

### Exploratory Data Analysis (EDA)
EDA was conducted to understand the distribution of features and identify important patterns. Visualization techniques were used to explore class distribution and correlations between features.

![Image 1](https://github.com/user-attachments/assets/e3743aad-69e8-4ac8-94f2-ecab47a6a223)
![Image 2](https://github.com/user-attachments/assets/51634151-e4b7-4243-91e9-c9c1d4620a9c)
![Image 3](https://github.com/user-attachments/assets/983764b7-81a1-4107-a4f6-c0b7bf0a83a4)
![Image 4](https://github.com/user-attachments/assets/9743773f-904f-44d1-81a6-4445b5dfc51a)
![Image 5](https://github.com/user-attachments/assets/b2d027c9-b623-49b9-9885-59d593ed8d3b)
![Image 6](https://github.com/user-attachments/assets/30402ebb-1284-487f-9e65-edb65445106b)


### Feature Engineering
Key features such as **ProviderFraudRate** and **ClaimsPerProvider** were engineered based on domain knowledge and the dataset's characteristics.

<img width="436" alt="image" src="https://github.com/user-attachments/assets/3022003a-3fc2-4f6c-b48c-a88312715212">

### Feature Correlations with Potential Fraud

| Feature                  | Correlation with Potential Fraud | Description |
|--------------------------|----------------------------------|-------------|
| **CostPerDay**            | 0.11                             | Moderately correlated with potential fraud. Can help identify unusually high costs per day for providers, which may indicate fraud. |
| **DeductibleRatio**       | Low (insignificant)              | Very low correlation. Less impactful for fraud detection, but may provide context on healthcare cost structures. |
| **ChronicConditionCount** | -0.10                            | Weak negative correlation. Suggests that chronic conditions might not strongly indicate fraudulent behavior but can help assess patient risk. |
| **GenderCostRatio**       | 0.09                             | Weak positive correlation. May help detect anomalies in cost allocation across genders. |
| **ClaimsPerProvider**     | 0.33                             | Strong correlation. A higher number of claims per provider could signal fraudulent activity. |
| **AvgProviderReimbursement** | 0.21                          | Relevant for detecting discrepancies between expected and actual reimbursements, indicating potential fraud. |
| **ProviderFraudRate**     | 0.85                             | Highly correlated. One of the most critical features for detecting fraudulent behavior in healthcare providers. |
| **CostPerCoverageMonth**  | 0.74                             | Strong correlation. Unusually high costs per coverage month are a strong indicator of fraudulent activity. |
| **DurationReimbursement** | 0.05                             | Weak correlation. Might not be highly indicative of fraud but still valuable in broader fraud detection models. |
| **StateFraudRate**        | 0.28                             | Moderate correlation. Provides geographic context, which may be useful for understanding state-level fraud trends. |
| **PotentialFraud**        | 1.0                              | The target variable. Its correlation with itself is 1, making it central to the analysis. |

### Key Insights
- **ProviderFraudRate** and **CostPerCoverageMonth** are among the most significant features for fraud detection, offering strong predictive power.
- **ClaimsPerProvider** and **AvgProviderReimbursement** also play an important role in identifying discrepancies and anomalies that may indicate fraudulent behavior.
- Features like **StateFraudRate** and **ChronicConditionCount** provide additional context and could be useful in broader fraud detection strategies.

<img width="356" alt="image" src="https://github.com/user-attachments/assets/b28d1cd4-fd73-4c51-833d-b946d6cc7c41">

### Feature Importance Analysis
The process begins by preparing the data, where categorical variables are one-hot encoded, and the dataset is split into training and testing sets. The presence of infinite and NaN values is checked and handled by replacing infinite values with the column maximum. A Random Forest model is then trained on the data, and feature importance is computed to identify the top features most strongly correlated with the target variable "PotentialFraud.The top features are displayed to guide model implementation. This analysis is essential for further model development and integration into a Streamlit dashboard for interactive fraud detection.

<img width="356" alt="image" src="https://github.com/user-attachments/assets/12926589-f2fc-4788-98b8-9c24eedd67b0">


### Modeling and Analysis
Several machine learning models were trained, including Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and LightGBM. **LightGBM** was chosen for its excellent performance, especially in handling imbalanced data.
### Logistic Regression
<img width="188" alt="image" src="https://github.com/user-attachments/assets/81f4973d-231d-4d93-9236-c755c6ea3f9a">

### Random Forest
<img width="183" alt="image" src="https://github.com/user-attachments/assets/29acaefe-f2da-4067-b426-2c8cf2aa78d1">

### XGBoost
<img width="170" alt="image" src="https://github.com/user-attachments/assets/ad07376a-b9c7-492d-bd82-8e1455875add">

### LightGBM
<img width="196" alt="image" src="https://github.com/user-attachments/assets/df1861ad-0335-4a35-bb72-219ceca08972">

### Gradient Boosting
<img width="194" alt="image" src="https://github.com/user-attachments/assets/0c489e76-5676-4a43-bde3-f2528391a807">


## Models Comparison
<img width="500" alt="image" src="https://github.com/user-attachments/assets/d80d0312-2bfe-488e-82af-4fc046e68c2f">
<img width="286" alt="image" src="https://github.com/user-attachments/assets/846bea8a-8765-4640-843d-0dcfdc23ba06">

- **LightGBM** demonstrated the highest performance across multiple metrics:
  - **Recall**: 97.26%
  - **F1-Score**: 0.9832
  - **AUC**: 0.9981
- LightGBM is particularly effective for fraud detection due to its ability to handle class imbalance and its strong generalization performance.


## Streamlit Dashboard

### Key Features
- **Fraud predictions**: Users can input data to get immediate predictions on whether a healthcare provider is fraudulent.
- **Visualizations**
  - **Fraud vs. Non-Fraud distribution**
  - **State-wise fraud rates**
  - **Correlation heatmap** for key features
- **User-Friendly Interface**: The dashboard is intuitive and requires no coding experience.

### Building the Dashboard
- **Data Integration**: Data was preprocessed, merged, and ready for integration into Streamlit.
- **Model Integration**: The trained LightGBM model was incorporated using **joblib** for predictions.

### Deployment
The Streamlit app was deployed on the **Streamlit Community Cloud**, making it easily accessible via a web link.

---

## Discussion

### Interpretation of Results
- The LightGBM model’s high performance (98% accuracy) and the use of **SMOTE** for class balancing were pivotal in achieving strong results.
- The **Streamlit dashboard** enhanced the model's usability, providing healthcare professionals with quick insights.

### Comparison with Existing Literature
- The findings align with other studies on healthcare fraud detection, which emphasize the importance of machine learning and handling class imbalance.

### Unexpected Findings
- The significance of **ProviderFraudRate** in detecting fraud was unexpected and highlights the value of feature engineering.

---

## Conclusion
This project lays the foundation for an efficient, automated healthcare fraud detection system, combining machine learning, data engineering, and interactive tools. With future enhancements, it has the potential to drastically reduce fraud, optimize healthcare resources, and improve overall healthcare delivery.

## Results and Insights

### 1. Model Performance and Accuracy
- **LightGBM Outperforms Other Models**
  The LightGBM model achieved an overall **accuracy of 98%**, with a balanced **F1-score of 0.98**, indicating strong predictive capabilities for both fraudulent and non-fraudulent healthcare providers. This suggests that LightGBM is highly reliable for detecting fraud with minimal bias, making it well-suited for real-world deployment in fraud detection systems.

- **Key Metrics**
  - **Recall**: The LightGBM model achieved a **test recall of 0.9726**, ensuring that the majority of fraudulent claims were correctly identified. Recall is crucial in fraud detection, where false negatives can be costly.
  - **AUC**: The **AUC score of 0.9981** reflects the model's exceptional ability to distinguish between fraud and non-fraud cases, providing a highly reliable fraud detection mechanism.

### 2. Addressing Class Imbalance
- **Impact of SMOTE on Performance**:  
  The application of **SMOTE** (Synthetic Minority Over-sampling Technique) successfully addressed the class imbalance issue, where fraudulent claims are far fewer than legitimate claims. By generating synthetic samples for the minority class (fraudulent claims), SMOTE helped the model identify fraudulent behavior more effectively, leading to improved recall and F1-scores.

- **Balanced Dataset**  
  After applying SMOTE, the dataset became more balanced, preventing the model from being biased towards the majority class (non-fraudulent claims). This ensures that the model can detect fraud with higher sensitivity.

### 3. Feature Importance and Insights
- **Critical Fraud Indicators**
  Feature engineering revealed that certain features, particularly **ProviderFraudRate** (0.85 correlation with fraud), **ClaimsPerProvider** (0.33 correlation with fraud), and **CostPerCoverageMonth** (0.74 correlation with fraud), are the strongest indicators of fraudulent behavior.

  - **ProviderFraudRate**: A high **ProviderFraudRate** indicates that a provider has a history of submitting fraudulent claims, making it a key indicator for fraud detection.
  - **ClaimsPerProvider**: A higher number of claims per provider strongly correlates with fraud, suggesting that fraudsters may submit more claims to exploit the system.
  - **CostPerCoverageMonth**: Unusually high costs per coverage month were identified as an indicator of fraudulent billing practices.

### 4. Insights from Streamlit Dashboard
- **Real-Time Fraud Prediction**
  The **Streamlit dashboard** enhanced the practical application of the fraud detection model by allowing healthcare administrators to interactively assess fraud risks. The dashboard provides real-time predictions on whether a healthcare provider is potentially fraudulent based on the data entered.

  - **User Interaction**: Healthcare professionals can easily input specific provider data and receive instant predictions, helping them identify suspicious providers swiftly.
  - **Data Visualizations**: The dashboard displays key visualizations like fraud distribution by state, feature importance, and fraud vs. non-fraud counts. These visualizations help stakeholders better understand fraud patterns and make data-driven decisions in a user-friendly format.

### 5. Practical and Financial Impact
- **Cost Reduction**
  By detecting fraudulent healthcare providers early, the model can prevent financial losses from fraudulent claims. The **98% accuracy** ensures that fraudulent claims are flagged with high reliability, allowing for quicker investigations and resource allocation, which could result in significant cost savings for healthcare systems.

- **Operational Efficiency** 
  The implementation of the machine learning model and the Streamlit dashboard enables **automated fraud detection**, reducing the need for manual claim reviews. This leads to faster identification of fraudulent claims, enhancing operational efficiency within healthcare organizations.

## Future Scope

- **Continuous Monitoring and Real-Time Integration:**
  The model's ability to predict fraud in real-time, through integration into a healthcare provider’s claims system, could enable **continuous fraud monitoring**, preventing fraudulent activities before they impact the healthcare system.

- **False Positive Management:** 
  While the model is highly accurate, **false positives** still remain a challenge. Introducing post-prediction analysis and a **scoring system** to prioritize fraudulent claims based on severity could help reduce unnecessary investigations and optimize resource allocation.

- **Anomaly Detection:**
  Implement **unsupervised learning models** to detect emerging or unknown fraud patterns that may not be captured by supervised models.

---

## **References**
1. **Bauder, R.A., & Khoshgoftaar, T.M. (2017)**. *Medicare fraud detection using machine learning methods.* IEEE ICMLA.
2. **Garmdareh, M.S., et al. (2023)**. *A Machine Learning-based Approach for Medical Insurance Anomaly Detection by Predicting Indirect Outpatients' Claim Price.* IEEE ICWR.

---

### Streamlit App
- [Interactive Streamlit App for Visualization](https://amritachandrasekar18-data-690-capstone-project-app-pwbhji.streamlit.app/)

