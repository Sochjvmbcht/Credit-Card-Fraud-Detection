# **Credit Card Fraud Detection**

This project aims to build machine learning models to detect fraudulent credit card transactions. The dataset is highly imbalanced, with fraudulent transactions making up a very small fraction of the total. Various models are implemented and evaluated to achieve the best accuracy in detecting fraud.

## **Dataset**

The dataset used for this project is the **Credit Card Fraud Detection** dataset from Kaggle. It contains transactions made by European cardholders in September 2013.

- **Link to Dataset**: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- The dataset contains 284,807 transactions, of which 492 are fraudulent.
- Features include anonymized transaction information such as time, amount, and PCA-derived components.

## **Project Structure**

1. **Read Data**:  
   Load the credit card transactions dataset and prepare it for exploration and analysis.

2. **Explore Data**:  
   Analyze the dataset to understand its structure, class distribution, and identify any anomalies or patterns.

3. **Data Preprocessing**:  
   - Handle missing values and inconsistencies.
   - Scale the feature values.
   - Encode categorical variables if necessary.

4. **Resampling Data**:  
   Due to class imbalance, apply techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset.

5. **Handle Outliers**:  
   Detect and address outliers in the data to improve the robustness of the models.

6. **Build Models**:  
   The following models were built and evaluated for fraud detection:
   - **Logistic Regression**
   - **Support Vector Machine (SVM)**
   - **Random Forest**
   - **XGBoost**
   - **Voting Classifier** (ensemble of the above models)

7. **Fine-Tuning**:  
   Hyperparameters for each model were fine-tuned to achieve optimal performance.

8. **Validation**:  
   Cross-validation techniques were used to evaluate model performance. Metrics like **Precision**, **Recall**, and **F1 Score** were emphasized due to the imbalanced nature of the dataset.

## **Results**

The accuracies of the various models are as follows:

| **Algorithm**             | **Accuracy**  |
|---------------------------|---------------|
| Logistic Regression        | 0.945946      |
| Support Vector Classifier  | 0.940541      |
| Random Forest              | 0.935135      |
| XGBoost Classifier         | 0.951351      |
| Voting Classifier          | 0.945946      |

The **XGBoost Classifier** achieved the highest accuracy at **0.951351**, making it the best-performing model for this task.

## **Technologies Used**

- Python (pandas, scikit-learn, XGBoost, etc.)
- Jupyter Notebook for development and exploration

