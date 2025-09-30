Titanic - Machine Learning from Disaster
Project Overview
This project implements a machine learning solution for the classic Kaggle Titanic dataset, predicting passenger survival using Random Forest classification.
Dataset
The project uses the Titanic dataset which includes passenger information such as:

Passenger demographics (Age, Sex, Class)
Family information (SibSp, Parch)
Ticket and fare information
Embarkation port

Project Structure
├── data/
│   ├── train.csv          # Training dataset
│   ├── test.csv           # Test dataset
│   └── predictions.csv    # Model predictions
├── Titanic_Machine_Learning_from_Disaster.ipynb
└── README.md
Methodology

1. Data Preprocessing

Handling Missing Values: Age values imputed using mean strategy
Feature Engineering:

One-hot encoding for categorical variables (Embarked, Sex)
Missing Embarked values filled with 'S' (Southampton)


Feature Selection: Removed non-predictive columns (Name, Ticket, Cabin)
Standardization: Applied StandardScaler to normalize features

2.  Data Splitting

Used StratifiedShuffleSplit for train-test split (80-20)
Stratified by Survived, Pclass, and Sex to maintain class distribution

3. Model Development

Algorithm: Random Forest Classifier
Hyperparameter Tuning: GridSearchCV with 3-fold cross-validation

n_estimators: [10, 100, 200, 500]
max_depth: [None, 5, 10]
min_samples_split: [2, 3, 4]



4. Pipeline Architecture
Custom preprocessing pipeline with three stages:

AgeImputer: Fills missing age values
FeatureEncoder: Encodes categorical variables
FeatureDropper: Removes unnecessary features

Results

Test Set Accuracy: ~81.6%
Best Model Parameters:

n_estimators: 500
max_depth: 10
min_samples_split: 4



Key Features

Correlation analysis visualization using heatmap
Stratified sampling to ensure representative train-test split
Automated feature engineering pipeline
Cross-validated hyperparameter optimization

Technologies Used

Python 3.x
Libraries:

pandas - Data manipulation
numpy - Numerical operations
scikit-learn - Machine learning models and preprocessing
matplotlib & seaborn - Data visualization



How to Run

Install required dependencies:

bashpip install pandas numpy scikit-learn matplotlib seaborn

Place the dataset files in the data/ directory
Run the Jupyter notebook:

bashjupyter notebook Titanic_Machine_Learning_from_Disaster.ipynb
Output
The model generates predictions saved to predictions.csv with PassengerId and predicted Survived status for the test set.
Future Improvements

Feature engineering (e.g., family size, title extraction from names)
Ensemble methods combining multiple algorithms
Deep learning approaches
Feature importance analysis

License
This project is part of the Kaggle Titanic competition.