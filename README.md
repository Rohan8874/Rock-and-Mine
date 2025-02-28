# Sonar Rock and Mine Classification

A machine learning project to classify sonar signals as either rocks or mines using Logistic Regression.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Code Structure](#code-structure)
- [License](#license)

## Project Description
This project implements a Logistic Regression classifier to differentiate between sonar signals reflected from:
- **Mines (Metal cylinders)** - Labeled 'M'
- **Rocks** - Labeled 'R'

The model achieves **76.19% accuracy** on test data, demonstrating basic classification capability for underwater object detection using sonar frequency data.

## Dataset
**File:** `sonar data.csv`  
**Source:** UCI Machine Learning Repository  
**Characteristics:**
- 208 instances
- 60 numerical features (0.0-1.0 range)
- Binary classification (M/R)
- Class distribution:
  - Mines (M): 111 samples
  - Rocks (R): 97 samples

## Technologies Used
- Python 3.x
- NumPy
- pandas
- scikit-learn

## Installation
1. Clone the repository:
```sh
git clone https://github.com/yourusername/sonar-classification.git
cd sonar-classification
```
## Install required packages
```sh
pip install numpy pandas scikit-learn
```
## Usage
Place sonar data.csv in the project directory

## Run the Python script
```sh
python sonar_classification.py
```
## Results
## Sample Output
- Accuracy on training data:  0.8342245989304813
- Accuracy on test data:  0.7619047619047619
- Prediction: ['R']
- The object is a Rock

## Model Performance
- Dataset	Accuracy
- Training	83.42%
- Test	76.19%

## Confusion Matrix (Interpretation)
- True Positives (Mine detection): ~16
- True Negatives (Rock detection): ~15
- Total Test Samples: 21

# Code Structure
1. Data Loading and Exploration
sonar_data = pd.read_csv('sonar data.csv', header=None)

2. Data Preprocessing
X = sonar_data.drop(60, axis=1)
Y = sonar_data[60]

3. Train-Test Split (90-10)
X_train, X_test, Y_train, Y_test = train_test_split(...)

4. Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

5. Evaluation
train_acc = accuracy_score(...)
test_acc = accuracy_score(...)

6. Prediction System
input_data = (...)  # 60 feature values
prediction = model.predict(...)

## License
