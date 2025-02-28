# Sonar Rock vs Mine Classification

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
```bash
git clone https://github.com/yourusername/sonar-classification.git
cd sonar-classification
