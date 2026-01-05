# AutoJudge — Programming Problem Difficulty Prediction

## Project Overview
AutoJudge predicts the difficulty of a programming problem using only its textual data. 

The model performs two tasks:
- Classification: Predicts the difficulty of the problem (Easy/Medium/Hard)
- Regression: Predicts a numerical difficulty score on a scale of 1-10

This helps automatically label a problem by difficulty and recommend similar problems to users.

---
## Dataset Used
The project uses the provided dataset (problems_data.csv)

Each sample contains the following fields:
  - Title
  - Problem description
  - Input description
  - Output description
  - Sample input/output
  - Problem Class (Easy/Medium/Hard)
  - Problem Score (between 1-10)
  - url

All text fields except the url are then combined in a single text column for feature extraction

---
## Approach and Models Used

### Data Preprocessing
Combined all important text fields into a single text input
Handeled missing values 
Converted all text to lowercase
Removed newline and tab characters
Normalized whitespace
Saved the cleaned dataset as a csv file to be used by both models

---
### Feature Engineering
Used the cleaned text column as the primary input for feature extraction

- Additional features (Numerical):
  - Text length (total number of characters)
  - Word count (total number of words)
  - Normalized Keyword frequencies (graph,dp,greedy,etc)

Combined all numeric features (textLength,wordCount and normailzed keyword frequencies) to a single list

Final dataset which is to be used is prepared by extracting both textual (title, description, input/output format, samples) and numerical features (text length, word count, normalized frequencies)

The data is split into training (80%) and testing (20%) set to maintain class distribution.

- A column-wise preprocessing pipeline is used:
  - TF-IDF vectorization is applied to the text part.
  - Standard scaling is applied to numerical features.

The result of this is used as input for all classification and regression models.

---
### Models Used

**Classification Models** 
- Logistic Regression
  - Custom class weights are used to control the class imbalance in the dataset
  - The maximum iterations are set to 1000 to ensure convergence
  - Achieved an accuracy of approximately 52.73%
  - Showed confusion majorly between Medium and Hard class
  - Most easy problems were predicted in medium and hard classes

- Support Vector Machine (Linear SVM)
  - The regularization parameter (C) is set to 0.15 to prevent overfitting
  - Class-weight is used to deal with different number of problem in the classes
  - The maximum number of iterations is set to 5000 to ensure convergence
  - Achieved the highest accuracy of approximately 54.07%
  - Easy and Medium are still mostly predicted in wrong classes
  - Medium-hard confusion remains dominant

- Random Forest Classifier
  - This model uses 400 decision trees, each tree learns decision rules and final output is based on the majority
  - Class-weighing is applied to reduce class imbalance
  - Achieved an accuracy of approximately 51.88%
  - Classified a majority of problems as Hard, was the most confused between Easy and Hard class among the three models

Support Vector Machine was selected as the final model as it showed better performance

---
**Regression Models**
- Linear Regression (Ridge)
  - Assumes a linear relationship between features and difficulty score
  - Ridge uses L2 regularization, it helps prevent overfitting in TF-IDF vectorization
  - The regularization parameter is set to alpha= 2.0
  - Achieved an MAE of approximately 1.648
  - Achieved an RMSE of approximately 1.988

- Random Forest Regressor
  - The model consists of 500 decision trees with maximum tree depth of 5 to control overfitting
  - This model can be used as a non-linear regression baseline
  - Achieved an MAE of approximately 1.767
  - Achieved an RMSE of approximately 2.087
  - Required approximately 2 minutes to train

- Gradient Boosting Regressor
  - The model uses 400 decision trees, each tree learns from the errors of previous trees and each tree has a maximum depth of 3 to prevent overfitting
  - Learning rate of 0.05 is used to control the contribution of each tree
  - Achieved an MAE of approximately 1.697
  - Achieved an RMSE of approximately 2.019
  - Required less training time and performed better than Random Forest Regressor

Linear Regression (Ridge) was selected as the final regression model due to its lowest RMSE 

---
## Evaluation Metrics
### Classification
- Linear SVM model
Accuracy: 54.07047387606318 %
Confusion Matrix:
        easy  hard  medium
easy      57    53      43
hard      16   293      80
medium    18   168      95

### Regression
- Linear Regression model 
MAE: 1.6483395076522422
RMSE: 1.9889106195226198
---
## Steps to Run the Project

git clone <repository-url>
cd <project-folder>

python -m venv venv
venv/bin/activate

pip install -r requirements.txt

streamlit run app.py

---
## Explanation of the Web Interface
The web interface predicts the difficulty of programming problem without requiring command-line usage.

The web interface enables users to:
- Paste or type a complete programming problem description including title, description, input and output description, and sample input and output
- On clicking "Predict Difficulty", the user receives:
  - A difficulty classification (Easy/Medium/Hard)
  - A difficulty score (from 1–10)
The interface provides results quickly in an easy to understand format.

---
### Web Interface Components
#### Text Input Area
- A text box where users enter the problem statement
- Accepts full descriptions including input/output format and constraints
- Applies the same process used during model training

#### Predict Button
- Sends the processed input to:
  - Linear SVM model for classification
  - Linear Regression model for regression

#### Output Display
- Displays the predicted results immediately after submission, the predicted difficulty class and score

The interface is simple and ensures that output is easy to understand.

---
## Link to demo video

---
## Details
- Name: Shreyas Dangi
- Enrollment Number: 24113121





