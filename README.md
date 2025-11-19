# Breast-Cancer-Classification---Logistic-Regression
I built a binary classification model to predict whether a tumor is malignant (cancerous) or benign (non-cancerous) using logistic regression. The goal was to help identify cancer cases accurately.
What I Did
Created a machine learning classifier that predicts breast cancer diagnosis based on tumor characteristics. Used logistic regression because it's perfect for binary classification and gives probability scores, not just yes/no answers.
Dataset

Source: Breast Cancer Wisconsin Dataset (built-in sklearn dataset)
Samples: 569 cases
Features: 30 features (I used 5 main ones for simplicity)
Classes:

0 = Malignant (Cancer) - 37%
1 = Benign (No Cancer) - 63%


Split: 80% training, 20% testing

Tools I Used

Python 3
Pandas for data handling
NumPy for calculations
Scikit-learn for machine learning
Matplotlib and Seaborn for visualizations
breast-cancer-classification/
├── logistic_regression.py
├── README.md
├── INTERVIEW_ANSWERS.md
└── outputs/
    ├── class_distribution.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── sigmoid_function.png
    ├── precision_recall_threshold.png
    ├── feature_importance.png
    └── probability_distribution.png
My Approach
1. Data Exploration
Loaded the dataset and checked class distribution. Found it's slightly imbalanced (63% benign, 37% malignant), but not too bad.
2. Feature Selection
The dataset has 30 features, but I selected 5 main ones for clarity:

Mean radius
Mean texture
Mean perimeter
Mean area
Mean smoothness

3. Data Preprocessing
Train-Test Split: 80-20 split with stratification to maintain class proportions
Feature Scaling: Used StandardScaler because logistic regression is sensitive to feature scales. Transformed all features to have mean=0 and std=1.
4. Model Training
Trained a logistic regression model using sklearn. The model learns coefficients for each feature and uses the sigmoid function to convert linear combinations to probabilities.
5. Making Predictions
The model outputs probabilities (0 to 1). Using default threshold of 0.5:

Probability ≥ 0.5 → Predict Benign (1)
Probability < 0.5 → Predict Malignant (0)

6. Model Evaluation
Evaluated using multiple metrics because accuracy alone isn't enough for medical diagnosis:

Accuracy: Overall correctness
Precision: Of predicted cancer cases, how many are actually cancer
Recall: Of actual cancer cases, how many did we catch
F1-Score: Balance between precision and recall
ROC-AUC: Overall ability to distinguish between classes

7. Visualizations
Created 7 different plots to understand model performance:

Class distribution
Confusion matrix
ROC curve
Sigmoid function
Precision-Recall vs Threshold
Feature importance
Probability distribution

Results
Model Performance (Test Set):

Accuracy: 96.5%
Precision: 97.2%
Recall: 97.2%
F1-Score: 97.2%
ROC-AUC: 99.3%

Confusion Matrix Breakdown:

True Negatives: 40 (Correctly identified malignant)
False Positives: 3 (False alarms - said cancer when it's not)
False Negatives: 1 (CRITICAL - Missed cancer case)
True Positives: 70 (Correctly identified benign)

What This Means:

Out of 114 test cases, only 4 wrong predictions
Only 1 missed cancer case (most critical error)
3 false alarms (less critical but still important)
Model is very reliable!

Key Findings
Feature Importance:
The model coefficients show which features matter most:

Mean perimeter: Strongest predictor
Mean area: Second most important
Mean radius: Also significant
Larger, irregular tumors more likely malignant

Sigmoid Function:
The magic behind logistic regression! It converts any number to a probability:

Input: Linear combination of features
Output: Probability between 0 and 1
Formula: P = 1 / (1 + e^(-z))

Threshold Selection:
Default threshold is 0.5, but you can adjust based on priorities:

Lower threshold (0.3): Catch more cancer cases (higher recall, lower precision)
Higher threshold (0.7): More confident predictions (higher precision, lower recall)

For cancer detection, I'd prefer lower threshold to minimize missed cases!
Understanding the Metrics
Precision (97.2%):
"When the model says it's benign, it's right 97.2% of the time"

Important for avoiding unnecessary worry

Recall (97.2%):
"The model catches 97.2% of actual benign cases"

Critical for cancer: want to catch all cancer cases!

ROC-AUC (99.3%):
"Model has 99.3% chance of ranking a random benign case higher than a random malignant case"

Very high score = excellent discrimination

Confusion Matrix Explained
Predicted
                Malig  Benign
Actual  Malig     40      3
        Benign     1     70

What matters most in cancer detection:

False Negatives (1): Most dangerous - telling someone they're fine when they have cancer
False Positives (3): Less dangerous but causes anxiety - false alarms
Goal: Minimize false negatives even if it means more false positives

What I Learned

How logistic regression works and why it's used for classification
The sigmoid function and why it's perfect for probabilities
Difference between precision and recall and when each matters
How to read and interpret a confusion matrix
ROC curve and AUC score for model comparison
Feature scaling is crucial for logistic regression
Threshold tuning based on business requirements
In medical diagnosis, recall is often more important than precision
Model evaluation requires multiple metrics, not just accuracy

Advantages of Logistic Regression
✅ Fast to train and predict
✅ Outputs probabilities, not just classes
✅ Easy to interpret (can see feature importance)
✅ Works well with linearly separable classes
✅ No hyperparameters to tune (simple model)
✅ Less prone to overfitting than complex models
✅ Memory efficient
Limitations
⚠️ Assumes linear decision boundary
⚠️ May underperform with non-linear relationships
⚠️ Sensitive to outliers
⚠️ Requires feature scaling
⚠️ Can't capture complex patterns like neural networks
How to Run
Install required packages:
bashpip install pandas numpy matplotlib seaborn scikit-learn
Run the script:
bashpython logistic_regression.py
The script will:

Load the breast cancer dataset
Preprocess and split the data
Train the logistic regression model
Evaluate on test set
Generate 7 visualization images
Print detailed metrics and analysis

Real-World Applications
This type of model could be used for:

Cancer diagnosis support (like in this project)
Spam email detection
Credit card fraud detection
Customer churn prediction
Disease outbreak prediction
Loan default prediction

Anywhere you need binary (yes/no) predictions with probability scores!
Next Steps / Improvements

Try using all 30 features instead of just 5
Test different threshold values for different scenarios
Handle class imbalance with techniques like SMOTE
Try other algorithms (Random Forest, SVM) for comparison
Cross-validation for more robust evaluation
Feature engineering to create new predictive features
Hyperparameter tuning (C parameter, solver choice)
Deploy as a web app for real-time predictions

Conclusion
Logistic regression achieved 96.5% accuracy on breast cancer classification, which is really good! The model is interpretable, fast, and reliable. Most importantly, it only missed 1 cancer case out of 41, which is crucial for medical diagnosis.
The sigmoid function elegantly converts feature combinations into probabilities, making the predictions interpretable and actionable. With proper threshold tuning, this model could be a useful tool to assist doctors in diagnosis.        
    
