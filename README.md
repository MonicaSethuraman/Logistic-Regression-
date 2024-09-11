# LOGISTICAL REGRESSION ANALYSIS


The goal of this study is to create a Logistic Regression model that predicts the risk level of pregnancies based on a variety of essential health markers such as age, blood pressure, blood glucose levels, body temperature, and heart rate. 

![image](https://github.com/user-attachments/assets/637d2842-ec54-4a29-ae5d-2d11d496b3bb)

# Leaning curve for Logistical Regression

In machine learning, the learning curve is used to assess how models will perform with varying numbers of training data. This is accomplished by tracking the training and validation scores (model accuracy) as the number of training samples increases. The blue dashed line represents training, the orange line represents validation, and the desired model accuracy is shown below.The graph shows “HIGH BIAS”. This model's training and cross validation accuracy are both low, indicating that it “underfits” the training data. One common solution is to increase the number of parameters in the model.
For example, more features can be collected or constructed, or the degree of regularization in logistic regression classifiers can be reduced.

![image](https://github.com/user-attachments/assets/1f5aa7ee-2975-4711-91e4-7871b3256d61)

# Confusion Matrix

The confusion matrix is a table that summarizes a classification model's performance by counting the number of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions. In a multi-class classification issue, it provides a thorough picture of the model's performance for each class.

![image](https://github.com/user-attachments/assets/070bbcfd-d1b7-4402-a2d4-bc8dbeffea6d)


![image](https://github.com/user-attachments/assets/617d33ed-f6aa-40b4-9813-029f8032fa3b)

# Key Insights 

True Positives (TP): Cases correctly predicted as High/Mid/Low risk.
True Negatives (TN): Cases correctly predicted as Not High/Mid/Low risk.
False Positives (FP): Cases incorrectly predicted as High/Mid/Low risk.
False Negatives (FN): Cases incorrectly predicted as Not High/Mid/Low risk.

### Precision :
Precision measures how many of the expected positive cases were actually positive. When the model predicts a favorable outcome, it is a measure of its accuracy.
The precision for the "High Risk" class is 0.80. This suggests that, of all the situations identified by the algorithm as "High Risk," 80% were indeed "High Risk."
The accuracy for the "Low Risk" class is 0.59. This suggests that, of all the cases projected as "Low Risk," 59% were truly "Low Risk."
The accuracy for the "Mid Risk" class is 0.53. This suggests that, of all the cases projected as "Mid Risk," 53% were truly "Mid Risk."

![image](https://github.com/user-attachments/assets/1ffecaf3-7f74-48b0-85e3-9fb2704c89f8)


### Recall:

Recall, also known as sensitivity or true positive rate, quantifies how many actual positive cases the model properly predicted.
The recall for the "High Risk" class is 0.64. This means that the model successfully recognized 64% of the actual "High Risk" situations.
The recall for the "Low Risk" class is 0.90. This means that the model successfully recognized 90% of the actual "Low Risk" situations.
The recall for the "Mid Risk" class is 0.28. This suggests that the model properly recognized only 28% of the actual "Mid Risk" situations.

### F1-Score:

The F1-score is the harmonic mean of precision and recall. It provides a balance between these two metrics and is useful when you want to consider both false positives and false negatives.
For the "High Risk" class, the F1-score is 0.71.
For the "Low Risk" class, the F1-score is 0.72.
For the "Mid Risk" class, the F1-score is 0.37.


# ROC Curve

True Positive Rate vs. False Positive Rate is plotted on a ROC curve at various categorization levels. Lowering the categorization threshold causes more items to be classified as positive, which increases both False Positives and True Positives.

# AUC Curve

AUC is a metric that aggregates performance across all categorization criteria. AUC can be interpreted as the likelihood that the model ranks a random positive case higher than a random negative example.


# Key insights from the ROC/ AUC Curve



### AUC Values:

The AUC values for each class's ROC curve provide information about the model's ability to identify that class from the others. AUC greater than one implies better separation. 


### Overall Model Performance: 

The model's overall performance across all classes is represented by the macro-average ROC curve and AUC. It is critical to determine whether the macro-average AUC is significantly better than random chance (the diagonal line). If this is the case, it implies that the model has some predictive power across all classes.

### Class-Specific Performance:

Examining how well the model performs in each class separately. For example, if the ROC curve for one class is very near to the diagonal line, the model is having difficulty distinguishing that class from others.


# Recommendations 

## Feature Engineering and Selection Explanation:

As demonstrated by the confusion matrix and classification report, the logistic regression model's performance might be enhanced by expanding the feature set. Implement feature selection strategies to find and keep just the most informative features. Removing irrelevant or superfluous characteristics can help to simplify the model and lessen the danger of overfitting. Critical features can be identified using techniques such as recursive feature elimination (RFE) or feature importance from tree-based models.

## Explanation of Model Tuning and Evaluation:

In terms of precision, recall, and F1-score, the existing logistic regression model should be improved. It is critical to fine-tune the model and examine several algorithms. Optimise the logistic regression model's hyperparameters using approaches such as grid search or random search. The regularisation strength (C), penalty type, and solver method are all important hyperparameters to consider. Tuning these parameters can result in a model that matches the data better.





# Machine Learning Model for Risk Classification

This Python script applies a Decision Tree and Random Forest classifier on the dataset to predict the risk levels (`low risk`, `mid risk`, `high risk`) based on various health parameters.


# Machine Learning Model for Risk Classification

This Python script applies a Decision Tree and Random Forest classifier on the dataset to predict the risk levels (`low risk`, `mid risk`, `high risk`) based on various health parameters.

## Python Code

```python
# Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Load Dataset
# Data set is loaded here in form of csv and 10 line items are created within head
dataset = pd.read_csv('./MHR_Dataset.csv')  
dataset.head(10)

# Create x and y variables
x = dataset.drop('RiskLevel', axis=1).to_numpy()
y = dataset['RiskLevel'].to_numpy()

# Create Train and Test Dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=100)

# Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Construct some pipelines 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create Pipeline
pipeline = []

pipe_rdf = Pipeline([('scl', StandardScaler()), ('clf', RandomForestClassifier(random_state=100))])
pipeline.insert(0, pipe_rdf)

pipe_dt = Pipeline([('scl', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=100))])
pipeline.insert(1, pipe_dt)

# Set grid search params
modelpara = []

param_gridrdf = {
    'clf__criterion': ['gini', 'entropy'],
    'clf__n_estimators': [100, 150, 200],
    'clf__bootstrap': [True, False]
}
modelpara.insert(0, param_gridrdf)

param_griddt = {'clf__criterion': ['gini', 'entropy'], 'clf__max_depth': range(1, 100)}
modelpara.insert(1, param_griddt)

# Define Plot for learning curve
from sklearn.model_selection import learning_curve

def plot_learning_curves(model):
    train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=x_train, y=y_train, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10), cv=10, 
                                                            scoring='recall_weighted', random_state=100)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    
    plt.grid(True)
    plt.xlabel('Number of training samples')
    plt.ylabel('Recall')
    plt.legend(loc='best')
    plt.ylim([0.5, 1.01])
    plt.show()

# Plot Learning Curve
print('Decision Tree - Learning Curve')
plot_learning_curves(pipe_dt)

print('\nRandom Forest - Learning Curve')
plot_learning_curves(pipe_rdf)

# Model Evaluation and Analysis
from sklearn.model_selection import RepeatedKFold, cross_val_score

models = [('Decision Tree', pipe_dt), ('Random Forest', pipe_rdf)]
scoring = 'recall_weighted'

print('Model Evaluation - Recall')
for name, model in models:
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)
    cv_results = cross_val_score(model, x_train, y_train, cv=rkf, scoring=scoring)
    print('{} {:.2f} +/- {:.2f}'.format(name, cv_results.mean(), cv_results.std()))

# Boxplot View
fig = plt.figure(figsize=(10, 5))
fig.suptitle('Boxplot View')
ax = fig.add_subplot(111)
sns.boxplot(data=[cross_val_score(pipe_dt, x_train, y_train, cv=rkf), cross_val_score(pipe_rdf, x_train, y_train, cv=rkf)])
ax.set_xticklabels(['Decision Tree', 'Random Forest'])
plt.ylabel('Recall')
plt.xlabel('Model')
plt.show()

# Define Gridsearch Function
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  

def Gridsearch_cv(model, params):
    
    # Cross-validation Function
    cv2 = RepeatedKFold(n_splits=10, n_repeats=5, random_state=100)
        
    # GridSearch CV
    gs_clf = GridSearchCV(model, params, cv=cv2, scoring='recall_weighted')
    gs_clf = gs_clf.fit(x_train, y_train)
    model = gs_clf.best_estimator_
    
    # Use best model and test data for final evaluation
    y_pred = model.predict(x_test)

    # Identify Best Parameters to Optimize the Model
    bestpara = str(gs_clf.best_params_)
    
    # Output Heading
    print('\nOptimized Model')
    print('\nModel Name:', str(pipeline.named_steps['clf']))
    print('\n')
    
    # Feature Importance - optimized
    print('Feature Importances')
    for name, score in zip(list(dataset), gs_clf.best_estimator_.named_steps['clf'].feature_importances_):
        print(name, round(score, 2))
    
    # Output Validation Statistics
    target_names = ['low risk', 'mid risk', 'high risk']
    print('\nBest Parameters:', bestpara)
    print('\n', confusion_matrix(y_test, y_pred))  
    print('\n', classification_report(y_test, y_pred, target_names=target_names))  

# Run Models
for pipeline, modelpara in zip(pipeline, modelpara):
    Gridsearch_cv(pipeline, modelpara)

    
   



















