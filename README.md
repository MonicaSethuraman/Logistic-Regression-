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






















