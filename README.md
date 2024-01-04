# Comparing 10 different machine learning models to find the best one for breast cancer classification

## To replicate:
1. Download .ipynb file
2. Upload to Google Collab
3. Run the kernels

## Logistic Regression

A machine learning model that is good for categorizing numerical data. 

Results from notebook:
Model: Logistic Regression
Confusion Matrix:
[[ 62   1]
 [  2 106]]
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.98      0.98        63
           1       0.99      0.98      0.99       108

    accuracy                            0.98       171
    macro avg       0.98      0.98      0.98       171
    weighted avg    0.98      0.98      0.98       171

AUC Score: 0.9980893592004703

Logistic Regression has a precision of 0.97, a recall of 0.98 and an f1 score of 0.98.

## K-Nearest Neighbors

A non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.

Results from notebook:
Model: K-Nearest Neighbors
Confusion Matrix:
[[ 59   4]
 [  3 105]]
Classification Report:
                precision    recall  f1-score   support

           0       0.95      0.94      0.94        63
           1       0.96      0.97      0.97       108

    accuracy                           0.96       171
    macro avg      0.96      0.95      0.96       171
    weighted avg   0.96      0.96      0.96       171

AUC Score: 0.9776601998824221

K-Nearest Neighbors has a precision of 0.95, a recall of 0.94 and an f1 score of 0.94.

## Support Vector Machine (SVC)

Support vector machines are supervised max-margin models with associated learning algorithms that analyze data for classification and regression analysis.

Results from notebook:
Model: Support Vector Machine
Confusion Matrix:
[[ 61   2]
 [  3 105]]
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.97      0.96        63
           1       0.98      0.97      0.98       108

    accuracy                           0.97       171
    macro avg      0.97      0.97      0.97       171
    weighted avg   0.97      0.97      0.97       171

AUC Score: 0.9964726631393297

SVC has a precision of 0.95, a recall of 0.97 and an f1 score of 0.96.

## Decision Tree Classifier

A non-parametric supervised learning algorithm, which is utilized for both classification and regression tasks. It has a hierarchical, tree structure, which consists of a root node, branches, internal nodes and leaf nodes.

Results from notebook:
Model: Decision Tree
Confusion Matrix:
[[ 60   3]
 [  7 101]]
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.95      0.92        63
           1       0.97      0.94      0.95       108

    accuracy                           0.94       171
    macro avg       0.93      0.94      0.94       171
    weighted avg    0.94      0.94      0.94       171

AUC Score: 0.9437830687830687

Decision Tree has a precision of 0.90, a recall of 0.95 and an f1 score of 0.92.

## Random Forest Classifier

Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees.

Results from notebook:
Model: Random Forest
Confusion Matrix:
[[ 60   3]
 [  1 107]]
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.95      0.97        63
           1       0.97      0.99      0.98       108

    accuracy                           0.98       171
    macro avg      0.98      0.97      0.97       171
    weighted avg   0.98      0.98      0.98       171

AUC Score: 0.9959582598471487

Random Forest has a precision of 0.98, a recall of 0.95 and an f1 score of 0.97.


description...

Results from the notebook:

conclusion for out results...


Final Rankings by f1:
1. 
2. 
3. 

Final Rankings by precision:
1. 
2.
3. 

