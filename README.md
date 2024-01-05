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

Logistic Regression has a precision of 0.98, a recall of 0.98 and an f1 score of 0.98.

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

K-Nearest Neighbors has a precision of 0.96, a recall of 0.96 and an f1 score of 0.96.

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

SVC has a precision of 0.97, a recall of 0.97 and an f1 score of 0.97.

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

Decision Tree has a precision of 0.94, a recall of 0.94 and an f1 score of 0.94.

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

Random Forest has a precision of 0.98, a recall of 0.98 and an f1 score of 0.98.

## Gradient Boosting

A functional gradient algorithm that repeatedly selects a function that leads in the direction of a weak hypothesis or negative gradient so that it can minimize a loss function. Gradient boosting classifier combines several weak learning models to produce a powerful predicting model.

Results from notebook:
Model: Gradient Boosting
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

AUC Score: 0.995296884185773

Gradient Boosting has a precision of 0.96, a recall of 0.96 and an f1 score of 0.96.

## Naïve Bayes

A supervised machine learning algorithm, which is used for classification tasks, like text classification. It is also part of a family of generative learning algorithms, meaning that it seeks to model the distribution of inputs of a given class or category.

Results from notebook:
Model: Naive Bayes
Confusion Matrix:
[[ 57   6]
 [  5 103]]
Classification Report:
              precision    recall  f1-score   support

           0       0.92      0.90      0.91        63
           1       0.94      0.95      0.95       108

    accuracy                           0.94       171
    macro avg      0.93      0.93      0.93       171
    weighted avg   0.94      0.94      0.94       171

AUC Score: 0.9926513815402704

Naïve Bayes has a precision of 0.94, a recall of 0.94 and an f1 score of 0.94.

## Neural Network (MLP Classifier)

The Multilayer Perceptron (MLP) Classiffier is an effective way to handle comples classification tasks. A misnomer for a modern feedforward artificial neural network, it consists of fully connected neurons with a nonlinear kind of activation function, organized in at least three layers, notable for being able to distinguish data that is not linearly separable.

Results from notebook:
Model: Neural Network (MLP Classifier)
Confusion Matrix:
[[ 61   2]
 [  2 106]]
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.97      0.97        63
           1       0.98      0.98      0.98       108

    accuracy                           0.98       171
    macro avg      0.97      0.97      0.97       171
    weighted avg   0.98      0.98      0.98       171

MLP has a precision of 0.98, a recall of 0.98 and an f1 score of 0.98.

## Adaptive Boosting (AdaBoost) Classifier

A Boosting technique used as an Ensemble Method in Machine Learning. It is called Adaptive Boosting as the weights are re-assigned to each instance, with higher weights assigned to incorrectly classified instances.

Results from notebook:
Model: AdaBoost
Confusion Matrix:
[[ 61   2]
 [  2 106]]
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.97      0.97        63
           1       0.98      0.98      0.98       108

    accuracy                           0.98       171
    macro avg      0.97      0.97      0.97       171
    weighted avg   0.98      0.98      0.98       171

AUC Score: 0.9961787184009406

AdaBoost has a precision of 0.98, a recall of 0.98 and an f1 score of 0.98.

## Extreme Gradient Boost (XGB) Classifier

Gradient Boosting, but extreme. 

Results from notebook:
Model: XGBoost
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

AUC Score: 0.9944150499706055

XGBoost has a precision of 0.97, a recall of 0.97 and an f1 score of 0.97.

# Conclusions

### Final Rankings by f1:
1. Random Forest, MLP, AdaBoost, Logistic Regression
2. XGB, SVC
3. KNeighbors, Gradient Boosting

### Final Rankings by precision:
1. Random Forest, MLP, AdaBoost, Logistic Regression
2. XGB, SVC
3. KNeighbors, Gradient Boosting


