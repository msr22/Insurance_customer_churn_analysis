#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from pylab import rcParams
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score


# Loading the CSV with pandas
data = pd.read_csv(r'D:\Learning\AIML\Projects\Churn analysis\New_for_Git\InsChurnFinal.csv')

data.columns

pd.isna(data).any()

# Data to plot
sizes = data['Exited'].value_counts(sort = True)
colors = ["grey","purple"] 
rcParams['figure.figsize'] = 5,5
labels = 'False', 'True'
explode = (0,0)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

data.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
data.dtypes

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

#replacing gender male=0, female=1
data['Gender'] = data['Gender'].map({'Male':0,'Female': 1})

#replacing dependents no=0, yes=1
data['Dependents'] = data['Dependents'].map({'No':0,'Yes': 1})

data = pd.get_dummies(data, prefix=['PaymentMethod'], columns=['PaymentMethod'])


data = pd.get_dummies(data, prefix=['Geography'], columns=['Geography'])


data.columns

data = data.rename(columns={'Exited': 'Churn'})


#data.head()


Training_data=data.drop(labels = ["Churn"],axis = 1)
Training_label=data.Churn

Training_data.columns

X= Training_data
y= Training_label
train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression

clf_LR = LogisticRegression()
fit_data= clf_LR.fit(train_data, train_label)
score=0
score=score+fit_data.score(train_data, train_label)
print('Logistic Regression Train accuracy= ', score)
prediction_LR = clf_LR.predict(test_data)
print('Logistic Regression Test accuracy= ', np.mean(prediction_LR == test_label))


# #### Decision Tree Classifier

for i in range(3,15):
    #print(i)
    clf_DTC = DecisionTreeClassifier(max_depth=i)
    fit_data= clf_DTC.fit(train_data, train_label)
    score=0
    score=score+fit_data.score(train_data, train_label)
    #print('Train accuracy= ', score)
    prediction_DTC = clf_DTC.predict(test_data)
    #print('Test accuracy= ', np.mean(prediction_DTC == test_label))

clf_DTC = DecisionTreeClassifier(max_depth=6)
fit_data= clf_DTC.fit(train_data, train_label)
score=0
score=score+fit_data.score(train_data, train_label)
print('Decision Tree Train accuracy= ', score)
prediction_DTC = clf_DTC.predict(test_data)
print('Decision Tree Test accuracy= ', np.mean(prediction_DTC == test_label))


# ## Random Forest Classifier

n_list=[3,5,7,9,11,13,17,29,33,47,51,101,203]
cv_err=[]
RF_err=[]
train_err=[]
for n in n_list:
    clf_RF=RandomForestClassifier(n_estimators=n, class_weight='balanced', n_jobs=-1)
    clf_RF.fit(train_data, train_label)
    #sig_clf=CalibratedClassifierCV(clf)
    #sig_clf.fit(train_data, train_label)
    
    
    #predict_y=sig_clf.predict_proba(test_data)
    #cv_err.append(log_loss(test_label, predict_y))
    
    predict_y=clf_RF.predict_proba(test_data)
    RF_err.append(log_loss(test_label, predict_y))
    
    #predict_y=sig_clf.predict_proba(train_data)
    #train_err.append(log_loss(train_label, predict_y))
    predict_y=clf_RF.predict_proba(train_data)
    train_err.append(log_loss(train_label, predict_y))

    
plt.plot(n_list, RF_err, label='RF prediction error', c='b')
plt.plot(n_list, train_err, label='RF train error', c='r')
plt.legend()
plt.show()

n_list[np.argmin(RF_err)]

np.min(RF_err)


# ### Confusion matrix for random forest classifier

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

prediction_RF = clf_RF.predict(test_data)

clf=RandomForestClassifier(n_estimators=n_list[np.argmin(RF_err)], class_weight='balanced')
clf.fit(train_data, train_label)


np.set_printoptions(precision=2)
class_names= data.Churn

# Random Forest accuracy
accuracy_score=accuracy_score(test_label, clf_RF.predict(test_data))
print('Random forest test accuracy= ', accuracy_score )

accuracy_score=accuracy_score(test_label, clf_RF.predict(test_data))
print('Random forest test accuracy= ', accuracy_score )

print('For Random Forest')
# Plot non-normalized confusion matrix
plot_confusion_matrix(test_label,prediction_RF, classes=class_names,title='Random forest Confusion matrix')

# Plot normalized confusion matrix
plot_confusion_matrix(test_label,prediction_RF, classes=class_names, normalize=True,title='Random forest Normalized confusion matrix')

plt.show()


# Random Forest Feature Importance


np.array([clf.feature_importances_]).T

feature_importance=pd.DataFrame(np.hstack((np.array([Training_data.columns[0:]]).T, np.array([clf_RF.feature_importances_]).T)), columns=['Features', 'Importance'])

feature_importance['Importance']=pd.to_numeric(feature_importance['Importance'])


feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)
