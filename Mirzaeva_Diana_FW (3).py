#!/usr/bin/env python
# coding: utf-8

# # Bachelor Thesis Code by: Diana Mirzaeva

# In[70]:


# Python 3 environment comes with many helpful analytics libraries installed


# In[71]:


import pandas as pd
import numpy as np
import matplotlib as maplib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, neighbors, tree, ensemble, linear_model
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV , StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')


# In[72]:


# We import the desired dataset from Kaggle and analyze it


# In[73]:


riskdata=pd.read_csv('german_credit_dataset.csv')


# In[74]:


riskdata.shape


# ## Preprocessing

# In[75]:


riskdata=riskdata.iloc[:, 1:] #Clean the dataset , we deletethe ID column


# In[76]:


riskdata.head()


# In[77]:


riskdata.info()


# In[78]:


#Defining missing data
miss = riskdata.isnull().sum()
miss


# In[79]:


#only 2 columns contain NaN: Saving accounts and checking accounts
#We can assume, that when it says NaN = client doesn't have an account or have smth else. So we assume Nan as 'none' or "other"
for col in ['Saving accounts', 'Checking account']:
    riskdata[col].fillna('none', inplace=True)
miss = riskdata.isnull().sum()
miss


# In[80]:


riskdata['Purpose'].unique()


# ### Data Analysis 

# In[81]:


riskdata['Housing'].value_counts().plot.pie( figsize=(5, 5))


# In[82]:


riskdata['Purpose'].value_counts().plot.pie( figsize=(5, 5))


# In[83]:


# We encode categorical variables using one hot encoding. 


# In[84]:


#Encoding binary variables as 0,1 including target variable
riskdata['Sex'] = riskdata['Sex'].map({'male' :0, 'female' :1})
riskdata['Risk'] = riskdata['Risk'].map({'good' :0, 'bad' :1})


# In[85]:


#Select categorical data and do one hot encoding.
one_hot_encoded_columns=pd.get_dummies(riskdata.select_dtypes(include=['object']))
riskdata.drop(list(riskdata.select_dtypes(include=['object']).columns),axis=1,inplace=True)
riskdata=pd.concat([riskdata,one_hot_encoded_columns],axis=1)
riskdata.head()


# In[86]:


corr = riskdata.corr()
corr


# In[87]:


#PIE CHART OF RISK
riskdata['Risk'].value_counts().plot.pie( figsize=(5, 5))


# In[89]:


#'Saving accounts:'none' :0, 'little' :1, 'moderate' :2, 'quite rich' :3, 'rich' :4
#Checking account = 'none' :0, 'little' :1, 'moderate' :2, 'rich' :3


# In[90]:


fig, ax = plt.subplots(1,2,figsize=(15,5))
#sns.histplot(riskdata, x='Age', bins=30, hue="Sex", ax=ax[0]).set_title("Age/Sex Distribution");
sns.histplot(riskdata, x='Age', bins=30, hue="Risk", ax=ax[0]).set_title("Age Distribution with Risk");
sns.boxplot(data=riskdata, x="Sex", y="Age", ax=ax[1]).set_title("Age/Sex Distribution");

fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.boxplot(data=riskdata, x='Risk', y='Duration', ax=ax[1]).set_title("Duration (in month) Distribution with Risk");
sns.boxplot(data=riskdata, x='Risk', y='Credit amount', ax=ax[0]).set_title("Credit Amount Distribution with Risk");
#sns.boxplot(data=riskdata, x='Risk', y='Age', ax=ax[0]).set_title("Age Distribution with Risk");
#sns.countplot(data=riskdata, x="Sex", hue="Risk", ax=ax[1]).set_title("Sex Distribution with Risk");


# In[91]:


#Correlation
fig, axes = plt.subplots(1,figsize=(45,10))

#heatmap using seaborn
corr = riskdata.corr()
plt.subplot(1,2,1)
sns.heatmap(corr, annot=True)


# In[92]:


riskdata.head()


# In[93]:


# Scaling of numerical values
ss=StandardScaler()
numerical_columns=['Age','Job','Credit amount','Duration']
riskdata[numerical_columns]=pd.DataFrame(ss.fit_transform(riskdata[numerical_columns]),columns=numerical_columns)


# ## Training Models

# In[94]:


y=riskdata['Risk'].values
riskdata.drop('Risk',axis=1,inplace=True)
X=riskdata.values


# In[95]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)


# In[96]:


print('Shape training dataset {}, Shape test dataset {}'.format(X_train.shape,X_test.shape))


# ### Hyperparameters Tuning 

# In[97]:


# Function to search for the best parameters of a model 
def grid_search_function (model, X_train, y_train, parameters):
    grid_search = GridSearchCV(estimator = model,
     param_grid = parameters,
     scoring = 'accuracy',
     cv = 5,
     n_jobs = -1)
    grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
    print('Best accuracy: {:.2f} %'.format(best_accuracy*100))
    print('Best parameters: ',best_parameters)
    return grid_search.best_estimator_


# In[98]:


def plot_confusion_matrix(y_test,y_pred,labels,alg_name):
    cm = confusion_matrix(y_test, y_pred,labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels) 
    fig=plt.figure(figsize=(8,6))
    plt.rc('font', family='serif')
    plt.rc('axes', axisbelow=True)
    disp.plot(cmap='Blues')
    plt.grid(False)
    plt.title('Confusion Matrix '+ alg_name)
    plt.show() 


# ### Model Definition 

# ### 1. KNN

# In[99]:


#kNN MODEL and metrics
knn=KNeighborsClassifier().fit(X_train,y_train)


# In[100]:


# Model Accuracy Before Hyperparameter Tuning 


# In[101]:


accuracies = cross_val_score(estimator = knn, X = X_train, y = y_train, cv = 5)
print('KNN: Mean Accuracy before tuning', format(accuracies.mean()*100))
print('KNN: Standard deviation before tuning',format(accuracies.std()*100))


# In[ ]:


#Hyperparameter Tuning 


# In[102]:


parameters_knn = [{'n_neighbors': list(range(1, 50))}]
best_knn=grid_search_function(knn, X_train, y_train, parameters_knn)
y_tr_knn_pred = best_knn.predict(X_train)
y_ts_knn_pred = best_knn.predict(X_test)
probs_knn = knn.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs_knn[:,1])


# In[103]:


#Train DATA METRICS!!! KNN
print('Accuracy: {:.3f} %'.format(accuracy_score(y_train, y_tr_knn_pred)*100))
print('Precision: {:.3f} %'.format(precision_score(y_train,y_tr_knn_pred)*100))
print('Recall: {:.3f} %'.format(recall_score(y_train,y_tr_knn_pred)*100))
print('ROC_AUC: {:.3f} %'.format(roc_auc_score(y_train,y_tr_knn_pred)*100))
print('F1: {:.3f} %'.format(f1_score(y_train,y_tr_knn_pred)*100))


# In[104]:


#Test DATA METRICS!!! KNN
print('Accuracy: {:.3f} %'.format(accuracy_score(y_test, y_ts_knn_pred)*100))
print('Precision: {:.3f} %'.format(precision_score(y_test,y_ts_knn_pred)*100))
print('Recall: {:.3f} %'.format(recall_score(y_test,y_ts_knn_pred)*100))
print('ROC_AUC: {:.3f} %'.format(roc_auc_score(y_test,y_ts_knn_pred)*100))
print('F1: {:.3f} %'.format(f1_score(y_test,y_ts_knn_pred)*100))


# In[105]:


#CONFUSION MATRIX TEST      
knn_confusion_matrix_tr = confusion_matrix(y_test,y_ts_knn_pred)
print(knn_confusion_matrix_tr)
plot_confusion_matrix(y_test,y_ts_knn_pred,[0,1],'KNN')


# ## 2. Logistic Regression 

# In[106]:


# LOGISTIC REGRESSION
log = LogisticRegression(max_iter=500).fit(X_train,y_train)


# In[107]:


# Model Accuracy Before Hyperparameter Tuning 


# In[108]:


accuracies = cross_val_score(estimator = log, X = X_train, y = y_train, cv = 5)
print('LOGISTIC REGRESSION: Mean Accuracy before tuning', format(accuracies.mean()*100))
print('LOGISTIC REGRESSION: Standard deviation before tuning',format(accuracies.std()*100))


# In[ ]:


#Hyperparameter Tuning 


# In[109]:


parameters_log = {'penalty':  ['none', 'l1', 'l2', 'elasticnet'], 'C': [100, 10, 1.0, 0.1, 0.01], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
best_log=grid_search_function(log, X_train, y_train, parameters_log)
y_tr_log_pred = best_log.predict(X_train)
y_ts_log_pred = best_log.predict(X_test)
probs_log = best_log.predict_proba(X_test)


# In[110]:


#Train DATA METRICS!!! LOG
print('Accuracy: {:.3f} %'.format(accuracy_score(y_train, y_tr_log_pred)*100))
print('Precision: {:.3f} %'.format(precision_score(y_train, y_tr_log_pred)*100))
print('Recall: {:.3f} %'.format(recall_score(y_train, y_tr_log_pred)*100))
print('ROC_AUC: {:.3f} %'.format(roc_auc_score(y_train, y_tr_log_pred)*100))
print('F1: {:.3f} %'.format(f1_score(y_train, y_tr_log_pred)*100))


# In[111]:


#Test DATA METRICS!!! LOG
print('Accuracy: {:.3f} %'.format(accuracy_score(y_test, y_ts_log_pred)*100))
print('Precision: {:.3f} %'.format(precision_score(y_test, y_ts_log_pred)*100))
print('Recall: {:.3f} %'.format(recall_score(y_test, y_ts_log_pred)*100))
print('ROC_AUC: {:.3f} %'.format(roc_auc_score(y_test, y_ts_log_pred)*100))
print('F1: {:.3f} %'.format(f1_score(y_test, y_ts_log_pred)*100))


# In[112]:


#CONFUSION MATRIX TEST
log_confusion_matrix_ts = confusion_matrix(y_test,y_ts_log_pred)
print(log_confusion_matrix_ts)
plot_confusion_matrix(y_test,y_ts_log_pred,[0,1],'Logistic R.')


# ### 3. Decision Trees 

# In[113]:


# DECISION TREE and Metrics
dtc=DecisionTreeClassifier().fit(X_train,y_train)


# In[114]:


# Model Accuracy Before Hyperparameter Tuning 


# In[115]:


accuracies = cross_val_score(estimator = dtc, X = X_train, y = y_train, cv = 5)
print('DT: Mean Accuracy before tuning', format(accuracies.mean()*100))
print('DT: Standard deviation before tuning',format(accuracies.std()*100))


# In[ ]:


#Hyperparameter Tuning 


# In[116]:


parameters_dtc = {'criterion':['gini', 'entropy'],'max_depth': list(range(2,50))}
best_dtc=grid_search_function(dtc, X_train, y_train, parameters_dtc)
y_tr_dtc_pred = best_dtc.predict(X_train)
y_ts_dtc_pred = best_dtc.predict(X_test)
probs_dtc = best_dtc.predict_proba(X_test)
fpr,tpr,thresh = metrics.roc_curve(y_test,probs_dtc[:,1])


# In[117]:


#Train DATA METRICS!!! DTC
print('Accuracy: {:.3f} %'.format(accuracy_score(y_train, y_tr_dtc_pred)*100))
print('Precision: {:.3f} %'.format(precision_score(y_train, y_tr_dtc_pred)*100))
print('Recall: {:.3f} %'.format(recall_score(y_train, y_tr_dtc_pred)*100))
print('ROC_AUC: {:.3f} %'.format(roc_auc_score(y_train, y_tr_dtc_pred)*100))
print('F1: {:.3f} %'.format(f1_score(y_train, y_tr_dtc_pred)*100))


# In[118]:


#Test DATA METRICS!!! DTC
print('Accuracy: {:.3f} %'.format(accuracy_score(y_test, y_ts_dtc_pred)*100))
print('Precision: {:.3f} %'.format(precision_score(y_test, y_ts_dtc_pred)*100))
print('Recall: {:.3f} %'.format(recall_score(y_test, y_ts_dtc_pred)*100))
print('ROC_AUC: {:.3f} %'.format(roc_auc_score(y_test, y_ts_dtc_pred)*100))
print('F1: {:.3f} %'.format(f1_score(y_test, y_ts_dtc_pred)*100))


# In[136]:


#CONFUSION MATRIX TEST DTC
dtc_confusion_matrix_ts = confusion_matrix(y_test,y_ts_dtc_pred)
print(dtc_confusion_matrix_ts)
plot_confusion_matrix(y_test,y_ts_dtc_pred,[0,1],'DT')


# In[120]:


fig,ax= plt.subplots(figsize=(30,10))
tree.plot_tree(best_dtc,fontsize=14,ax=ax,feature_names=riskdata.columns)
plt.show()


# ## 4. Random Forest 

# In[121]:


#Random Forest
rfc=RandomForestClassifier().fit(X_train,y_train)


# In[122]:


# Model accuracy before hyperparameter tuning 


# In[123]:


accuracies = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 5)
print('DT: Mean Accuracy before tuning', format(accuracies.mean()*100))
print('DT: Standard deviation before tuning',format(accuracies.std()*100))


# In[ ]:


#Hyperparameter Tuning 


# In[124]:


parameters_rf = {'n_estimators': [40,60,80,100,150,200], 'max_features': ['auto', 'sqrt', 'log2'],'max_depth' : list(range(5,45,5)), 'criterion' :['gini', 'entropy']}
best_rf=grid_search_function(rfc, X_train, y_train, parameters_rf)
y_tr_rfc_pred = best_rf.predict(X_train)
y_ts_rfc_pred = best_rf.predict(X_test)
probs_rfc = best_rf.predict_proba(X_test)
fpr,tpr,thresh = metrics.roc_curve(y_test,probs_rfc[:,1])


# In[125]:


#METRICS Random FOREST TRAIN
print('Accuracy: {:.3f} %'.format(accuracy_score(y_train, y_tr_rfc_pred)*100))
print('Precision: {:.3f} %'.format(precision_score(y_train, y_tr_rfc_pred)*100))
print('Recall: {:.3f} %'.format(recall_score(y_train, y_tr_rfc_pred)*100))
print('ROC_AUC: {:.3f} %'.format(roc_auc_score(y_train, y_tr_rfc_pred)*100))
print('F1: {:.3f} %'.format(f1_score(y_train, y_tr_rfc_pred)*100))


# In[126]:


#METRICS Random FOREST TEST
print('Accuracy: {:.3f} %'.format(accuracy_score(y_test, y_ts_rfc_pred)*100))
print('Precision: {:.3f} %'.format(precision_score(y_test, y_ts_rfc_pred)*100))
print('Recall: {:.3f} %'.format(recall_score(y_test, y_ts_rfc_pred)*100))
print('ROC_AUC: {:.3f} %'.format(roc_auc_score(y_test, y_ts_rfc_pred)*100))
print('F1: {:.3f} %'.format(f1_score(y_test, y_ts_rfc_pred)*100))


# In[127]:


rfc_confusion_matrix_ts = confusion_matrix(y_test,y_ts_rfc_pred)
print(rfc_confusion_matrix_ts)
plot_confusion_matrix(y_test,y_ts_rfc_pred,[0,1],'Random Forest')


# ### RESULTS OF CROSS VALIDATION

# In[128]:


#Function to compare different models using cross validation results based on a given evaluation metric.
def boxplot_models(results_models,evaluation_metric,models_names):
    results=[ result['test_{}'.format(evaluation_metric)]  for result in results_models]
    data=pd.DataFrame(results,index=models_names).transpose()
    data_melt=data.melt()
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(x='variable',y='value',data=data_melt,palette=sns.color_palette("hls", 3),showmeans=True,meanprops={"marker":"s","markerfacecolor":"black", "markeredgecolor":"black"})
    plt.ylabel(evaluation_metric)
    plt.xlabel('models')
    plt.show()
    


# In[131]:


models_names=['KNN','LR','DT','RF']
list_of_models=[ best_knn,best_log,best_dtc,best_rf]
kfold = StratifiedKFold(n_splits=5, random_state=1,shuffle=True)
scoring=['accuracy','precision','recall','f1','roc_auc']
accuracy_metrics=['Accuracy','Precision','Recall','F1_Score','ROC-AUC']
all_results=[]
avt=[]
avt=[]
for model in list_of_models:
    results=cross_validate(model,X_train,y_train,cv=10,scoring=scoring,return_train_score=True)
    all_results.append(results)
    model_results_validation=[]
    for score_metric in scoring:
        model_results_validation.append(results['test_'+score_metric].mean())
    avt.append(model_results_validation)
summary_validation=pd.DataFrame(avt,index=models_names,columns=accuracy_metrics).transpose()
boxplot_models(all_results,'accuracy',models_names)


# In[132]:


summary_validation


# In[ ]:


#Timing of each model


# In[133]:


cv_results = pd.DataFrame(columns=['model', 'fit_time', 'score_time'])
for i,key in enumerate(list_of_models):
    cv_res = model_selection.cross_validate(key, X_train, y_train, 
                                             return_train_score=True,
                                             scoring="accuracy",
                                             cv=5, n_jobs=-1)
    res = {
        'model': models_names[i], 
        'fit_time': cv_res["fit_time"].mean(),
        'score_time': cv_res["score_time"].mean(),
        }
    cv_results = cv_results.append(res, ignore_index=True)
    print("CV for model:", key, "done.")
cv_results


# ### RESULTS TEST SET

# In[134]:


#Metrics TEST
# KNN
print("KNN:")
print('KNN_Accuracy: {:.3f} %'.format(accuracy_score(y_test, y_ts_knn_pred)*100))
print('KNN_Precision: {:.3f} %'.format(precision_score(y_test, y_ts_knn_pred)*100))
print('KNN_Recall: {:.3f} %'.format(recall_score(y_test, y_ts_knn_pred)*100))
print('KNN_ROC_AUC: {:.3f} %'.format(roc_auc_score(y_test, y_ts_knn_pred)*100))
print('KNN_F1: {:.3f} %'.format(f1_score(y_test, y_ts_knn_pred)*100))
print("---------------")
#Log
print("Log:")
print('LOG_Accuracy: {:.3f} %'.format(accuracy_score(y_test, y_ts_log_pred)*100))
print('LOG_Precision: {:.3f} %'.format(precision_score(y_test, y_ts_log_pred)*100))
print('LOG_Recall: {:.3f} %'.format(recall_score(y_test, y_ts_log_pred)*100))
print('LOG_ROC_AUC: {:.3f} %'.format(roc_auc_score(y_test, y_ts_log_pred)*100))
print('LOG_F1: {:.3f} %'.format(f1_score(y_test, y_ts_log_pred)*100))
print("---------------")

#Decision Trees
print("Decision Trees:")
print('DTC_Accuracy: {:.3f} %'.format(accuracy_score(y_test, y_ts_dtc_pred)*100))
print('DTC_Precision: {:.3f} %'.format(precision_score(y_test, y_ts_dtc_pred)*100))
print('DTC_Recall: {:.3f} %'.format(recall_score(y_test, y_ts_dtc_pred)*100))
print('DTC_ROC_AUC: {:.3f} %'.format(roc_auc_score(y_test, y_ts_dtc_pred)*100))
print('DTC_F1: {:.3f} %'.format(f1_score(y_test, y_ts_dtc_pred)*100))
print("---------------")
#Random Forest
print("Random Forest:")
print('Accuracy: {:.3f} %'.format(accuracy_score(y_test,y_ts_rfc_pred)*100))
print('Precision: {:.3f} %'.format(precision_score(y_test,y_ts_rfc_pred)*100))
print('Recall: {:.3f} %'.format(recall_score(y_test,y_ts_rfc_pred)*100))
print('ROC_AUC: {:.3f} %'.format(roc_auc_score(y_test,y_ts_rfc_pred)*100))
print('F1: {:.3f} %'.format(f1_score(y_test,y_ts_rfc_pred)*100))


# In[135]:


#ROC_AUC
fig = plt.figure(figsize=(8,5))
plt.plot([0, 1], [0, 1],'r--')

#KNN
preds_proba_knn = best_knn.predict_proba(X_test)
probs_knn = preds_proba_knn[:, 1]
fpr, tpr, thresh = metrics.roc_curve(y_test, probs_knn)
auc_knn = roc_auc_score(y_test, probs_knn)
plt.plot(fpr, tpr, label=f'KNN, AUC = {str(round(auc_knn,3))}')

#Logistic Regression
preds_proba_log = best_log.predict_proba(X_test)
probs_log = preds_proba_log[:, 1]
fpr, tpr, thresh = metrics.roc_curve(y_test, probs_log)
auclg = roc_auc_score(y_test, probs_log)
plt.plot(fpr, tpr, label=f'Logistic Regression, AUC = {str(round(auclg,3))}')

#DecisionTree Classifier
preds_proba_dtc =best_dtc.predict_proba(X_test)
probs_dtc = preds_proba_dtc[:, 1]
fpr, tpr, thresh = metrics.roc_curve(y_test, probs_dtc)
auc_dtc = roc_auc_score(y_test, probs_dtc)
plt.plot(fpr, tpr, label=f'DecisionTree Classifier, AUC = {str(round(auc_dtc,3))}')


# Random Forest
preds_proba_rfc = best_rf.predict_proba(X_test)
probs_rfc = preds_proba_rfc[:, 1]
fpr, tpr, thresh = metrics.roc_curve(y_test, probs_rfc)
auc_rf = roc_auc_score(y_test, probs_rfc)
plt.plot(fpr, tpr, label=f'RandomForestClassifier, AUC = {str(round(auc_rf,3))}')


plt.ylabel("True Positive Rate", fontsize=12)
plt.xlabel("False Positive Rate", fontsize=12)
plt.title("ROC curve")
plt.rcParams['axes.titlesize'] = 16
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




