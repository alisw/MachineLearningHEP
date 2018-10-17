from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss, confusion_matrix
import seaborn as sn
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn_evaluation import plot
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

def cross_validation_mse(names_,classifiers_,X_train_,y_train_,cv_,ncores):
  df_scores = pd.DataFrame()
  for name, clf in zip(names_, classifiers_):
    scores = cross_val_score(clf, X_train_, y_train_, scoring="neg_mean_squared_error", cv=cv_, n_jobs=ncores)
    tree_rmse_scores = np.sqrt(-scores)
    df_scores[name] =  tree_rmse_scores
  return df_scores


def plot_cross_validation_mse(names_,df_scores_,suffix_):
  figure1 = plt.figure(figsize=(15,10))
  i=1
  for name in names_:
    ax = plt.subplot(2, len(names_)/2, i)  
    bin_values = np.arange(start=0.2, stop=0.4, step=0.005)  
    l=plt.hist(df_scores_[name], color="blue",bins=bin_values)
    #mystring='$\mu$=%8.2f, \sigma$=%8.2f' % (df_scores_[name].mean(),df_scores_[name].std())
    mystring='$\mu=%8.2f, \sigma=%8.2f$' % (df_scores_[name].mean(),df_scores_[name].std())
    plt.text(0.2, 4., mystring,fontsize=16)
    plt.title(name, fontsize=16)   
    plt.xlabel("scores RMSE",fontsize=16) 
    plt.ylim(0, 5)
    plt.ylabel("Entries",fontsize=16)
    figure1.subplots_adjust(hspace=.5)
    i += 1
  plotname='plots/scoresRME%s.png' % (suffix_)
  plt.savefig(plotname)
  

def confusion(mylistvariables_,names_,classifiers_,suffix_,X_train,y_train,cv):
  figure1 = plt.figure(figsize=(20,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

  i=1
  for name, clf in zip(names_, classifiers_):
    ax = plt.subplot(2, len(names_)/2, i)  
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=cv)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    df_cm = pd.DataFrame(norm_conf_mx,range(2),range(2))
    sn.set(font_scale=1.4)#for label size
    ax.set_title(name+"tot diag=0")
    plt.ylabel("True values",fontsize=16)
    plt.ylabel("Predicted values",fontsize=16)

    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    i += 1
  plotname='plots/confusion_matrix%s_Diag0.png' % (suffix_)
  plt.savefig(plotname)
  
  figure2 = plt.figure(figsize=(20,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

  i=1
  for name, clf in zip(names_, classifiers_):
    ax = plt.subplot(2, len(names_)/2, i)  
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=cv)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    df_cm = pd.DataFrame(norm_conf_mx,range(2),range(2))
    sn.set(font_scale=1.4)#for label size
    ax.set_title(name+"tot diag=0")
    plt.ylabel("True values",fontsize=16)
    plt.ylabel("Predicted values",fontsize=16)

    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    i += 1
  plotname='plots/confusion_matrix%s.png' % (suffix_)
  plt.savefig(plotname)

def precision_recall(mylistvariables_,names_,classifiers_,suffix_,X_train,y_train,cv):
  figure1 = plt.figure(figsize=(20,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

  i=1
  for name, clf in zip(names_, classifiers_):
    ax = plt.subplot(2, len(names_)/2, i)  
    y_proba = cross_val_predict(clf, X_train, y_train, cv=cv,method="predict_proba")
    y_scores = y_proba[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("probability",fontsize=16)
    ax.set_title(name,fontsize=16)
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    i += 1
  plotname='plots/precision_recall%s.png' % (suffix_)
  plt.savefig(plotname)
  
  figure2 = plt.figure(figsize=(20,15))
  i=1
  aucs = []

  for name, clf in zip(names_, classifiers_):
    y_proba = cross_val_predict(clf, X_train, y_train, cv=cv,method="predict_proba")
    y_scores = y_proba[:, 1]
    fpr, tpr, thresholds_forest = roc_curve(y_train,y_scores)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.xlabel('False Positive Rate or (1 - Specifity)',fontsize=20)
    plt.ylabel('True Positive Rate or (Sensitivity)',fontsize=20)
    plt.title('Receiver Operating Characteristic',fontsize=20)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC %s (AUC = %0.2f)' % (names_[i-1], roc_auc), linewidth=4.0)
    plt.legend(loc="lower center",  prop={'size':18})
    i += 1
  plotname='plots/ROCcurve%s.png' % (suffix_)
  plt.savefig(plotname)
