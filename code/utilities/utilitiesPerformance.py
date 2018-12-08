###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

"""
Methods to: model performance evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, mean_squared_error, f1_score, precision_score
from sklearn_evaluation import plot

def cross_validation_mse(names_,classifiers_,X_train_,y_train_,cv_,ncores):
  df_scores = pd.DataFrame()
  for name, clf in zip(names_, classifiers_): 
    if "Keras" in name:
      ncores=1
    kfold = StratifiedKFold(n_splits=cv_, shuffle=True, random_state=1)
    scores = cross_val_score(clf, X_train_, y_train_, cv=kfold, scoring="neg_mean_squared_error",n_jobs=ncores)
    tree_rmse_scores = np.sqrt(-scores)
    df_scores[name] =  tree_rmse_scores
  return df_scores

def cross_validation_mse_continuous(names_,classifiers_,X_train_,y_train_,cv_,ncores):
  df_scores = pd.DataFrame()
  for name, clf in zip(names_, classifiers_): 
    if "Keras" in name:
      ncores=1
    scores = cross_val_score(clf, X_train_, y_train_, cv=cv_, scoring="neg_mean_squared_error",n_jobs=ncores)
    tree_rmse_scores = np.sqrt(-scores)
    df_scores[name] =  tree_rmse_scores
  return df_scores

def plot_cross_validation_mse(names_,df_scores_,suffix_,folder):
  figure1 = plt.figure(figsize=(20,15))
  i=1
  for name in names_:
    ax = plt.subplot(2, (len(names_)+1)/2, i)  
    ax.set_xlim([0,(df_scores_[name].mean()*2)])    
#     bin_values = np.arange(start=0, stop=maxx, step=stepsize)  
    l=plt.hist(df_scores_[name].values, color="blue")
    #mystring='$\mu$=%8.2f, \sigma$=%8.2f' % (df_scores_[name].mean(),df_scores_[name].std())
    mystring='$\mu=%8.2f, \sigma=%8.2f$' % (df_scores_[name].mean(),df_scores_[name].std())
    plt.text(0.2, 4., mystring,fontsize=16)
    plt.title(name, fontsize=16)   
    plt.xlabel("scores RMSE",fontsize=16) 
    plt.ylim(0, 5)
#     plt.xlim(minx,maxx)
    plt.ylabel("Entries",fontsize=16)
    figure1.subplots_adjust(hspace=.5)
    i += 1
  plotname=folder+'/scoresRME%s.png' % (suffix_)
  plt.savefig(plotname)

def plotdistributiontarget(names_,testset,myvariablesy,suffix_,folder):
  figure1 = plt.figure(figsize=(20,15))
  i=1
  for name in names_:
    ax = plt.subplot(2, (len(names_)+1)/2, i)  
    l=plt.hist(testset[myvariablesy].values, color="blue",bins=100,label="true value")
    l=plt.hist(testset['y_test_prediction'+name].values, color="red",bins=100,label="predicted value")
    plt.title(name, fontsize=16)   
    plt.xlabel(myvariablesy,fontsize=16) 
    plt.ylabel("Entries",fontsize=16)
    figure1.subplots_adjust(hspace=.5)
    i += 1
  plt.legend(loc="center right")
  plotname=folder+'/distributionregression%s.png' % (suffix_)
  plt.savefig(plotname)

def plotscattertarget(names_,testset,myvariablesy,suffix_,folder):
  figure1 = plt.figure(figsize=(20,15))
  i=1
  for name in names_:
    ax = plt.subplot(2, (len(names_)+1)/2, i)  
    l=plt.scatter(testset[myvariablesy].values,testset['y_test_prediction'+name].values, color="blue")
    plt.title(name, fontsize=16)   
    plt.xlabel(myvariablesy + "true",fontsize=16) 
    plt.ylabel(myvariablesy + "predicted",fontsize=16) 
    figure1.subplots_adjust(hspace=.5)
    i += 1
  plotname=folder+'/scatterplotregression%s.png' % (suffix_)
  plt.savefig(plotname)


def confusion(mylistvariables_,names_,classifiers_,suffix_,X_train,y_train,cv,folder):
  figure1 = plt.figure(figsize=(25,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

  i=1
  for name, clf in zip(names_, classifiers_):
    ax = plt.subplot(2, (len(names_)+1)/2, i)  
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=cv)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    df_cm = pd.DataFrame(norm_conf_mx,range(2),range(2))
    sn.set(font_scale=1.4)#for label size
    ax.set_title(name+"tot diag=0")
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.xaxis.set_ticklabels(['signal', 'background']); ax.yaxis.set_ticklabels(['signal', 'background']);

    i += 1
  plotname=folder+'/confusion_matrix%s_Diag0.png' % (suffix_)
  plt.savefig(plotname)
  
  figure2 = plt.figure(figsize=(20,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

  i=1
  for name, clf in zip(names_, classifiers_):
    ax = plt.subplot(2, (len(names_)+1)/2, i)  
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=cv)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    df_cm = pd.DataFrame(norm_conf_mx,range(2),range(2))
    sn.set(font_scale=1.4)#for label size
    ax.set_title(name)
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.xaxis.set_ticklabels(['signal', 'background']); ax.yaxis.set_ticklabels(['signal', 'background']);

    i += 1
  plotname=folder+'/confusion_matrix%s.png' % (suffix_)
  plt.savefig(plotname)

def precision_recall(mylistvariables_,names_,classifiers_,suffix_,X_train,y_train,cv,folder):
  figure1 = plt.figure(figsize=(25,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

  i=1
  for name, clf in zip(names_, classifiers_):
    ax = plt.subplot(2, (len(names_)+1)/2, i)  
    y_proba = cross_val_predict(clf, X_train, y_train, cv=cv,method="predict_proba")
    y_scores = y_proba[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision=TP/(TP+FP)")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall=TP/(TP+FN)")
    plt.xlabel("probability",fontsize=16)
    ax.set_title(name,fontsize=16)
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    i += 1
  plotname=folder+'/precision_recall%s.png' % (suffix_)
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
    plt.plot(fpr, tpr, alpha=0.3, label='ROC %s (AUC = %0.2f)' % (names_[i-1], roc_auc), linewidth=4.0)
    plt.legend(loc="lower center",  prop={'size':18})
    i += 1
  plotname=folder+'/ROCcurve%s.png' % (suffix_)
  plt.savefig(plotname)
  
def plot_learning_curves(names_, classifiers_,suffix_,folder,X,y,npoints,ystring='RMSE', threshold=0.5):
  figure1 = plt.figure(figsize=(20,15))
  i=1
  X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)
  for name, clf in zip(names_, classifiers_):
    ax = plt.subplot(2, (len(names_)+1)/2, i)  
    train_errors, val_errors = [],[]
    max=len(X_train)
    min=100
    step_=int((max-min)/npoints)
    arrayvalues=np.arange(start=min,stop=max,step=step_)
    ytype = {'RMSE':0, 'f1score':1, 'sig':2, 'bkg':3}
    for m in arrayvalues:
      clf.fit(X_train[:m],y_train[:m])
      y_train_predict = np.transpose((clf.predict_proba(X_train[:m]) >= threshold).astype(int))[1]
      y_val_predict = np.transpose((clf.predict_proba(X_val) >= threshold).astype(int))[1]
      if (ytype.get(ystring,-1)==1):
        yMetric_train = f1_score(y_train_predict,y_train[:m])
        yMetric_val = f1_score(y_val_predict,y_val)
      elif (ytype.get(ystring,-1)==2):
        yMetric_train =  precision_score(y_train_predict,y_train[:m])
        yMetric_val = precision_score(y_val_predict,y_val)
      elif (ytype.get(ystring,-1)==3):
        tn, fp, fn, tp = confusion_matrix(y_train_predict,y_train[:m]).ravel()
        yMetric_train = tn / (tn+fn)
        tn, fp, fn, tp = confusion_matrix(y_val_predict,y_val).ravel()
        yMetric_val = tn / (tn+fn)
      else:
        yMetric_train = mean_squared_error(y_train_predict,y_train[:m])
        yMetric_val = mean_squared_error(y_val_predict,y_val)  
      train_errors.append(yMetric_train)
      val_errors.append(yMetric_val)
    ax.set_ylim([0,np.amax(np.sqrt(val_errors))*2])    
    plt.plot(arrayvalues,np.sqrt(train_errors),"r-+",linewidth=3,label="training")
    plt.plot(arrayvalues,np.sqrt(val_errors),"b-",linewidth=3,label="testing")
    plt.title(name, fontsize=16)
    plt.xlabel("Training set size",fontsize=16)
    yAxisLabel = ("RMSE", "f1 score", "signal efficiency","background efficieny")
    plt.ylabel(yAxisLabel[ytype.get(ystring,0)],fontsize=16)
    figure1.subplots_adjust(hspace=.5)
    plt.legend(loc="lower center",  prop={'size':18})
    i += 1
  suffix_=suffix_+'_'+str(threshold)
  typesuffix = ("RMSE", "f1score", "sig","bkg")
  suffix_= suffix_+"_"+typesuffix[ytype.get(ystring,0)]
  plotname=folder+'/learning_curve%s.png' % (suffix_)
  plt.savefig(plotname)
