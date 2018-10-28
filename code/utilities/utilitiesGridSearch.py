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


def do_gridsearch(namesCV_,classifiersCV_,mylistvariables_,param_gridCV_,X_train_,y_train_,cv_,ncores):
  grid_search_models_=[]
  grid_search_bests_=[]
  for nameCV, clfCV, gridCV in zip(namesCV_, classifiersCV_,param_gridCV_):
    grid_search = GridSearchCV(clfCV, gridCV, cv=cv_,scoring='neg_mean_squared_error',n_jobs=ncores)
    grid_search_model=grid_search.fit(X_train_, y_train_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
      print(np.sqrt(-mean_score), params)
    grid_search_best=grid_search.best_estimator_.fit(X_train_, y_train_)
    grid_search_models_.append(grid_search_model)
    grid_search_bests_.append(grid_search_best)
  return grid_search_models_,grid_search_bests_
    

def plot_gridsearch(namesCV_,change_,grid_search_models_,output_,suffix_):

  for nameCV,change,gridCV in zip(namesCV_,change_,grid_search_models_):
    figure = plt.figure(figsize=(10,10))
    plot.grid_search(gridCV.grid_scores_, change=change,kind='bar')
    plt.title('Grid search results '+ nameCV, fontsize=17)
    plt.ylim(-0.8,0)
    plt.ylabel('negative mean squared error',fontsize=17)
    plt.xlabel(change,fontsize=17)
    plotname=output_+"/GridSearchResults"+nameCV+suffix_+".png"
    plt.savefig(plotname)
