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


def getclassifiers():
  classifiers = [GradientBoostingClassifier(learning_rate=0.01, n_estimators=2500, max_depth=1),
                    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                    AdaBoostClassifier(),DecisionTreeClassifier(max_depth=5)]
                  
  names = ["GradientBoostingClassifier","Random_Forest","AdaBoost","Decision_Tree"]
  return classifiers, names

def getvariablestraining():
  mylistvariables=['d_len_xy_ML','norm_dl_xy_ML','cos_p_ML','cos_p_xy_ML','imp_par_xy_ML','sig_vert_ML',"delta_mass_KK_ML",'cos_PiDs_ML',"cos_PiKPhi_3_ML"]
  return mylistvariables

def getvariablesothers():
  mylistvariablesothers=['inv_mass_ML','pt_cand_ML']
  return mylistvariablesothers

def getvariableissignal():
  myvariablesy='signal_ML'
  return myvariablesy

def getvariablesall():
  mylistvariables=getvariablestraining()
  mylistvariablesothers=getvariablesothers
  myvariablesy=getvariableissignal()
  return mylistvariablesall

def preparestringforuproot(myarray):
  arrayfinal=[]
  for str in myarray:
    arrayfinal.append(str+"*")
  return arrayfinal
    
def fit(names_, classifiers_,X_train_,y_train_):
  for name, clf in zip(names_, classifiers_):
    clf.fit(X_train_, y_train_)
    fileoutmodel = "models/"+name+".sav"
    pickle.dump(clf, open(fileoutmodel, 'wb'))


def test(names_, classifiers_,X_test_,test_set_):
  for name, clf in zip(names_, classifiers_):
    y_test_prediction=[]
    y_test_prob=[]
    fileoutmodel = "models/"+name+".sav"
    model = pickle.load(open(fileoutmodel, 'rb'))
    y_test_prediction=model.predict(X_test_)
    y_test_prob=model.predict_proba(X_test_)[:,1]
    test_set_['y_test_prediction'+name] = pd.Series(y_test_prediction, index=test_set_.index)
    test_set_['y_test_prob'+name] = pd.Series(y_test_prob, index=test_set_.index)
  return test_set_
