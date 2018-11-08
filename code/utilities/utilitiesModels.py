from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression

###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################

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
from sklearn.feature_extraction import DictVectorizer
from matplotlib.colors import ListedColormap
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def getclassifiers(MLtype):
  classifiers=[]
  names =[]
  if (MLtype=="BinaryClassification"):
    classifiers = [
      GradientBoostingClassifier(learning_rate=0.01, n_estimators=2500, max_depth=1),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), AdaBoostClassifier(),DecisionTreeClassifier(max_depth=5)
#       LinearSVC(C=1, loss="hinge"),SVC(kernel="rbf", gamma=5, C=0.001), LogisticRegression()
    ]
                                        
    names = [
      "ScikitGradientBoostingClassifier","ScikitRandom_Forest","ScikitAdaBoost","ScikitDecision_Tree"
#       "ScikitLinearSVC", "ScikitSVC_rbf","ScikitLogisticRegression"
    ]

  if (MLtype=="Regression"):
    classifiers = [
      LinearRegression()
    ]
                                        
    names = [
      "Scikit_linear_regression"
    ]
  return classifiers, names

def getclassifiersDNN(MLtype,lengthInput):
  classifiers=[]
  names =[]
  
  if (MLtype=="BinaryClassification"):
    def create_model_Sequential():
      model = Sequential()
      model.add(Dense(12, input_dim=lengthInput, activation='relu'))
      model.add(Dense(8, activation='relu'))
      model.add(Dense(1, activation='sigmoid'))
      model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
      return model
  
    classifiers = [KerasClassifier(build_fn=create_model_Sequential, epochs=1000, batch_size=50, verbose=0)]
    names = ["KerasSequential"]
    
  if (MLtype=="Regression"):
    print ("No Keras models implemented for Regression")
  return classifiers,names 

def fit(names_, classifiers_,X_train_,y_train_):
  trainedmodels_=[]
  for name, clf in zip(names_, classifiers_):
    clf.fit(X_train_, y_train_)
    trainedmodels_.append(clf)
  return trainedmodels_

def test(MLtype,names_,trainedmodels_,test_set_,mylistvariables_,myvariablesy_):
  X_test_=test_set_[mylistvariables_]
  y_test_=test_set_[myvariablesy_].values.reshape(len(X_test_),)
  test_set_[myvariablesy_] = pd.Series(y_test_, index=test_set_.index)
  for name, model in zip(names_, trainedmodels_):
    y_test_prediction=[]
    y_test_prob=[]
    y_test_prediction=model.predict(X_test_)
    y_test_prediction=y_test_prediction.reshape(len(y_test_prediction),)
    test_set_['y_test_prediction'+name] = pd.Series(y_test_prediction, index=test_set_.index)

    if (MLtype=="BinaryClassification"):
      y_test_prob=model.predict_proba(X_test_)[:,1]
      test_set_['y_test_prob'+name] = pd.Series(y_test_prob, index=test_set_.index)
  return test_set_

def savemodels(names_,trainedmodels_,folder_,suffix_):
    for name, model in zip(names_, trainedmodels_):
      if "Keras" in name: 
        architecture_file= folder_+"/"+name+suffix_+"_architecture.json"
        weights_file= folder_+"/"+name+suffix_+"_weights.h5"
        arch_json = model.model.to_json()
        with open(architecture_file, 'w') as json_file:
          json_file.write(arch_json)
        model.model.save_weights(weights_file)
      else: 
        fileoutmodel = folder_+"/"+name+suffix_+".sav"
        pickle.dump(model, open(fileoutmodel, 'wb'))
        
def readmodels(names_,folder_,suffix_):
  trainedmodels_=[]
  for name in names_:
    fileinput = folder_+"/"+name+suffix_+".sav"
    model = pickle.load(open(fileinput, 'rb'))
    trainedmodels_.append(model)
  return trainedmodels_



def importanceplotall(mylistvariables_,names_,trainedmodels_,suffix_,folder):
  figure1 = plt.figure(figsize=(20,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

  i=1
  for name, model in zip(names_, trainedmodels_):
    if "SVC" in name: 
      continue
    if "Logistic" in name: 
      continue
    ax = plt.subplot(2, (len(names_)+1)/2, i)  
    #plt.subplots_adjust(left=0.3, right=0.9)
    feature_importances_ = model.feature_importances_
    y_pos = np.arange(len(mylistvariables_))
    ax.barh(y_pos, feature_importances_, align='center',color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mylistvariables_, fontsize=17)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance',fontsize=17)
    ax.set_title('Importance features '+name, fontsize=17)
    ax.xaxis.set_tick_params(labelsize=17)
    plt.xlim(0, 0.7)
    i += 1
  plotname=folder+'/importanceplotall%s.png' % (suffix_)
  plt.savefig(plotname)


def decisionboundaries(names_,trainedmodels_,suffix_,X_train_,y_train_,folder):
  mylistvariables_=X_train_.columns.tolist()
  dictionary_train = X_train_.to_dict(orient='records')
  vec = DictVectorizer()
  X_train_array_ = vec.fit_transform(dictionary_train).toarray()

  figure = plt.figure(figsize=(20,15))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)
  h = .10
  cm = plt.cm.RdBu
  cm_bright = ListedColormap(['#FF0000', '#0000FF'])

  x_min, x_max = X_train_array_[:, 0].min() - .5, X_train_array_[:, 0].max() + .5
  y_min, y_max = X_train_array_[:, 1].min() - .5, X_train_array_[:, 1].max() + .5
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

  i=1
  for name, model in zip(names_, trainedmodels_):
    if hasattr(model, "decision_function"):
      Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
      Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
      
    ax = plt.subplot(2, (len(names_)+1)/2, i)  

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    # Plot also the training points
    ax.scatter(X_train_array_[:, 0], X_train_array_[:, 1], c=y_train_, cmap=cm_bright, edgecolors='k',alpha=0.3)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    score = model.score(X_train_, y_train_)
    ax.text(xx.max() - .3, yy.min() + .3, ('accuracy=%.2f' % score).lstrip('0'), size=15,horizontalalignment='right',verticalalignment='center')
    ax.set_title(name,fontsize=17)
    ax.set_ylabel(mylistvariables_[1],fontsize=17)
    ax.set_xlabel(mylistvariables_[0],fontsize=17)
    figure.subplots_adjust(hspace=.5)
    i += 1
  plotname=folder+'/decisionboundaries%s.png' % (suffix_)
  plt.savefig(plotname)

