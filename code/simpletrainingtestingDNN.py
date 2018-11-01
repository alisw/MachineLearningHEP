###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M.Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################
from myimports import *
from utilitiesModels import getclassifiers,fit,test,savemodels,importanceplotall,decisionboundaries
from BinaryMultiFeaturesClassification import getvariablestraining,getvariablesothers,getvariableissignal,getvariablesall,getvariablecorrelation,getgridsearchparameters,getDataMCfiles,getTreeName,prepareMLsample,getvariablesBoundaries
from utilitiesPerformance import precision_recall,plot_learning_curves,confusion,precision_recall,plot_learning_curves,cross_validation_mse,plot_cross_validation_mse
from utilitiesPCA import GetPCADataFrameAndPC,GetDataFrameStandardised,plotvariancePCA
from utilitiesCorrelations import scatterplot,correlationmatrix,vardistplot
from utilitiesGeneral import filterdataframe_pt,splitdataframe_sigbkg,checkdir,getdataframe,getdataframeDataMC,filterdataframe,filterdataframeDataMC,createstringselection
from utilitiesGridSearch import do_gridsearch,plot_gridsearch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

############### this is the only place where you should change parameters ################
classtype="HFmeson"
optionClassification="Ds"
var_skimming=["pt_cand_ML"]
# classtype="PID"
# optionClassification="PIDKaon"
# var_skimming=["pdau0_ML"]
nevents=1000
varmin=[2]
varmax=[5]
string_selection=createstringselection(var_skimming,varmin,varmax)
suffix="Nevents%d_BinaryClassification%s_%s" % (nevents,optionClassification,string_selection)

############### activate your channel ################
dosampleprep=1
docorrelation=0
doStandard=0
doPCA=0
dotraining=0
doimportance=0
dotesting=0
docrossvalidation=0
doRoCLearning=0
doBoundary=0
doBinarySearch=0
doDNN=1

ncores=-1

dataframe="dataframes_%s" % (suffix)
plotdir="plots_%s" % (suffix)
output="output_%s" % (suffix)
checkdir(dataframe)
checkdir(plotdir)
checkdir(output)


classifiers, names=getclassifiers()
mylistvariables=getvariablestraining(optionClassification)
mylistvariablesothers=getvariablesothers(optionClassification)
myvariablesy=getvariableissignal(optionClassification)
mylistvariablesx,mylistvariablesy=getvariablecorrelation(optionClassification)
mylistvariablesall=getvariablesall(optionClassification)


if(dosampleprep==1): 
  fileData,fileMC=getDataMCfiles(optionClassification)
  trename=getTreeName(optionClassification)
  dataframeData,dataframeMC=getdataframeDataMC(fileData,fileMC,trename,mylistvariablesall)
  dataframeData,dataframeMC=filterdataframeDataMC(dataframeData,dataframeMC,var_skimming,varmin,varmax)  
  ## prepare ML sample
  dataframeML=prepareMLsample(classtype,optionClassification,dataframeData,dataframeMC,nevents)
  dataframeML=shuffle(dataframeML)
  ### split in training/testing sample
  train_set, test_set = train_test_split(dataframeML, test_size=0.2, random_state=42)
  ### save the dataframes
  train_set.to_pickle(dataframe+"/dataframetrainsampleN%s.pkl" % (suffix))
  test_set.to_pickle(dataframe+"/dataframetestsampleN%s.pkl" % (suffix))

train_set = pd.read_pickle(dataframe+"/dataframetrainsampleN%s.pkl" % (suffix))
test_set = pd.read_pickle(dataframe+"/dataframetestsampleN%s.pkl" % (suffix))

print ("dimension of the dataset",len(train_set))

X_train= train_set[mylistvariables]
y_train=train_set[myvariablesy]
X_test= test_set[mylistvariables]
y_test=test_set[myvariablesy]

trainedmodels=[]

if(docorrelation==1):
  train_set_ptsel_sig,train_set_ptsel_bkg=splitdataframe_sigbkg(train_set,myvariablesy)
  vardistplot(train_set_ptsel_sig, train_set_ptsel_bkg,mylistvariablesall,plotdir)
  scatterplot(train_set_ptsel_sig, train_set_ptsel_bkg,mylistvariablesx,mylistvariablesy,plotdir)
  correlationmatrix(train_set_ptsel_sig,plotdir,"signal")
  correlationmatrix(train_set_ptsel_bkg,plotdir,"background")

if (doStandard==1):
  X_train=GetDataFrameStandardised(X_train)

if (doPCA==1):
  n_pca=9
  X_train,pca=GetPCADataFrameAndPC(X_train,n_pca)
  plotvariancePCA(pca,plotdir)

if (dotraining==1):
  trainedmodels=fit(names, classifiers,X_train,y_train)
  savemodels(names,trainedmodels,output,suffix)
  
if (doimportance==1):
  importanceplotall(mylistvariables,names,trainedmodels,suffix,plotdir)
  
if (docrossvalidation==1): 
  df_scores=cross_validation_mse(names,classifiers,X_train,y_train,10,ncores)
  plot_cross_validation_mse(names,df_scores,suffix,plotdir)

if (doRoCLearning==1):
#   confusion(mylistvariables,names,classifiers,suffix,X_train,y_train,5)
  precision_recall(mylistvariables,names,classifiers,suffix,X_train,y_train,5,plotdir)
  plot_learning_curves(names,classifiers,suffix,plotdir,X_train,y_train,100,3000,300)
  
if (dotesting==1):
  filenametest_set_ML=output+"/testsample%sMLdecision.pkl" % (suffix)
  ntuplename="fTreeFlagged%s" % (optionClassification)
  test_setML=test(names,trainedmodels,X_test,test_set)
  test_set.to_pickle(filenametest_set_ML)

if (doBoundary==1):
  X_train_boundary=train_set[getvariablesBoundaries(optionClassification)]
  trainedmodels2var=fit(names, classifiers,X_train_boundary,y_train)
  mydecisionboundaries=decisionboundaries(names,trainedmodels2var,suffix+"2var",X_train_boundary,y_train,plotdir)
#   X_train_2PC,pca=GetPCADataFrameAndPC(X_train,2)
#   trainedmodelsPCA=fit(names, classifiers,X_train_2PC,y_train)
#   mydecisionboundaries=decisionboundaries(names,trainedmodelsPCA,suffix+"2PCA",X_train_2PC,y_train,plotdir)

if (doBinarySearch==1):
  namesCV,classifiersCV,param_gridCV,changeparameter=getgridsearchparameters(optionClassification)
  grid_search_models,grid_search_bests=do_gridsearch(namesCV,classifiersCV,mylistvariables,param_gridCV,X_train,y_train,3,ncores)
  savemodels(names,grid_search_models,output,"GridSearchCV"+suffix)
  plot_gridsearch(namesCV,changeparameter,grid_search_models,plotdir,suffix)

if (doDNN==1):
  from sklearn.metrics import roc_curve, auc
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.wrappers.scikit_learn import KerasClassifier
  from sklearn.model_selection import StratifiedKFold
  from sklearn.model_selection import cross_val_score
  import keras

#   feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
#   dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=2, feature_columns=feature_cols)
#   dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)  # if TensorFlow >= 1.1
#   dnn_clf.fit(np.array(X_train, dtype = 'float32'),np.array(y_train, dtype = 'int64'), batch_size=50, steps=50000)
# 
#   y_pred = dnn_clf.predict(np.array(X_test, dtype = 'float32'))
#   y_test_prediction=y_pred['classes']
#   y_test_prob=y_pred['probabilities'][:,1]
#   
#   print (X_test.shape)
#   print (y_test_prob.shape)
#   
#   aucs=[]
#   fpr, tpr, thresholds_forest = roc_curve(y_test,y_test_prob)
#   roc_auc = auc(fpr, tpr)
#   aucs.append(roc_auc)
#   plt.xlabel('False Positive Rate or (1 - Specifity)',fontsize=20)
#   plt.ylabel('True Positive Rate or (Sensitivity)',fontsize=20)
#   plt.title('Receiver Operating Characteristic',fontsize=20)
#   plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC %s (AUC = %0.2f)' % ("DNN", roc_auc), linewidth=4.0)
#   plt.legend(loc="lower center",  prop={'size':18})
#     
    
#   model = Sequential()
#   model.add(Dense(12, input_dim=7, activation='relu'))
#   model.add(Dense(8, activation='relu'))
#   model.add(Dense(1, activation='sigmoid'))
#   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#   model.fit(X_train,y_train, epochs=10, batch_size=32)
  
  
  def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=9, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  model = KerasClassifier(build_fn=create_model, epochs=1000, batch_size=50, verbose=0)
  model.fit(X_train,y_train)

  architecture_file="SequentialPIDkaon.json"
  weights_file="weightSequentialPIDkaon.h5"
  arch_json = model.model.to_json()
  with open(architecture_file, 'w') as json_file:
    json_file.write(arch_json)
  # Save weights only.
  model.model.save_weights(weights_file)


  y_test_prediction=model.predict(X_test)
  y_test_prob=model.predict_proba(X_test)[:,1]
  y_test_prediction.reshape(len(y_test_prediction),)
  print (y_test_prob.shape)
  
  aucs=[]
  fpr, tpr, thresholds_forest = roc_curve(y_test,y_test_prob)
  roc_auc = auc(fpr, tpr)
  aucs.append(roc_auc)
  plt.xlabel('False Positive Rate or (1 - Specifity)',fontsize=20)
  plt.ylabel('True Positive Rate or (Sensitivity)',fontsize=20)
  plt.title('Receiver Operating Characteristic',fontsize=20)
  plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC %s (AUC = %0.2f)' % ("DNN", roc_auc), linewidth=4.0)
  plt.legend(loc="lower center",  prop={'size':18})
  plt.show()

  test_set['y_test_predictionDNN_TensorFlow'] = pd.Series(y_test_prediction, index=test_set.index)
  test_set['y_test_probDNN_TensorFlow'] = pd.Series(y_test_prob, index=test_set.index)
