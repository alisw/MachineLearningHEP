###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M.Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################
from myimports import *
from utilitiesModels import getclassifiers,fit,test,savemodels,importanceplotall,decisionboundaries
from BinaryMultiFeaturesClassification import getvariablestraining,getvariablesothers,getvariableissignal,getvariablesall,getvariablecorrelation,getgridsearchparameters,getDataMCfiles,getTreeName,prepareMLsample
from utilitiesPerformance import precision_recall,plot_learning_curves,confusion,precision_recall,plot_learning_curves,cross_validation_mse,plot_cross_validation_mse
from utilitiesPCA import GetPCADataFrameAndPC,GetDataFrameStandardised,plotvariancePCA
from utilitiesCorrelations import scatterplot,correlationmatrix,vardistplot
from utilitiesGeneral import filterdataframe_pt,splitdataframe_sigbkg,checkdir,getdataframe,getdataframeDataMC,filterdataframe,filterdataframeDataMC
from utilitiesGridSearch import do_gridsearch,plot_gridsearch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

############### this is the only place where you should change parameters ################
optionClassification="Ds"
nevents=500
ptmin=1
ptmax=100
suffix="Nevents%d_BinaryClassification%s_ptmin%d_ptmax%d" % (nevents,optionClassification,ptmin,ptmax)
var_pt="pt_cand_ML"
varmin=[4,4]
varmax=[100,100]
var_skimming=["pt_cand_ML","pt_cand_ML"]

############### activate your channel ################
dosampleprep=1
docorrelation=1
doStandard=0
doPCA=0
dotraining=1
doimportance=0
dotesting=0
docrossvalidation=1
doRoCLearning=0
doBoundary=0
doBinarySearch=0
ncores=-1

##########################################################################################
# var_pt="pt_cand_ML"
# var_signal="signal_ML"
# path = "./plotdir/%.1f_%.1f_GeV"%(ptmin,ptmax)
# checkdir(path)

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
  ### prepare ML sample
  dataframeML=prepareMLsample(optionClassification,dataframeData,dataframeMC,nevents,"old")
  dataframeML=shuffle(dataframeML)
  ### split in training/testing sample
  train_set, test_set = train_test_split(dataframeML, test_size=0.2, random_state=42)
  ### save the dataframes
  dataframeML.to_pickle(dataframe+"/dataframeML%s.pkl" % (suffix))
  dataframeML.to_csv(dataframe+"/dataframeML%s.csv" % (suffix))
  train_set.to_pickle(dataframe+"/dataframetrainsampleN%s.pkl" % (suffix))
  test_set.to_pickle(dataframe+"/dataframetestsampleN%s.pkl" % (suffix))

train_set = pd.read_pickle(dataframe+"/dataframetrainsampleN%s.pkl" % (suffix))
test_set = pd.read_pickle(dataframe+"/dataframetestsampleN%s.pkl" % (suffix))

X_train= train_set[mylistvariables]
y_train=train_set[myvariablesy]
X_test= test_set[mylistvariables]
y_test=test_set[myvariablesy]

trainedmodels=[]

if(docorrelation==1):
  train_set_ptsel_sig,train_set_ptsel_bkg=splitdataframe_sigbkg(train_set,myvariablesy)
  vardistplot(train_set_ptsel_sig, train_set_ptsel_bkg,mylistvariables,plotdir)
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
  mydecisionboundaries=decisionboundaries(names,trainedmodels,suffix,X_train,y_train,plotdir)
  X_train_2PC,pca=GetPCADataFrameAndPC(X_train,2)
  trainedmodels=fit(names, classifiers,X_train_2PC,y_train)
  mydecisionboundaries=decisionboundaries(names,trainedmodels,suffix+"PCAdecomposition",X_train_2PC,y_train,plotdir)

if (doBinarySearch==1):
  namesCV,classifiersCV,param_gridCV,changeparameter=getgridsearchparameters(optionClassification)
  grid_search_models,grid_search_bests=do_gridsearch(namesCV,classifiersCV,mylistvariables,param_gridCV,X_train,y_train,3,ncores)
  savemodels(names,grid_search_models,output,"GridSearchCV"+suffix)
  plot_gridsearch(namesCV,changeparameter,grid_search_models,plotdir,suffix)


