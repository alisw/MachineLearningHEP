###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################
from myimports import *
from utilitiesModels import getclassifiers,fit,test,savemodels,importanceplotall,decisionboundaries,getclassifiersDNN,getclassifiersXGBoost,apply
from DataBaseMLparameters import getvariablestraining,getvariablesothers,getvariableissignal,getvariabletarget,getvariablesall,getvariablecorrelation,getgridsearchparameters,getDataMCfiles,getTreeName,prepareMLsample,getvariablesBoundaries
from utilitiesPerformance import precision_recall,plot_learning_curves,confusion,cross_validation_mse,cross_validation_mse_continuous,plot_cross_validation_mse,plotdistributiontarget,plotscattertarget
from utilitiesPCA import GetPCADataFrameAndPC,GetDataFrameStandardised,plotvariancePCA
from utilitiesCorrelations import scatterplot,correlationmatrix,vardistplot
from utilitiesGeneral import splitdataframe_sigbkg,checkdir,getdataframeDataMC,filterdataframeDataMC,createstringselection,writeTree
from utilitiesGridSearch import do_gridsearch,plot_gridsearch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utilitiesOptimisation import studysignificance

############### choose your ML method ################
nevents=5000
MLtype="BinaryClassification" #other options are "Regression", "BinaryClassification"
MLsubtype="HFmeson" #other options are "PID","HFmeson","test"
optionanalysis="Ds" #other options are "Ds, Bplus,Lc,PIDKaon,PIDPion,testregression

############### choose the skimming parameters for your dataset ################
var_skimming=["pt_cand_ML"] #other options are "pdau0_ML" in case of PID
varmin=[2]
varmax=[4]

############### choose if you want scikit or keras models or both ################
activateScikitModels=1; activateXGBoostModels=1; activateKerasModels=0
loadsampleOption=0 #0=loadfromTree,1=loadfromDF,2=loadyourownDFfortesting
docorrelation=0; doStandard=0; doPCA=0
dotraining=1; dotesting=1; doapplytodata=1
doLearningCurve=1; docrossvalidation=1
doROCcurve=1; doOptimisation=0; doBinarySearch=0; doBoundary=0; doimportance=1 #classification specifics
doplotdistributiontargetregression=0 #regression specifics

ncores=-1

if (MLtype=="Regression"):
  print ("the following tests cannot be performed since meaningless for regression analysis or not yet implemented:")
  print ("- doROCcurve")
  print ("- doOptimisation")
  print ("- doBinarySearch")
  print ("- doBoundary")
  print ("- doImportance")
  doROCcurve=0
  doOptimisation=0
  doBinarySearch=0
  doBoundary=0
  activateKerasModels=0
  doimportance=0
  
if (MLtype=="BinaryClassification"):
  print ("the following tests cannot be performed since meaningless for classification analysis or not yet implemented:")
  print ("- plotdistributiontargetregression")
  doplotdistributiontargetregression=0
  

string_selection=createstringselection(var_skimming,varmin,varmax)
suffix="Nevents%d_%s%s_%s" % (nevents,MLtype,optionanalysis,string_selection)

dataframe="dataframes_%s" % (suffix)
plotdir="plots_%s" % (suffix)
output="output_%s" % (suffix)
checkdir(dataframe)
checkdir(plotdir)
checkdir(output)

classifiers=[]
classifiersScikit=[]
classifiersDNN=[]

names=[]
namesScikit=[]
namesDNN=[]

trainedmodels=[]
X_train=[]
y_train=[]
  
mylistvariables=getvariablestraining(optionanalysis)
mylistvariablesothers=getvariablesothers(optionanalysis)
myvariablessignal=getvariableissignal(optionanalysis)
myvariablesy=getvariabletarget(optionanalysis)
mylistvariablesx,mylistvariablesy=getvariablecorrelation(optionanalysis)
mylistvariablesall=getvariablesall(optionanalysis)


if(loadsampleOption==0): 
  fileData,fileMC=getDataMCfiles(optionanalysis)
  trename=getTreeName(optionanalysis)
  dataframeData,dataframeMC=getdataframeDataMC(fileData,fileMC,trename,mylistvariablesall)
  dataframeData,dataframeMC=filterdataframeDataMC(dataframeData,dataframeMC,var_skimming,varmin,varmax)  
  print (len(dataframeData))
  print (len(dataframeMC))
  ## prepare ML sample
  dataframeML,dataframe_sig,dataframe_bkg=prepareMLsample(MLtype,MLsubtype,optionanalysis,dataframeData,dataframeMC,nevents)
  dataframeML=shuffle(dataframeML)
  ### split in training/testing sample
  train_set, test_set = train_test_split(dataframeML, test_size=0.2, random_state=42)
  ### save the dataframes
  train_set.to_pickle(dataframe+"/dataframetrainsampleN%s.pkl" % (suffix))
  test_set.to_pickle(dataframe+"/dataframetestsampleN%s.pkl" % (suffix))
  X_train= train_set[mylistvariables]
  y_train=train_set[myvariablesy]

if(loadsampleOption==1): 
  train_set = pd.read_pickle(dataframe+"/dataframetrainsampleN%s.pkl" % (suffix))
  test_set = pd.read_pickle(dataframe+"/dataframetestsampleN%s.pkl" % (suffix))
  print ("dimension of the dataset",len(train_set))
  X_train= train_set[mylistvariables]
  y_train=train_set[myvariablesy]

if(loadsampleOption==2): 
  mylistvariables=["volume"]
  mylistvariablesothers=[]
  myvariablessignal="signal"
  myvariablesy='mass'
  mylistvariablesx=["mass"]
  mylistvariablesy=["volume"]
  mylistvariablesall=["volume","mass","signal"]
  
  ntrials=10000
  X=2*np.random.rand(ntrials,1)
  print (X)
  signal=np.ones(shape=(ntrials,1))
  density=20;
  y=density*X+np.random.randn(ntrials,1)
  X=np.c_[X,y]
  X=np.c_[X,signal]
  print ("finish df creation")
  dataframeML = pd.DataFrame(X,columns=mylistvariablesall)
  print (dataframeML)

  train_set, test_set = train_test_split(dataframeML, test_size=0.2, random_state=42)
  X_train= train_set[mylistvariables]
  y_train=train_set[myvariablesy]

  
if(docorrelation==1):
  train_set_ptsel_sig,train_set_ptsel_bkg=splitdataframe_sigbkg(train_set,myvariablessignal)
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


if (activateScikitModels==1):
  classifiersScikit,namesScikit=getclassifiers(MLtype)
  classifiers=classifiers+classifiersScikit
  names=names+namesScikit

if (activateXGBoostModels==1):
  classifiersXGBoost,namesXGBoost=getclassifiersXGBoost(MLtype)
  classifiers=classifiers+classifiersXGBoost
  names=names+namesXGBoost

if (activateKerasModels==1):
  classifiersDNN,namesDNN=getclassifiersDNN(MLtype,len(X_train.columns))
  classifiers=classifiers+classifiersDNN
  names=names+namesDNN

if (dotraining==1):
  print (names)
  trainedmodels=fit(names, classifiers,X_train,y_train)
  savemodels(names,trainedmodels,output,suffix)
  
if (dotesting==1):
  filenametest_set_ML=output+"/testsample%sMLdecision.pkl" % (suffix)
  filenametest_set_ML_root=output+"/testsample%sMLdecision.root" % (suffix)
  ntuplename=getTreeName(optionanalysis)+"Tested"
  test_setML=test(MLtype,names,trainedmodels,test_set,mylistvariables,myvariablesy)
  test_setML.to_pickle(filenametest_set_ML)
  writeTree(filenametest_set_ML_root,ntuplename,test_setML)

if (doapplytodata==1):
  fileData,fileMC=getDataMCfiles(optionanalysis)
  trename=getTreeName(optionanalysis)
  dataframeData,dataframeMC=getdataframeDataMC(fileData,fileMC,trename,mylistvariablesall)
  dataframeDataML=apply(MLtype,names,trainedmodels,dataframeData,mylistvariables)
  dataframeMCML=apply(MLtype,names,trainedmodels,dataframeMC,mylistvariables)
  filenameData_ML_root=output+"/AnalysisRoot%sDataMLdecision.root" % (suffix)
  filenameMC_ML_root=output+"/AnalysisRoot%sMCMLdecision.root" % (suffix)
  ntuplename=getTreeName(optionanalysis)+"Tested"
  writeTree(filenameData_ML_root,ntuplename,dataframeDataML)
  writeTree(filenameMC_ML_root,ntuplename,dataframeMCML)

if (docrossvalidation==1): 
  df_scores=[]
  if (MLtype=="Regression" ):
    df_scores=cross_validation_mse_continuous(names,classifiers,X_train,y_train,5,ncores)
  if (MLtype=="BinaryClassification" ):
    df_scores=cross_validation_mse(names,classifiers,X_train,y_train,5,ncores)
  plot_cross_validation_mse(names,df_scores,suffix,plotdir)


if (doLearningCurve==1):
#   confusion(mylistvariables,names,classifiers,suffix,X_train,y_train,5)
  plot_learning_curves(names,classifiers,suffix,plotdir,X_train,y_train,10)
  
if (doROCcurve==1):
  precision_recall(mylistvariables,names,classifiers,suffix,X_train,y_train,5,plotdir)

if(doOptimisation==1):
  if not ((MLsubtype=="HFmeson") & (optionanalysis=="Ds")):
    print ("==================ERROR==================")
    print ("Optimisation is not implemented for this classification problem. The code is going to fail")
    sys.exit()   
  studysignificance(optionanalysis,varmin[0],varmax[0],test_set,names,myvariablesy,suffix,plotdir) 

if (doBoundary==1):
  classifiersScikit2var,names2var=getclassifiers(MLtype)
  classifiersDNN2var,namesDNN2var=getclassifiersDNN(MLtype,2)
  classifiers2var=classifiersScikit2var+classifiersDNN2var
  names2var=names2var+namesDNN2var
  X_train_boundary=train_set[getvariablesBoundaries(optionanalysis)]
  trainedmodels2var=fit(names2var,classifiers2var,X_train_boundary,y_train)
  mydecisionboundaries=decisionboundaries(names2var,trainedmodels2var,suffix+"2var",X_train_boundary,y_train,plotdir)
  X_train_2PC,pca=GetPCADataFrameAndPC(X_train,2)
  trainedmodelsPCA=fit(names2var, classifiers2var,X_train_2PC,y_train)
  mydecisionboundaries=decisionboundaries(names2var,trainedmodelsPCA,suffix+"2PCA",X_train_2PC,y_train,plotdir)

if (doBinarySearch==1):
  namesCV,classifiersCV,param_gridCV,changeparameter=getgridsearchparameters(optionanalysis)
  grid_search_models,grid_search_bests=do_gridsearch(namesCV,classifiersCV,mylistvariables,param_gridCV,X_train,y_train,3,ncores)
  savemodels(namesCV,grid_search_models,output,"GridSearchCV"+suffix)
  plot_gridsearch(namesCV,changeparameter,grid_search_models,plotdir,suffix)

if (doimportance==1):
  importanceplotall(mylistvariables,namesScikit+namesXGBoost,classifiersScikit+classifiersXGBoost,suffix,plotdir)
  
if (doplotdistributiontargetregression==1):
  plotdistributiontarget(names,test_setML,myvariablesy,suffix,plotdir)
  plotscattertarget(names,test_setML,myvariablesy,suffix,plotdir)
  

