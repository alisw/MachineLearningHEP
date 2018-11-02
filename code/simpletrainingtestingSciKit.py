###############################################################
##                                                           ##
##     Software for single-label classification with Scikit  ##
##      Origin: G.M. Innocenti (CERN)(ginnocen@cern.ch)       ##
##                                                           ##
###############################################################
from myimports import *
from utilitiesModels import getclassifiers,fit,test,savemodels,importanceplotall,decisionboundaries
from BinaryMultiFeaturesClassification import getvariablestraining,getvariablesothers,getvariableissignal,getvariablesall,getvariablecorrelation,getgridsearchparameters,getDataMCfiles,getTreeName,prepareMLsample,getvariablesBoundaries,getbackgroudev_testingsample,getFONLLdataframe_FF
from utilitiesPerformance import precision_recall,plot_learning_curves,confusion,precision_recall,plot_learning_curves,cross_validation_mse,plot_cross_validation_mse
from utilitiesPCA import GetPCADataFrameAndPC,GetDataFrameStandardised,plotvariancePCA
from utilitiesCorrelations import scatterplot,correlationmatrix,vardistplot
from utilitiesGeneral import filterdataframe_pt,splitdataframe_sigbkg,checkdir,getdataframe,getdataframeDataMC,filterdataframe,filterdataframeDataMC,createstringselection
from utilitiesGridSearch import do_gridsearch,plot_gridsearch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utilitiesOptimisation import getfonllintegrated,plotfonll,get_efficiency_effnum_effden,plot_efficiency,calculatesignificance,plot_significance

############### this is the only place where you should change parameters ################
classtype="HFmeson"
optionClassification="Ds"
var_skimming=["pt_cand_ML"]
# classtype="PID"
# optionClassification="PIDKaon"
# var_skimming=["pdau0_ML"]
nevents=5000
varmin=[0]
varmax=[100]
string_selection=createstringselection(var_skimming,varmin,varmax)
suffix="Nevents%d_BinaryClassification%s_%s" % (nevents,optionClassification,string_selection)

############### activate your channel ################
dosampleprep=1
docorrelation=0
doStandard=0
doPCA=0
dotraining=1
doimportance=0
dotesting=1
docrossvalidation=0
doRoCLearning=1
doBoundary=0
doBinarySearch=0
doOptimisation=1
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
  
if(doOptimisation==1):
   if((classtype=="HFmeson") & (optionClassification=="Ds")):
     df,FF= getFONLLdataframe_FF(optionClassification)
     plotfonll(df.pt,df.central*FF,optionClassification,suffix,plotdir)
     sig=getfonllintegrated(df,varmin[0],varmax[0])*FF
     bkg=getbackgroudev_testingsample(optionClassification)
     efficiencySig_array,xaxisSig,num_arraySig,den_arraySig=get_efficiency_effnum_effden(test_set,names,myvariablesy,1,0.01)
     efficiencyBkg_array,xaxisBkg,num_arrayBkg,den_arrayBkg=get_efficiency_effnum_effden(test_set,names,myvariablesy,0,0.01)
     plot_efficiency(names,efficiencySig_array,xaxisSig,"signal",suffix,plotdir)
     plot_efficiency(names,efficiencyBkg_array,xaxisBkg,"background",suffix,plotdir)
     significance_array= calculatesignificance(efficiencySig_array,sig,efficiencyBkg_array,bkg)
     plot_significance(names,significance_array,xaxisSig,suffix,plotdir)
  

