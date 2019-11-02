#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

"""
Methods to: model performance evaluation
TODO Optimisation: In each function model is retrained by calling fit again. 
Avoid this retraining by saving the trained model in optimiser.py and passing
that saved model to the functions defined in this script. Of course if do_train
is disabled, this means no trained models would be available for evaluating...so
we could add an if condition to check for that case and handle it accordingly.

"""
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import cross_val_score, cross_val_predict, \
    train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, \
    mean_squared_error
import itertools # to be removed



def cross_validation_mse(names_, classifiers_, x_train, y_train, cv_, ncores):
    df_scores = pd.DataFrame()
    for name, clf in zip(names_, classifiers_):
        if "Keras" in name:
            ncores = 1
        kfold = StratifiedKFold(n_splits=cv_, shuffle=True, random_state=1)
        scores = cross_val_score(clf, x_train, y_train, cv=kfold,
                                 scoring="neg_mean_squared_error", n_jobs=ncores)
        tree_rmse_scores = np.sqrt(-scores)
        df_scores[name] = tree_rmse_scores
    return df_scores


def cross_validation_mse_continuous(names_, classifiers_, x_train, y_train, cv_, ncores):
    df_scores = pd.DataFrame()
    for name, clf in zip(names_, classifiers_):
        if "Keras" in name:
            ncores = 1
        scores = cross_val_score(clf, x_train, y_train, cv=cv_,
                                 scoring="neg_mean_squared_error", n_jobs=ncores)
        tree_rmse_scores = np.sqrt(-scores)
        df_scores[name] = tree_rmse_scores
    return df_scores


def plot_cross_validation_mse(names_, df_scores_, suffix_, folder):
    figure1 = plt.figure(figsize=(20, 15))
    i = 1
    for name in names_:
        ax = plt.subplot(2, (len(names_)+1)/2, i)
        ax.set_xlim([0, (df_scores_[name].mean()*2)])
        plt.hist(df_scores_[name].values, color="blue")
        mystring = '$\\mu=%8.2f, \\sigma=%8.2f$' % (df_scores_[name].mean(), df_scores_[name].std())
        plt.text(0.2, 4., mystring, fontsize=16)
        plt.title(name, fontsize=16)
        plt.xlabel("scores RMSE", fontsize=16)
        plt.ylim(0, 5)
        plt.ylabel("Entries", fontsize=16)
        figure1.subplots_adjust(hspace=.5)
        i += 1
    plotname = folder+'/scoresRME%s.png' % (suffix_)
    plt.savefig(plotname)
    img_scoresRME = BytesIO()
    plt.savefig(img_scoresRME, format='png')
    img_scoresRME.seek(0)
    return img_scoresRME


def plotdistributiontarget(names_, testset, myvariablesy, suffix_, folder):
    figure1 = plt.figure(figsize=(20, 15))
    i = 1
    for name in names_:
        _ = plt.subplot(2, (len(names_)+1)/2, i)
        plt.hist(testset[myvariablesy].values, color="blue", bins=100, label="true value")
        plt.hist(
            testset['y_test_prediction'+name].values,
            color="red", bins=100, label="predicted value")
        plt.title(name, fontsize=16)
        plt.xlabel(myvariablesy, fontsize=16)
        plt.ylabel("Entries", fontsize=16)
        figure1.subplots_adjust(hspace=.5)
        i += 1
    plt.legend(loc="center right")
    plotname = folder+'/distributionregression%s.png' % (suffix_)
    plt.savefig(plotname)
    img_dist_reg = BytesIO()
    plt.savefig(img_dist_reg, format='png')
    img_dist_reg.seek(0)
    return img_dist_reg


def plotscattertarget(names_, testset, myvariablesy, suffix_, folder):
    _ = plt.figure(figsize=(20, 15))
    i = 1
    for name in names_:
        figure1 = plt.subplot(2, (len(names_)+1)/2, i)
        plt.scatter(
            testset[myvariablesy].values,
            testset['y_test_prediction'+name].values, color="blue")
        plt.title(name, fontsize=16)
        plt.xlabel(myvariablesy + "true", fontsize=16)
        plt.ylabel(myvariablesy + "predicted", fontsize=16)
        figure1.subplots_adjust(hspace=.5)
        i += 1
    plotname = folder+'/scatterplotregression%s.png' % (suffix_)
    plt.savefig(plotname)
    img_scatt_reg = BytesIO()
    plt.savefig(img_scatt_reg, format='png')
    img_scatt_reg.seek(0)
    return img_scatt_reg


def confusion(names_, classifiers_, suffix_, x_train, y_train, x_test, y_test, cvgen, folder):
    figure1 = plt.figure(figsize=(25, 15))  # pylint: disable=unused-variable
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)
    i = 1
    for name, clf in zip(names_, classifiers_):
        ax = plt.subplot(2, (len(names_)+1)/2, i)
        y_train_pred = cross_val_predict(clf, x_train, y_train, cv=cvgen, n_jobs=-1)
        conf_mx = confusion_matrix(y_train, y_train_pred)
        print("Confusion matrix on training data:") # to be removed
        print(conf_mx) # to be removed        
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        df_cm = pd.DataFrame(norm_conf_mx, range(2), range(2))
        sn.set(font_scale=1.4)  # for label size
        ax.set_title(name)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.xaxis.set_ticklabels(['background', 'signal'])
        ax.yaxis.set_ticklabels(['background', 'signal'])
        i += 1
    plotname = folder+'/confusion_matrix%s_train.png' % (suffix_)
    plt.savefig(plotname)
    img_confmatrix_train = BytesIO()
    plt.savefig(img_confmatrix_train, format='png')
    img_confmatrix_train.seek(0)

    figure2 = plt.figure(figsize=(25, 15))  # pylint: disable=unused-variable
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)
    i = 1
    for name, clf in zip(names_, classifiers_):
        ax = plt.subplot(2, (len(names_)+1)/2, i)
        y_test_pred = clf.predict(x_test)
        conf_mx = confusion_matrix(y_test, y_test_pred)
        print("Confusion matrix on test data:") # to be removed
        print(conf_mx) # to be removed
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        df_cm = pd.DataFrame(norm_conf_mx, range(2), range(2))
        sn.set(font_scale=1.4)  # for label size
        ax.set_title(name)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.xaxis.set_ticklabels(['background', 'signal'])
        ax.yaxis.set_ticklabels(['background', 'signal'])
        i += 1
    plotname = folder+'/confusion_matrix%s_test.png' % (suffix_)
    plt.savefig(plotname)
    img_confmatrix_test = BytesIO()
    plt.savefig(img_confmatrix_test, format='png')
    img_confmatrix_test.seek(0)    

    return img_confmatrix_train, img_confmatrix_test 


def precision_recall(names_, classifiers_, suffix_, x_train, y_train, cvgen, folder):
    figure1 = plt.figure(figsize=(25, 15))  # pylint: disable=unused-variable
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)

    i = 1
    for name, clf in zip(names_, classifiers_):
        ax = plt.subplot(2, (len(names_)+1)/2, i)
        y_proba = cross_val_predict(clf, x_train, y_train, cv=cvgen, method="predict_proba")
        y_scores = y_proba[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision=TP/(TP+FP)")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall=TP/(TP+FN)")
        plt.xlabel("probability", fontsize=16)
        ax.set_title(name, fontsize=16)
        plt.legend(loc="center left")
        plt.ylim([0, 1])
        i += 1
    plotname = folder+'/precision_recall%s.png' % (suffix_)
    plt.savefig(plotname)
    img_precision_recall = BytesIO()
    plt.savefig(img_precision_recall, format='png')
    img_precision_recall.seek(0)

    figure2 = plt.figure(figsize=(20, 15))  # pylint: disable=unused-variable
    i = 1
    aucs = []

    for name, clf in zip(names_, classifiers_):
        y_proba = cross_val_predict(clf, x_train, y_train, cv=cvgen, method="predict_proba")
        y_scores = y_proba[:, 1]
        fpr, tpr, _ = roc_curve(y_train, y_scores)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.xlabel('False Positive Rate or (1 - Specifity)', fontsize=20)
        plt.ylabel('True Positive Rate or (Sensitivity)', fontsize=20)
        plt.title('Receiver Operating Characteristic', fontsize=20)
        plt.plot(fpr, tpr, alpha=0.3, label='ROC %s (AUC = %0.2f)' %
                 (names_[i-1], roc_auc), linewidth=4.0)
        plt.legend(loc="lower center", prop={'size': 18})
        i += 1
    plotname = folder+'/ROCcurve%s.png' % (suffix_)
    plt.savefig(plotname)
    img_roc = BytesIO()
    plt.savefig(img_roc, format='png')
    img_roc.seek(0)

    return img_precision_recall, img_roc


def plot_learning_curves(names_, classifiers_, suffix_, folder, x_data, y_data, npoints):

    figure1 = plt.figure(figsize=(20, 15))
    i = 1
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
    for name, clf in zip(names_, classifiers_):
        ax = plt.subplot(2, (len(names_)+1)/2, i)
        train_errors, val_errors = [], []
        high = len(x_train)
        low = 100
        step_ = int((high-low)/npoints)
        arrayvalues = np.arange(start=low, stop=high, step=step_)
        for m in arrayvalues:
            clf.fit(x_train[:m], y_train[:m])
            y_train_predict = clf.predict(x_train[:m])
            y_val_predict = clf.predict(x_val)
            train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict, y_val))
        ax.set_ylim([0, np.amax(np.sqrt(val_errors))*2])
        plt.plot(arrayvalues, np.sqrt(train_errors), "r-+", linewidth=3, label="training")
        plt.plot(arrayvalues, np.sqrt(val_errors), "b-", linewidth=3, label="testing")
        plt.title(name, fontsize=16)
        plt.xlabel("Training set size", fontsize=16)
        plt.ylabel("RMSE", fontsize=16)
        figure1.subplots_adjust(hspace=.5)
        plt.legend(loc="lower center", prop={'size': 18})
        i += 1
    plotname = folder+'/learning_curve%s.png' % (suffix_)
    plt.savefig(plotname)
    img_learn = BytesIO()
    plt.savefig(img_learn, format='png')
    img_learn.seek(0)
    return img_learn

def plot_overtraining(names_, classifiers_, suffix_, folder, \
        x_train, y_train, x_val, y_val, bins=30):
    img_overtrain_array = []
    for name, clf in zip(names_, classifiers_):
        plt.figure()

        decisions = []
        for x, y in ((x_train, y_train), (x_val, y_val)):
            d1 = clf.predict_proba(x[y > 0.5])[:, 1]
            d2 = clf.predict_proba(x[y < 0.5])[:, 1]
            decisions += [d1, d2]

        plt.hist(decisions[0],
                 color='r', alpha=0.5, range=[0, 1], bins=bins,
                 histtype='stepfilled', density=True,
                 label='S (train)')
        plt.hist(decisions[1],
                 color='b', alpha=0.5, range=[0, 1], bins=bins,
                 histtype='stepfilled', density=True,
                 label='B (train)')

        hist, bins = np.histogram(decisions[2],
                                  bins=bins, range=[0, 1], density=True)
        scale = len(decisions[2]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        center = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

        hist, bins = np.histogram(decisions[3],
                                  bins=bins, range=[0, 1], density=True)
        scale = len(decisions[3]) / sum(hist)
        err = np.sqrt(hist * scale) / scale

        plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

        plt.xlabel("output response")
        plt.ylabel("Arbitrary units")
        plt.legend(loc='best')

        plotname = folder+'/overtraining_%s%s.png' % (name, suffix_)
        plt.savefig(plotname)
        img_overtrain = BytesIO()
        plt.savefig(img_overtrain, format='png')
        img_overtrain.seek(0)
        img_overtrain_array += img_overtrain
        plt.clf()
    return img_overtrain_array

# new confusion matrix function (to be deleted later on)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()