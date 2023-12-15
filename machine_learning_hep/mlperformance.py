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
"""
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import mean_squared_error

from machine_learning_hep.utilities_plot import prepare_fig

HIST_COLORS = ['r', 'b', 'g']

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
    figure, nrows, ncols = prepare_fig(len(names_))
    for ind, name in enumerate(names_, start = 1):
        ax = plt.subplot(nrows, ncols, ind)
        ax.set_xlim([0, (df_scores_[name].mean()*2)])
        plt.hist(df_scores_[name].values, color="blue")
        mystring = f"$\\mu={df_scores_[name].mean():8.2f}, \\sigma={df_scores_[name].std():8.2f}$"
        plt.text(0.2, 4., mystring, fontsize=16)
        plt.title(name, fontsize=16)
        plt.xlabel("scores RMSE", fontsize=16)
        plt.ylim(0, 5)
        plt.ylabel("Entries", fontsize=16)
    figure.savefig(f"{folder}/scoresRME{suffix_}.png", bbox_inches='tight')
    plt.close(figure)


def plotdistributiontarget(names_, testset, myvariablesy, suffix_, folder):
    figure, nrows, ncols = prepare_fig(len(names_))
    for ind, name in enumerate(names_, start = 1):
        plt.subplot(nrows, ncols, ind)
        plt.hist(testset[myvariablesy].values, color="blue", bins=100, label="true value")
        plt.hist(
            testset['y_test_prediction'+name].values,
            color="red", bins=100, label="predicted value")
        plt.title(name, fontsize=16)
        plt.xlabel(myvariablesy, fontsize=16)
        plt.ylabel("Entries", fontsize=16)
    plt.legend(loc="center right")
    figure.savefig(f"{folder}/distributionregression{suffix_}.png", bbox_inches='tight')
    plt.close(figure)


def plotscattertarget(names_, testset, myvariablesy, suffix_, folder):
    figure, nrows, ncols = prepare_fig(len(names_))
    for ind, name in enumerate(names_, start = 1):
        plt.subplot(nrows, ncols, ind)
        plt.scatter(
            testset[myvariablesy].values,
            testset['y_test_prediction'+name].values, color="blue")
        plt.title(name, fontsize=16)
        plt.xlabel(myvariablesy + "true", fontsize=20)
        plt.ylabel(myvariablesy + "predicted", fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
    figure.savefig(f"{folder}/scatterplotregression{suffix_}.png", bbox_inches='tight')
    plt.close(figure)


def confusion(names_, classifiers_, suffix_, x_train, y_train, cvgen, folder, do_diag0):
    figure, nrows, ncols = prepare_fig(len(names_))
    if len(names_) > 1:
        figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.2)
    for ind, (name, clf) in enumerate(zip(names_, classifiers_), start = 1):
        ax = plt.subplot(nrows, ncols, ind)
        y_train_pred = cross_val_predict(clf, x_train, y_train, cv=cvgen)
        conf_mx = confusion_matrix(y_train, y_train_pred)
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        if do_diag0:
            np.fill_diagonal(norm_conf_mx, 0)
        df_cm = pd.DataFrame(norm_conf_mx, range(2), range(2))
        sn.set(font_scale=1.4)  # for label size
        ax_title = f"{name} tot diag = 0" if do_diag0 else name
        ax.set_title(ax_title)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.xaxis.set_ticklabels(['signal', 'background'])
        ax.yaxis.set_ticklabels(['signal', 'background'])
    suffix_0 = "_Diag0" if do_diag0 else ""
    figure.savefig(f"{folder}/confusion_matrix{suffix_}{suffix_0}.png", bbox_inches='tight')
    plt.close(figure)


def plot_precision_recall(names_, classifiers_, suffix_, x_train, y_train,
                          nkfolds, folder, class_labels):

    def do_plot_precision_recall(y_truth, y_score, label, color):
        precisions, recalls, thresholds = precision_recall_curve(y_truth, y_score)
        plt.plot(thresholds, precisions[:-1], f"{color}--",
                 label=f"Precision {label} = TP/(TP+FP)", linewidth=5.0)
        plt.plot(thresholds, recalls[:-1], f"{color}-", alpha=0.5,
                 label=f"Recall {label} = TP/(TP+FN)", linewidth=5.0)

    figure, nrows, ncols = prepare_fig(len(names_))
    for ind, (name, clf) in enumerate(zip(names_, classifiers_), start = 1):
        ax = plt.subplot(nrows, ncols, ind)
        y_score = cross_val_predict(clf, x_train, y_train, cv=nkfolds, method="predict_proba")
        if len(class_labels) == 2:
            do_plot_precision_recall(y_train, y_score[:, 1], "signal", HIST_COLORS[0])
        else:
            for cls_hyp, (label_hyp, color) in enumerate(zip(class_labels, HIST_COLORS)):
                do_plot_precision_recall(y_train.iloc[:, cls_hyp], y_score[:, cls_hyp],
                                         label_hyp, color)
            do_plot_precision_recall(y_train.ravel(), y_score.ravel(), "average", "black")

        ax.set_xlabel("Probability", fontsize=20)
        ax.set_ylabel("Precision or Recall", fontsize=20)
        ax.set_title(f"Precision, Recall {name}", fontsize=20)
        ax.legend(loc="best", prop={'size': 30})
        ax.set_ylim([0, 1])
        ax.tick_params(labelsize=20)
    figure.savefig(f"{folder}/precision_recall{suffix_}.png", bbox_inches='tight')
    plt.close(figure)


def plot_roc_ovr(names_, classifiers_, suffix_, x_train, y_train, nkfolds, folder,
                 class_labels, save=True):
    def plot_roc(y_truth, y_score, name, label, color):
        fpr, tpr, _ = roc_curve(y_truth, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, f"{color}-", label=f"ROC {name} {label} vs rest, "\
                 f"AUC = {roc_auc:.2f}", linewidth=5.0)

    figure, nrows, ncols = prepare_fig(len(names_))
    for ind, (name, clf) in enumerate(zip(names_, classifiers_), start = 1):
        ax = plt.subplot(nrows, ncols, ind)
        y_score = cross_val_predict(clf, x_train, y_train, cv=nkfolds, method="predict_proba")
        if len(class_labels) == 2:
            plot_roc(y_train, y_score[:, 1], name, "signal", HIST_COLORS[0])
        else:
            for cls_hyp, (label_hyp, color) in enumerate(zip(class_labels, HIST_COLORS)):
                plot_roc(y_train.iloc[:, cls_hyp], y_score[:, cls_hyp], name, label_hyp, color)
        ax.set_xlabel("False Positive Rate", fontsize=30)
        ax.set_ylabel("True Positive Rate", fontsize=30)
        ax.set_title(f"ROC one vs. rest {name}", fontsize=30)
        ax.legend(loc='lower right', prop={'size': 25})
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=20)
    if save:
        figure.savefig(f"{folder}/ROC_OvR_{suffix_}.png", bbox_inches='tight')
        #plt.close(figure)
    return figure


def plot_roc_ovo(names_, classifiers_, suffix_, x_train, y_train, nkfolds, folder,
                 class_labels, save=True):
    if len(class_labels) <= 2:
        raise ValueError("ROC OvO cannot be computed for binary classification")
    figure, nrows, ncols = prepare_fig(len(names_))
    for ind, (name, clf) in enumerate(zip(names_, classifiers_), start = 1):
        ax = plt.subplot(nrows, ncols, ind)
        y_score = cross_val_predict(clf, x_train, y_train, cv=nkfolds, method="predict_proba")
        label_pairs = itertools.combinations(class_labels, 2)
        for label_pair, color in zip(label_pairs, HIST_COLORS):
            ind_lab1 = class_labels.index(label_pair[0])
            ind_lab2 = class_labels.index(label_pair[1])
            mask_or = np.logical_or(y_train.iloc[:, ind_lab1], y_train.iloc[:, ind_lab2])
            for ind, (ind_lab, alpha) in enumerate(zip((ind_lab1, ind_lab2), (1.0, 0.5))):
                mask = y_train == ind_lab
                fpr, tpr, _ = roc_curve(mask[mask_or], y_score[mask_or, ind_lab])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, f"{color}-", alpha=alpha, label=f"ROC "\
                         f"{label_pair[ind]} vs {label_pair[1-ind]} (AUC = {roc_auc:.2f})",
                         linewidth=5.0)
        global_roc_auc = roc_auc_score(y_train, y_score, average="macro", multi_class='ovo')
        plt.plot([], [], ' ', label=f'Unweighted average OvO ROC AUC: {global_roc_auc:.2f}')
        ax.set_xlabel("First class efficiency", fontsize=20)
        ax.set_ylabel("Second class efficiency", fontsize=20)
        ax.set_title(f"ROC one vs. one {name}", fontsize=30)
        ax.legend(loc='lower right', prop={'size': 25})
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=20)
    if save:
        figure.savefig(f"{folder}/ROC_OvO_{suffix_}.png", bbox_inches='tight')
        #plt.close(figure)
    return figure


def roc_train_test(names_, classifiers_, suffix_, x_train, y_train, x_test, y_test,
                   nkfolds, folder, class_labels, binlims, roc_type):
    binmin, binmax = binlims
    if roc_type not in ("roc_ovr", "roc_ovo"):
        raise ValueError("ROC type can be only roc_ovr or roc_ovo")
    roc_fun = plot_roc_ovr if roc_type == "roc_ovr" else plot_roc_ovo
    fig_train = roc_fun(names_, classifiers_, suffix_, x_train, y_train, nkfolds,
                        folder, class_labels, save=False)
    fig_test = roc_fun(names_, classifiers_, suffix_, x_test, y_test, nkfolds,
                       folder, class_labels, save=False)

    figure, nrows, ncols = prepare_fig(len(names_))
    for ind, (ax_train, ax_test) in enumerate(zip(fig_train.get_axes(), fig_test.get_axes()),
                                              start = 1):
        ax = plt.subplot(nrows, ncols, ind)
        for roc_train, roc_test in zip(ax_train.lines, ax_test.lines):
            for roc_t, set_name, alpha, ls in zip((roc_train, roc_test), ("train", "test"),
                                                  (0.4, 0.8), ("-", "-.")):
                plt.plot(roc_t.get_xdata(), roc_t.get_ydata(), lw=roc_t.get_lw(), c=roc_t.get_c(),
                         alpha=alpha, marker=roc_t.get_marker(), linestyle=ls,
                         label=f"{roc_t.get_label()}, {set_name} set")
        ax.set_xlabel("False Positive Rate", fontsize=30)
        ax.set_ylabel("True Positive Rate", fontsize=30)
        ax.legend(loc='lower right', prop={'size': 25})
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=20)

        ax.text(0.7, 0.5,
                 f" ${binmin} < p_\\mathrm{{T}}/(\\mathrm{{GeV}}/c) < {binmax}$",
                 verticalalignment="center", transform=ax.transAxes, fontsize=30)

    figure.savefig(f"{folder}/ROCtraintest_OvR_{suffix_}.png", bbox_inches='tight')
    plt.close(figure)


def plot_learning_curves(names_, classifiers_, suffix_, folder, x_data, y_data, npoints):
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
    high = len(x_train)
    low = 100
    step_ = int((high-low)/npoints)
    figure, nrows, ncols = prepare_fig(len(names_))
    for ind, (name, clf) in enumerate(zip(names_, classifiers_), start = 1):
        ax = plt.subplot(nrows, ncols, ind)
        train_errors, val_errors = [], []
        arrayvalues = np.arange(start=low, stop=high, step=step_)
        for m in arrayvalues:
            clf.fit(x_train[:m], y_train[:m])
            y_train_predict = clf.predict(x_train[:m])
            y_val_predict = clf.predict(x_val)
            train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict, y_val))
        plt.plot(arrayvalues, np.sqrt(train_errors), "r-+", linewidth=5, label="training")
        plt.plot(arrayvalues, np.sqrt(val_errors), "b-", linewidth=5, label="testing")
        ax.set_xlabel("Training set size", fontsize=30)
        ax.set_ylabel("MSE", fontsize=30)
        ax.set_title(f"Learning curve {name}", fontsize=30)
        ax.legend(loc="best", prop={'size': 30})
        ax.set_ylim([0, np.amax(np.sqrt(val_errors))*2])
        ax.tick_params(labelsize=20)
    figure.savefig(f"{folder}/learning_curve{suffix_}.png", bbox_inches='tight')
    plt.close(figure)


def plot_model_pred(names, classifiers, suffix, x_train, y_train, x_test, y_test, folder,
                    class_labels, bins=50):
    def truth_condition(y_t, cls):
        if len(class_labels) == 2:
            return ~y_t if cls == class_labels.index("bkg") else y_t
        else:
            return y_t.iloc[:, cls]

    for name, clf in zip(names, classifiers):
        predict_probs_train = clf.predict_proba(x_train)
        predict_probs_test = clf.predict_proba(x_test)
        for cls_hyp, label_hyp in enumerate(class_labels):
            figure = plt.figure(figsize=(10, 8))
            for cls, (label, color) in enumerate(zip(class_labels, HIST_COLORS)):
                truth_train = truth_condition(y_train, cls)
                truth_test = truth_condition(y_test, cls)
                # d1 = clf.predict_proba(x[y > 0.5])[:, 1] # signal
                # d2 = clf.predict_proba(x[y < 0.5])[:, 1] # background
                plt.hist(predict_probs_train[truth_train, cls_hyp],
                         color=color, alpha=0.5, range=[0, 1], bins=bins,
                         histtype='stepfilled', density=True, label=f'{label}, train')
                predicted_probs = predict_probs_test[truth_test, cls_hyp]
                hist, bins = np.histogram(predicted_probs, bins=bins, range=[0, 1], density=True)
                scale = len(predicted_probs) / sum(hist)
                err = np.sqrt(hist * scale) / scale
                center = (bins[:-1] + bins[1:]) / 2
                plt.errorbar(center, hist, yerr=err, fmt='o', c=color, label=f'{label}, test')
            plt.xlabel(f"ML score for {label_hyp}", fontsize=15)
            plt.ylabel("Arbitrary units", fontsize=15)
            plt.legend(loc="best", frameon=False, fontsize=15)
            plt.yscale("log")
            figure.savefig(f"{folder}/ModelOutDistr_{label_hyp}_{name}_{suffix}.png",
                           bbox_inches='tight')
            plt.close(figure)
