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

import binascii
import os.path
from io import BytesIO
from pkg_resources import resource_filename

from klein import Klein
from jinja2 import Environment, PackageLoader
from twisted.web.static import File
import matplotlib; matplotlib.use("Agg")  # pylint: disable=multiple-statements,wrong-import-position

from machine_learning_hep.general import get_database_ml_parameters, getdataframe
from machine_learning_hep.general import createstringselection, filterdataframe_singlevar
from machine_learning_hep.general import get_database_ml_gridsearch
from machine_learning_hep.functions import create_mlsamples, do_correlation
from machine_learning_hep.io import parse_yaml, checkdir
from machine_learning_hep.models import getclf_scikit, getclf_xgboost, getclf_keras
from machine_learning_hep.models import fit, savemodels, decisionboundaries
from machine_learning_hep.models import importanceplotall
from machine_learning_hep.mlperformance import cross_validation_mse, cross_validation_mse_continuous
from machine_learning_hep.mlperformance import plot_cross_validation_mse, plot_learning_curves
from machine_learning_hep.mlperformance import precision_recall
from machine_learning_hep.grid_search import do_gridsearch, read_grid_dict, perform_plot_gridsearch
from machine_learning_hep.config import Configuration


APP = Klein()
DATA_PREFIX = os.path.expanduser("~/.machine_learning_hep")

# Initialize the Jinja2 template engine
JENV = Environment(loader=PackageLoader("machine_learning_hep.webapp", "templates"))
JENV.filters["render_image"] = lambda x: \
    (f'<img src="data:image/png;base64, {binascii.b2a_base64(x.read()).decode("utf-8")}"' +
     'alt="ML plot" height="500">') if x else "<!-- no such image -->"


def jfilter_tab_header(img, title, label, active=False):  # pylint: disable=unused-argument
    disabled = "" if img else "disabled"
    active = "active" if active else ""
    return f"""
        <li class="nav-item">
            <a class="nav-link {disabled} {active}" id="{label}-tab" data-toggle="tab" href="#{label}" role="tab" aria-controls="{label}" aria-selected="false">{title}</a>
        </li>"""


def jfilter_tab_content(img, title, label, active=False):
    image = '<img src="data:image/png;base64, ' + \
            f'{binascii.b2a_base64(img.read()).decode("utf-8")}"' + \
            ' alt="ML plot" height="1000">' if img else "<!-- no such image -->"
    active = "show active" if active else ""
    return f"""
        <div class="tab-pane fade {active}" id="{label}" role="tabpanel" aria-labelledby="{label}-tab">
            <div style="text-align: center;">{image}</div>
            <div style="text-align: center;">{title}</div>
        </div>"""

def jdisplaytext(txt):
    return f"{txt}"

JENV.globals["tab_header"] = jfilter_tab_header
JENV.globals["tab_content"] = jfilter_tab_content
JENV.globals["displaytext"] = jdisplaytext


@APP.route("/static/", branch=True)
def static(req):  # pylint: disable=unused-argument
    """Serve all static files. Works for pip-installed packages too. See:
       https://klein.readthedocs.io/en/latest/introduction/1-gettingstarted.html#static-files"""
    staticPrefix = resource_filename(__name__, "static")
    return File(staticPrefix)


@APP.route("/")
def root(req):  # pylint: disable=unused-argument
    """Serve the home page."""
    return JENV.get_template("display0.html").render()


def get_form(req, label, var_type=str, get_list=False):  # pylint: disable=unused-argument
    """Get elements from a form in an intuitive way. `label` is a string. If `type` is not specified
       the value of the first element from the form list is returned (use `type=list` to return the
       whole list). If `type` is `bool` then some smart comparison on strings meaning `True` is
       performed."""
    if isinstance(label, str):
        label = label.encode()  # to bytes
    val = []
    for i in req.args.get(label, [b"off"]) if var_type == bool else req.args[label]:
        i = i.decode("utf-8")  # to string
        if var_type == bool:
            i = i.lower() in ["on", "true", "yes", "1"]
        val.append(i)
    return val if get_list else val[0]


@APP.route('/formContinue', methods=["POST"])
def post_continue(req):  # pylint: disable=unused-argument
    """Serve the configuration page."""
    subtype = get_form(req, "slct1")
    case = get_form(req, "slct2")
    data = get_database_ml_parameters()
    filesig, filebkg = data[case]["sig_bkg_files"]
    filesig = os.path.join(DATA_PREFIX, filesig)
    filebkg = os.path.join(DATA_PREFIX, filebkg)
    trename = data[case]["tree_name"]
    var_all = data[case]["var_all"]
    var_all_str = ','.join(var_all)
    var_signal = data[case]["var_signal"]
    sel_signal = data[case]["sel_signal"]
    sel_bkg = data[case]["sel_bkg"]
    sel_bkg_str = ''
    for i in sel_bkg:
        if i == '<':
            sel_bkg_str += '&lt;'
        elif i == '>':
            sel_bkg_str += '&gt;'
        elif i == ' ':
            sel_bkg_str += ','
        else:
            sel_bkg_str += i
    var_training = data[case]["var_training"]
    var_training_str = ','.join(var_training)
    var_corr_x, var_corr_y = data[case]["var_correlation"]
    var_corr_x_str = ','.join(var_corr_x)
    var_corr_y_str = ','.join(var_corr_y)
#    var_binning = [data[case]["var_binning"]]
#    var_binning_str = ','.join(var_binning)
#    varmin = ['0']
#    var_binning_min_str = ','.join(varmin)
#    varmax = ['100']
#    var_binning_max_str = ','.join(varmax)
    var_binning = data[case]["var_binning"]
    var_binning_min = 2
    var_binning_max = 3
    presel_reco = data[case]["presel_reco"]
    presel_reco_str = None
    if presel_reco is not None:
        presel_reco_str = ''
        for i in presel_reco:
            if i == '<':
                presel_reco_str += '&lt;'
            elif i == '>':
                presel_reco_str += '&gt;'
            elif i == ' ':
                presel_reco_str += ','
            else:
                presel_reco_str += i

    return JENV.get_template("test.html").render(
        subtype=subtype, case=case,
        filesig=filesig, filebkg=filebkg,
        trename=trename, var_all_str=var_all_str,
        var_signal=var_signal,
        sel_signal=sel_signal,
        sel_bkg_str=sel_bkg_str,
        var_training_str=var_training_str,
        var_corr_x_str=var_corr_x_str,
        var_corr_y_str=var_corr_y_str,
        var_binning=var_binning,
        var_binning_min=var_binning_min,
        var_binning_max=var_binning_max, presel_reco_str=presel_reco_str)

@APP.route('/formContinue/formSubmit', methods=["POST"])
def post_form(req):  # pylint: disable=too-many-locals, too-many-statements, too-many-branches

    # Collect configuration in a dictionary for further processing
    run_config = {}

    mltype = "BinaryClassification"
    run_config["mltype"] = mltype
    case = get_form(req, "case")
    run_config["case"] = case
    filesig = get_form(req, "filesig")
    filebkg = get_form(req, "filebkg")
    trename = get_form(req, "tree_name")
    var_all_str = get_form(req, "var_all")
    var_all = var_all_str.split(',')
    var_signal = get_form(req, "var_signal")
    sel_signal = get_form(req, "sel_signal")
    sel_bkg_str = get_form(req, "sel_bkg")
    sel_bkg = ''
    for i in sel_bkg_str:
        if i == ',':
            sel_bkg += ' '
        elif i == '&lt;':
            sel_bkg += '<'
        elif i == '&gt;':
            sel_bkg += '>'
        else:
            sel_bkg += i

    var_training_str = get_form(req, "var_training")
    var_training = var_training_str.split(',')
    var_corr_x_str = get_form(req, "var_correlation_x")
    var_corr_y_str = get_form(req, "var_correlation_y")
    var_corr_x = var_corr_x_str.split(',')
    var_corr_y = var_corr_y_str.split(',')
#    var_binning_str = get_form(req, "var_binning")
#    var_binning = var_binning_str.split(',')
#    var_binning_min_str = get_form(req, "var_binning_min_str")
#    varmin = [int(i) for i in var_binning_min_str.split(',')]
#    var_binning_max_str = get_form(req, "var_binning_max_str")
#    varmax = [int(i) for i in var_binning_max_str.split(',')]
    var_binning = get_form(req, "var_binning")
    var_binning_min = float(get_form(req, 'var_binning_min', var_type=float))
    var_binning_max = float(get_form(req, 'var_binning_max', var_type=float))
    run_config["binmin"] = var_binning_min
    run_config["binmax"] = var_binning_max
    presel_reco_str = get_form(req, "presel_reco")

    presel_reco = ''
    if presel_reco_str == 'None':
        presel_reco = None
    else:
        for i in presel_reco_str:
            if i == ',':
                presel_reco += ' '
            elif i == '&lt;':
                presel_reco += '<'
            elif i == '&gt;':
                presel_reco += '>'
            else:
                presel_reco += i

    activate_scikit = get_form(req, 'activate_scikit', var_type=bool)
    activate_xgboost = get_form(req, 'activate_xgboost', var_type=bool)
    activate_keras = get_form(req, 'activate_keras', var_type=bool)

    docorrelation = get_form(req, 'docorrelation', var_type=bool)
    run_config["docorrelation"] = docorrelation
    dotraining = get_form(req, 'dotraining', var_type=bool)
    run_config["dotraining"] = dotraining
    doROC = get_form(req, 'doROC', var_type=bool)
    run_config["doROC"] = doROC
    dolearningcurve = get_form(req, 'dolearningcurve', var_type=bool)
    run_config["dolearningcurve"] = dolearningcurve
    docrossvalidation = get_form(req, 'docrossvalidation', var_type=bool)
    run_config["docrossvalidation"] = docrossvalidation
    doimportance = get_form(req, 'doimportance', var_type=bool)
    run_config["doimportance"] = doimportance
    dogridsearch = get_form(req, 'dogridsearch', var_type=bool)
    run_config["dogridsearch"] = dogridsearch

    rnd_shuffle = int(get_form(req, 'rnd_shuffle', var_type=int))
    run_config["rnd_shuffle"] = rnd_shuffle
    nevt_sig = int(get_form(req, 'nevt_sig', var_type=int))
    run_config["nevt_sig"] = nevt_sig
    nevt_bkg = int(get_form(req, 'nevt_bkg', var_type=int))
    run_config["nevt_bkg"] = nevt_bkg
    test_frac = float(get_form(req, 'test_frac', var_type=float))
    run_config["test_frac"] = test_frac
    rnd_splt = int(get_form(req, 'rnd_splt', var_type=int))
    run_config["rnd_splt"] = rnd_splt
    nkfolds = int(get_form(req, 'nkfolds', var_type=int))
    run_config["nkfolds"] = nkfolds
    ncores = int(get_form(req, 'ncores', var_type=int))
    run_config["ncores"] = ncores

    data = get_database_ml_parameters()

    # Construct Configuration object from run_config
    conf = Configuration(run_config_input=run_config)
    conf.configure()

    model_config = conf.get_model_config()

    string_selection = createstringselection(var_binning, var_binning_min, var_binning_max)
    suffix = f"nevt_sig{nevt_sig}_nevt_bkg{nevt_bkg}_" \
             f"{mltype}{case}_{string_selection}"

    dataframe = f"dataframes_{suffix}"
    plotdir = f"plots_{suffix}"
    output = f"output_{suffix}"
    checkdir(dataframe)
    checkdir(plotdir)
    checkdir(output)

    classifiers = []
    classifiers_scikit = []
    classifiers_xgboost = []
    classifiers_keras = []

    names = []
    names_scikit = []
    names_xgboost = []
    names_keras = []

    trainedmodels = []

    df_sig = getdataframe(filesig, trename, var_all)
    df_bkg = getdataframe(filebkg, trename, var_all)
    if presel_reco is not None:
        df_sig = df_sig.query(presel_reco)
        df_bkg = df_bkg.query(presel_reco)
    df_sig = filterdataframe_singlevar(df_sig, var_binning, var_binning_min, var_binning_max)
    df_bkg = filterdataframe_singlevar(df_bkg, var_binning, var_binning_min, var_binning_max)

    # Output images
    imageIO_vardist: BytesIO = None
    imageIO_scatterplot: BytesIO = None
    imageIO_corr_sig: BytesIO = None
    imageIO_corr_bkg: BytesIO = None
    imageIO_precision_recall: BytesIO = None
    imageIO_ROC: BytesIO = None
    imageIO_plot_learning_curves: BytesIO = None
    img_scoresRME: BytesIO = None
    img_import: BytesIO = None
    img_gridsearch: BytesIO = None

    # pylint: disable=unused-variable
    _, _, df_sig_train, df_bkg_train, _, _, x_train, y_train, x_test, y_test = \
        create_mlsamples(df_sig, df_bkg, sel_signal, data[case], sel_bkg, rnd_shuffle,
                         var_signal, var_training, nevt_sig, nevt_bkg, test_frac, rnd_splt)
    if docorrelation:
        imageIO_vardist, imageIO_scatterplot, imageIO_corr_sig, imageIO_corr_bkg = \
            do_correlation(df_sig_train, df_bkg_train, var_all, var_corr_x, var_corr_y, plotdir)


    # Using the activate_* flags is for now a work-around
    if activate_scikit:
        classifiers_scikit, names_scikit = getclf_scikit(model_config)
        classifiers = classifiers+classifiers_scikit
        names = names+names_scikit

    if activate_xgboost:
        classifiers_xgboost, names_xgboost = getclf_xgboost(model_config)
        classifiers = classifiers+classifiers_xgboost
        names = names+names_xgboost

    if activate_keras:
        classifiers_keras, names_keras = getclf_keras(model_config, len(x_train.columns))
        classifiers = classifiers+classifiers_keras
        names = names+names_keras

    if dotraining:
        trainedmodels = fit(names, classifiers, x_train, y_train)
        savemodels(names, trainedmodels, output, suffix)

    if doROC:
        imageIO_precision_recall, imageIO_ROC = \
            precision_recall(names, classifiers, suffix, x_train, y_train, nkfolds, plotdir)

    if docrossvalidation:
        df_scores = []
        if mltype == "Regression":
            df_scores = cross_validation_mse_continuous(
                names, classifiers, x_train, y_train, nkfolds, ncores)
        if mltype == "BinaryClassification":
            df_scores = cross_validation_mse(names, classifiers, x_train, y_train,
                                             nkfolds, ncores)
        img_scoresRME = plot_cross_validation_mse(names, df_scores, suffix, plotdir)

    if doimportance:
        img_import = importanceplotall(var_training, names_scikit+names_xgboost,
                                       classifiers_scikit+classifiers_xgboost, suffix, plotdir)

    if dolearningcurve:
        npoints = 10
        imageIO_plot_learning_curves = plot_learning_curves(
            names, classifiers, suffix, plotdir, x_train, y_train, npoints)

    if dogridsearch:
        datasearch = get_database_ml_gridsearch()
        analysisdb = datasearch[mltype]
        names_cv, clf_cv, par_grid_cv, refit_cv, var_param, \
            par_grid_cv_keys = read_grid_dict(analysisdb)
        _, _, dfscore = do_gridsearch(
            names_cv, clf_cv, par_grid_cv, refit_cv, x_train, y_train, nkfolds,
            ncores)
        img_gridsearch = perform_plot_gridsearch(
            names_cv, dfscore, par_grid_cv, par_grid_cv_keys, var_param, plotdir, suffix, 0.1)

    return JENV.get_template("display.html").render(
        imageIO_vardist=imageIO_vardist,
        imageIO_scatterplot=imageIO_scatterplot,
        imageIO_corr_sig=imageIO_corr_sig,
        imageIO_corr_bkg=imageIO_corr_bkg,
        imageIO_precision_recall=imageIO_precision_recall,
        imageIO_ROC=imageIO_ROC,
        imageIO_plot_learning_curves=imageIO_plot_learning_curves,
        img_scoresRME=img_scoresRME,
        img_import=img_import,
        img_gridsearch=img_gridsearch)


def main():
    APP.run(host="127.0.0.1", port=8080)
