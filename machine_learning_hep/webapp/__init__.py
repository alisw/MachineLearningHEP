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

import random
import datetime
import sys
import base64
import binascii



from io import BytesIO, StringIO
import uproot
from flask import Flask, render_template, request
from flask import send_file

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position
from machine_learning_hep.functions import do_correlation, create_mlsamples # pylint: disable=wrong-import-position
from machine_learning_hep.general import get_database_ml_parameters, getdataframe # pylint: disable=wrong-import-position



APP = Flask(__name__)

@APP.route('/') # https://test-app-227515.appspot.com
def root():
    # For the sake of example, use static information to inflate the template.
    # This will be replaced with real information in later steps.
    return render_template('test.html')

@APP.route('/formSubmit', methods=['POST']) # https://test-app-227515.appspot.com/
def post_form():
  # request.args {'slct1': 'JetTagging', 'slct2': 'hypertritium'}
#   file = request.form.files[

    mlsubtype = request.form['slct1']
    case = request.form['slct2']
    data = get_database_ml_parameters()
    filesig, filebkg = data[case]["sig_bkg_files"]
    trename = data[case]["tree_name"]
    var_all = data[case]["var_all"]
    var_signal = data[case]["var_signal"]
    sel_signal = data[case]["sel_signal"]
    sel_bkg = data[case]["sel_bkg"]
    var_training = data[case]["var_training"]
    var_corr_x, var_corr_y = data[case]["var_correlation"]

#     filedata, filemc = data[case]["data_mc_files"]
#     var_target = data[case]["var_target"]
#     var_boundaries = data[case]["var_boundaries"]

    responseString = '{0},{1}'.format(mlsubtype, case)

    filesig = "../"+filesig
    filebkg = "../"+filebkg
    var_skimming = ["pt_cand_ML"]
    varmin = [0]
    varmax = [10]
    rnd_shuffle = 12
    nevt_sig = 1000
    nevt_bkg = 1000
    test_frac = 0.2
    rnd_splt = 12
    plotdir = "./"

    df_sig = getdataframe(filesig, trename, var_all)
    df_bkg = getdataframe(filebkg, trename, var_all)
    # pylint: disable=unused-variable
    _, _, df_sig_train, df_bkg_train, _, _, \
    x_train, y_train, x_test, y_test = \
        create_mlsamples(df_sig, df_bkg, sel_signal, sel_bkg, rnd_shuffle,
                         var_skimming, varmin, varmax, var_signal, var_training,
                         nevt_sig, nevt_bkg, test_frac, rnd_splt)
    imageIO_vardist, imageIO_scatterplot, imageIO_corr_sig, imageIO_corr_bkg = \
        do_correlation(df_sig_train, df_bkg_train, var_all, var_corr_x, var_corr_y, plotdir)

    imageIO_vardist = binascii.b2a_base64(imageIO_vardist.read())
    imageIO_scatterplot = binascii.b2a_base64(imageIO_scatterplot.read())
    imageIO_corr_sig = binascii.b2a_base64(imageIO_corr_sig.read())
    imageIO_corr_bkg = binascii.b2a_base64(imageIO_corr_bkg.read())
#     print(pngData.decode("utf-8"))
    return render_template('display.html', responseString=responseString,
                           imageIO_vardist=imageIO_vardist.decode("utf-8"), \
                           imageIO_scatterplot=imageIO_scatterplot.decode("utf-8"), \
                           imageIO_corr_sig=imageIO_corr_sig.decode("utf-8"), \
                           imageIO_corr_bkg=imageIO_corr_bkg.decode("utf-8"))

def main():
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    APP.run(host='127.0.0.1', port=8080, debug=True)
