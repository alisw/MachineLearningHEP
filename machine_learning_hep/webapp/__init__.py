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
from machine_learning_hep.functions import vardistplot # pylint: disable=wrong-import-position


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
    slct1 = request.form['slct1']
    slct2 = request.form['slct2']
    responseString = '{0},{1}'.format(slct1, slct2)
    data = [random.randint(1, 100) for _ in range(100)]
    file = uproot.open("/Users/gianmicheleinnocenti/Desktop/MLPackage/MachineLearningHEP/"
                       "machine_learning_hep/data/inputroot/"
                       "AnalysisResults_Lambdac_Data_CandBased_skimmed.root")
    tree = file["fTreeLcFlagged"]
    mylistvariables = ["inv_mass_ML", "pt_cand_ML", "d_len_ML", "d_len_xy_ML"]
    df = tree.pandas.df(mylistvariables)
    data = df.inv_mass_ML
    plt.figure(figsize=(15, 15))
    plt.hist(data)
#     imageIO_vardist, imageIO_scatterplot, imageIO_scatterplot, imageIO_scatterplot \
    imageIO_vardist = vardistplot(df, df, mylistvariables, "./")

    pngData = binascii.b2a_base64(imageIO_vardist.read())
    print(pngData.decode("utf-8"))
    return render_template('display.html', responseString=responseString,
                           plotBase64=pngData.decode("utf-8"))

def main():
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    APP.run(host='127.0.0.1', port=8080, debug=True)
