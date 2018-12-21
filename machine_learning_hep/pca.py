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
Methods to: apply Principal Component Analysis (PCA) and to standardize features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_dataframe_pca(dataframe, n_pca):
    data_values = dataframe.values
    pca = PCA(n_pca)
    principal_comp = pca.fit_transform(data_values)
    pca_name_list = []
    for i_pca in range(n_pca):
        pca_name_list.append("princ_comp_%d" % (i_pca+1))
    pca_dataframe = pd.DataFrame(data=principal_comp, columns=pca_name_list)

    return pca_dataframe, pca, pca_name_list

def apply_pca(dataframe, pca, var_names):
    data_values = dataframe.values
    principal_comp = pca.transform(data_values)
    pca_dataframe = pd.DataFrame(data=principal_comp, columns=var_names)

    return pca_dataframe


def get_dataframe_std(dataframe):
    data_values = dataframe.values
    std_scal = StandardScaler()
    data_values_std = std_scal.fit_transform(data_values)
    std_dataframe = pd.DataFrame(data=data_values_std, columns=list(dataframe.columns.values))

    return std_dataframe, std_scal


def apply_std(dataframe, std_scal):
    data_values = dataframe.values
    data_values_std = std_scal.transform(data_values)
    std_dataframe = pd.DataFrame(data=data_values_std, columns=list(dataframe.columns.values))

    return std_dataframe


def plotvariance_pca(pca_object, output_):
    figure = plt.figure(figsize=(15, 10)) # pylint: disable=unused-variable
    plt.plot(np.cumsum(pca_object.explained_variance_ratio_))
    plt.plot([0, 10], [0.95, 0.95])
    plt.xlabel('number of components', fontsize=16)
    plt.ylabel('cumulative explained variance', fontsize=16)
    plt.title('Explained variance', fontsize=16)
    plt.ylim([0, 1])
    plotname = output_+'/PCAvariance.png'
    plt.savefig(plotname, bbox_inches='tight')
