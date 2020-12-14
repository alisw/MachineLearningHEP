#!/bin/bash

reset

Tag="scenario3"
Remove="No"
if [[ -z $Remove ]]; then
    rm -rf /data/Run5data_mlhep/prod_test/pklmc_$Tag
    rm -rf /data/Run5data_mlhep/prod_test/pkldata_$Tag
    rm -rf /data/Run5data_mlhep/prod_test/pklskmc_$Tag
    rm -rf /data/Run5data_mlhep/prod_test/evttotmc_$Tag
    rm -rf /data/Run5data_mlhep/prod_test/pklskdata_$Tag
    rm -rf /data/Run5data_mlhep/prod_test/evttotdata_$Tag
    rm -rf /data/Run5data_mlhep/prod_test/pklskmlmc_$Tag
    rm -rf /data/Run5data_mlhep/prod_test/pklskmldata_$Tag
    rm -rf /data/Run5data_mlhep/prod_test/mltotmc_$Tag
    rm -rf /data/Run5data_mlhep/prod_test/mltotdata_$Tag
fi

DISPLAY="" python do_entire_analysis.py -r submission/default_complete.yml -d data/data_run5/database_ml_parameters_Dzero_$Tag.yml -a $Tag |& tee run5.log
