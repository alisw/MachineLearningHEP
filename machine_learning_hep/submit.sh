#!/bin/bash

#STAGE="pre"
#STAGE="train"
#STAGE="apply"
STAGE="complete"
#STAGE="analyzer"
#STAGE="systematics"

#DBDIR="JetAnalysis"
DBDIR="data_prod_20200304"
#DBDIR="data_prod_20200417"
#DBDIR="pKpi"

DATABASE="D0pp_0304"
#DATABASE="D0pp_0417"
#DATABASE="Dspp"
#DATABASE="LcpK0spp_0304"
#DATABASE="LcpKpi010"
#DATABASE="LcpKpi3050"

#ANALYSIS="MBvspt_perc_v0m"
#ANALYSIS="MBvspt_ntrkl"
#ANALYSIS="SPDvspt_ntrkl"
#ANALYSIS="MBjetvspt"
#ANALYSIS="jet_FF"
ANALYSIS="jet_zg"
#ANALYSIS="jet_rg"
#ANALYSIS="jet_nsd"

python do_entire_analysis.py -r submission/default_${STAGE}.yml -d data/${DBDIR}/database_ml_parameters_${DATABASE}.yml -a ${ANALYSIS}

