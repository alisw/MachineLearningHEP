#!/bin/bash

reset

#rm -rf  mlout/ 
#rm -rf  mlplot/

DISPLAY="" python do_entire_analysis.py -r default_complete.yml -d data/data_run5/database_ml_parameters_xiccanalysis_scenario3.yml -a scenario3


