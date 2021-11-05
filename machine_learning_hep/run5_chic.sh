#!/bin/bash

reset

REMOVE="NO"
if [ $REMOVE = "YES" ]; then
	rm -rf  /data/chicALICE3/signal_files/pkldata_scenario3
	rm -rf  /data/chicALICE3/background_files/pkldata_scenario3
	rm -rf  /data/chicALICE3/signal_files/pklskdata_scenario3
	rm -rf  /data/chicALICE3/signal_files/evttotdata_scenario3
	rm -rf  /data/chicALICE3/background_files/pklskdata_scenario3
	rm -rf  /data/chicALICE3/background_files/evttotdata_scenario3
	rm -rf  /data/chicALICE3/signal_files/pklskmldata_scenario3
	rm -rf  /data/chicALICE3/background_files/pklskmldata_scenario3
	rm -rf  /data/chicALICE3/signal_files/mltotdata_scenario3
	rm -rf  /data/chicALICE3/background_files/mltotdata_scenario3
fi
#rm -rf  mlout/ 
#rm -rf  mlplot/

DISPLAY="" python do_entire_analysis.py -r submission/default_complete.yml -d data/data_run5/database_ml_parameters_chicanalysis_scenario3.yml -a scenario3


