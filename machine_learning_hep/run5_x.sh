rm -rf  /data/X3872ALICE3/newvariables/mc/pkldata_scenario3
rm -rf  /data/X3872ALICE3/newvariables/data/pkldata_scenario3
rm -rf  /data/X3872ALICE3/newvariables/mc/pklskdata_scenario3
rm -rf  /data/X3872ALICE3/newvariables/mc/evttotdata_scenario3
rm -rf  /data/X3872ALICE3/newvariables/data/pklskdata_scenario3
rm -rf  /data/X3872ALICE3/newvariables/data/evttotdata_scenario3
rm -rf  /data/X3872ALICE3/newvariables/mc/pklskmldata_scenario3
rm -rf  /data/X3872ALICE3/newvariables/data/pklskmldata_scenario3
rm -rf  /data/X3872ALICE3/newvariables/mc/mltotdata_scenario3
rm -rf  /data/X3872ALICE3/newvariables/data/mltotdata_scenario3
rm -rf  mlout/ 
rm -rf  mlplot/

DISPLAY="" python do_entire_analysis.py -r default_complete.yml -d data/data_run5/database_ml_parameters_Xanalysis_scenario3.yml -a scenario3


