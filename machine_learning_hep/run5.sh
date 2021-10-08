rm -rf  /data/X3872ALICE3/extraone/mc/pkldata_scenario3
rm -rf  /data/X3872ALICE3/extraone/data/pkldata_scenario3
rm -rf  /data/X3872ALICE3/extraone/mc/pklskdata_scenario3
rm -rf  /data/X3872ALICE3/extraone/mc/evttotdata_scenario3
rm -rf  /data/X3872ALICE3/extraone/data/pklskdata_scenario3
rm -rf  /data/X3872ALICE3/extraone/data/evttotdata_scenario3
rm -rf  /data/X3872ALICE3/extraone/mc/pklskmldata_scenario3
rm -rf  /data/X3872ALICE3/extraone/data/pklskmldata_scenario3
rm -rf  /data/X3872ALICE3/extraone/mc/mltotdata_scenario3
rm -rf  /data/X3872ALICE3/extraone/data/mltotdata_scenario3
rm -rf  mlout
rm -rf  mlplot


DISPLAY="" python do_entire_analysis.py -r default_complete.yml -d data/data_run5/database_ml_parameters_Xanalysis_scenario3.yml -a scenario3


