rm -rf  /data/Run5data_mlhep/prod_test/pklmc_scenario3
rm -rf  /data/Run5data_mlhep/prod_test/pkldata_scenario3
rm -rf  /data/Run5data_mlhep/prod_test/pklskmc_scenario3
rm -rf  /data/Run5data_mlhep/prod_test/evttotmc_scenario3
rm -rf  /data/Run5data_mlhep/prod_test/pklskdata_scenario3
rm -rf  /data/Run5data_mlhep/prod_test/evttotdata_scenario3
rm -rf  /data/Run5data_mlhep/prod_test/pklskmlmc_scenario3
rm -rf  /data/Run5data_mlhep/prod_test/pklskmldata_scenario3
rm -rf  /data/Run5data_mlhep/prod_test/mltotmc_scenario3
rm -rf  /data/Run5data_mlhep/prod_test/mltotdata_scenario3
rm -rf mlout_scenario3* mlplot_scenario3*
DISPLAY="" python do_entire_analysis.py -r submission/default_complete.yml -d data/data_run5/database_ml_parameters_Dzero_scenario3.yml -a scenario3
