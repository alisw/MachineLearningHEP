rm -rf  /data/Run5data_mlhep/prod_test/mltotmc
rm -rf  /data/Run5data_mlhep/prod_test/mltotdata
rm -rf  /data/Run5data_mlhep/prod_test/pklskmc
rm -rf  /data/Run5data_mlhep/prod_test/evttotmc
rm -rf  /data/Run5data_mlhep/prod_test/pklskdata
rm -rf  /data/Run5data_mlhep/prod_test/evttotdata
rm -rf  /data/Run5data_mlhep/prod_test/pklskmlmc
rm -rf  /data/Run5data_mlhep/prod_test/pklskmldata
rm -rf  /data/Run5data_mlhep/prod_test/pklmc
rm -rf  /data/Run5data_mlhep/prod_test/pkldata
rm -rf  /data/Run5data_mlhep/prod_test/resultsmc
rm -rf  /data/Run5data_mlhep/prod_test/resultsmctot
rm -rf  /data/Run5data_mlhep/prod_test/resultsdata
rm -rf  /data/Run5data_mlhep/prod_test/resultsdatatot
rm -rf  /data/Run5data_mlhep/prod_test/mlout
rm -rf  /data/Run5data_mlhep/prod_test/mlplot

DISPLAY="" python do_entire_analysis.py -r submission/default_complete.yml -d data/data_run5/database_ml_parameters_Dzero.yml -a scenario2
