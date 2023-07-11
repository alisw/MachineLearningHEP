rm -rf /home/jklein/data/test/lc/pkl
rm -rf  /home/jklein/data/test/lc/pklsk
rm -rf  /home/jklein/data/test/lc/pp_data_evttot
rm -rf  /home/jklein/data/test/lc/pklskml

python3 machine_learning_hep/do_entire_analysis.py -r machine_learning_hep/submission/default_complete.yml -d machine_learning_hep/data/data_run3/database_ml_parameters_Lcpp_jet.yml -a jet_zg $*
