rm -rf /home/jklein/data/test/pkl
rm -rf  /home/jklein/data/test/pklsk
rm -rf  /home/jklein/data/pp_data_evttot
rm -rf  /home/jklein/data/test/pklskml

python3 machine_learning_hep/do_entire_analysis.py -r machine_learning_hep/submission/default_complete.yml -d machine_learning_hep/data/data_run3/database_ml_parameters_D0pp_jet.yml -a jet_zg $*
