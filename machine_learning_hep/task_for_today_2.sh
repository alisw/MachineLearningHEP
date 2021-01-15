mkdir /data/Derived_testResults/Jets/D0kAnywithJets
mkdir /data/Derived_testResults/Jets/D0kAnywithJets/2_6
mkdir /data/Derived_testResults/Jets/D0kAnywithJets/6_12

nice python do_entire_analysis.py -a jet_r_shape -r submission/default_analyzer.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824.yml -c

cp -r /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1 /data/Derived_testResults/Jets/D0kAnywithJets/2_6/.
cp -r /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1 /data/Derived_testResults/Jets/D0kAnywithJets/6_12/.

nice python do_entire_analysis.py -a jet_r_shape -r submission/default_feeddown.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824_2_6.yml -c
nice python do_entire_analysis.py -a jet_r_shape -r submission/default_feeddown.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824_6_12.yml -c
