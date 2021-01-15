
rm -rf  /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1/pp_2016_data/resultsMBjetvspt                                                          
rm -rf  /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1/pp_2017_data/resultsMBjetvspt                                                          
rm -rf  /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1/pp_2018_data/resultsMBjetvspt                                                          
rm -rf  /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1/pp_data/resultsMBjetvspt 

rm -rf  /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1/pp_2016_mc_prodD2H/resultsMBjetvspt                                                    
rm -rf  /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1/pp_2017_mc_prodD2H/resultsMBjetvspt                                                    
rm -rf  /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1/pp_2018_mc_prodD2H/resultsMBjetvspt                                                    
rm -rf  /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1/pp_mc_prodD2H/resultsMBjetvspt  

nice python do_entire_analysis.py -a jet_r_shape -r submission/default_analyzer.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824.yml -c


mkdir /data/Derived_testResults/Jets/D0kAnywithJets/2_6
mkdir /data/Derived_testResults/Jets/D0kAnywithJets/6_12

cp -r /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1 /data/Derived_testResults/Jets/D0kAnywithJets/2_6/.
cp -r /data/Derived_testResults/Jets/D0kAnywithJets/vAN-20200824_ROOT6-1 /data/Derived_testResults/Jets/D0kAnywithJets/6_12/.

nice python do_entire_analysis.py -a jet_r_shape -r submission/default_feeddown.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824_2_6.yml -c
nice python do_entire_analysis.py -a jet_r_shape -r submission/default_feeddown.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824_6_12.yml -c
