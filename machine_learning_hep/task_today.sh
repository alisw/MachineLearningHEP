##################################################################################################################################################
#rm -rf /data/Derived_testResults/Jets/Lc/2_6/vAN-20200824_ROOT6-1/*
#rm -rf /data/Derived_testResults/Jets/Lc/6_12/vAN-20200824_ROOT6-1/*
#
#nice python do_entire_analysis.py -a jet_FF -r submission/default_analyzer.yml -d data/data_prod_20200824/database_ml_parameters_LcpK0spp_0824_jet_2_6.yml -c
#nice python do_entire_analysis.py -a jet_FF -r submission/default_analyzer.yml -d data/data_prod_20200824/database_ml_parameters_LcpK0spp_0824_jet_6_12.yml -c
#             
#pkill -9 -u talazare do_entire_analysis
###################################################################################################################################################
#
#rm -rf /data/Derived_testResults/Jets/D0kAnywithJets/2_6/vAN-20200824_ROOT6-1/*
#rm -rf /data/Derived_testResults/Jets/D0kAnywithJets/6_12/vAN-20200824_ROOT6-1/*
#
#nice python do_entire_analysis.py -a jet_r_shape -r submission/default_analyzer.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824_2_6.yml -c
#nice python do_entire_analysis.py -a jet_r_shape -r submission/default_analyzer.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824_6_12.yml -c
            
pkill -9 -u talazare do_entire_analysis
##################################################################################################################################################
#mv /data/Derived_testResults/Jets/Lc/2_6 /data/Derived_testResults/Jets/Lc/2_6_old_eff
#mv /data/Derived_testResults/Jets/Lc/6_12 /data/Derived_testResults/Jets/Lc/6_12_old_eff
##
nice python do_entire_analysis.py -a jet_FF -r submission/default_analyzer_ef.yml -d data/data_prod_20200824/database_ml_parameters_LcpK0spp_0824_jet_2_6_ef.yml -c
nice python do_entire_analysis.py -a jet_FF -r submission/default_analyzer_ef.yml -d data/data_prod_20200824/database_ml_parameters_LcpK0spp_0824_jet_6_12_ef.yml -c
#             
pkill -9 -u talazare do_entire_analysis
##################################################################################################################################################
#
#mv /data/Derived_testResults/Jets/D0kAnyWithJets/2_6 /data/Derived_testResults/Jets/D0kAnyWithJets/2_6_old_eff
#mv /data/Derived_testResults/Jets/D0kAnyWithJets/6_12 /data/Derived_testResults/Jets/D0kAnyWithJets/6_12_old_eff
#
nice python do_entire_analysis.py -a jet_r_shape -r submission/default_analyzer_ef.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824_2_6_ef.yml -c
nice python do_entire_analysis.py -a jet_r_shape -r submission/default_analyzer_ef.yml -d data/data_prod_20200824/database_ml_parameters_D0pp_0824_6_12_ef.yml -c
            
pkill -9 -u talazare do_entire_analysis
