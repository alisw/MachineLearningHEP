rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190922_ROOT6-1/pp_2016_mc_prodLcpK0s/243_20190922-2234/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190922_ROOT6-1/pp_2017_mc_prodLcpK0s/243_20190922-2234/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190922_ROOT6-1/pp_2018_mc_prodLcpK0s/243_20190922-2234/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190922_ROOT6-1/pp_mc_prodLcpK0s/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190923_ROOT6-1/pp_2016_data/244_20190923-1806/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190923_ROOT6-1/pp_2017_data/245_20190923-1806/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190923_ROOT6-1/pp_2018_data/246_20190923-1806/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190923_ROOT6-1/pp_data/resultsMBjetvspt

DISPLAY="" python do_entire_analysis.py -r submission/default_complete.yaml -d data/database_ml_parameters_LcpK0spp_test_0923.yml

