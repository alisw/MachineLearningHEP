rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_2016_data/211_20190909-2119/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_2016_data/212_20190909-2119/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_2017_data/210_20190909-2119/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_2018_data/209_20190909-2118/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_data/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_2016_mc_prodLcpK0s/213_20190909-2123/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_2016_mc_prodLcpK0s/215_20190909-2123/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_2017_mc_prodLcpK0s/218_20190909-2124/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_2018_mc_prodLcpK0s/219_20190909-2125/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7HighMultwithJets/vAN-20190909_ROOT6-1-bkp/pp_mc_prodLcpK0s/resultsMBjetvspt

DISPLAY="" python do_entire_analysis.py -r submission/default_complete.yml -d data/JetAnalysis/database_ml_parameters_LcpK0spp_test_0909.yml

