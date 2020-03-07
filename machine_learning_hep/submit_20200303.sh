rm -rf  /data/DerivedVal/dataval_16_test
rm -rf  /data/DerivedVal/dataval_17_test
rm -rf  /data/DerivedVal/dataval_18_test
rm -rf  /data/Derived_testVal/datavaltot_test
rm -rf  /data/DerivedVal/mcval_16_test
rm -rf  /data/DerivedVal/mcval_17_test
rm -rf  /data/DerivedVal/mcval_18_test
rm -rf  /data/Derived_testVal/mcvaltot_test


rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2016_mc_prodD2H/353_20200201-1929/pkl
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2017_mc_prodD2H/353_20200201-1929/pkl
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2018_mc_prodD2H/353_20200201-1929/pkl
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2016_data/350_20200201-1926/pkl
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2017_data/351_20200202-0239/pkl
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2018_data/352_20200202-0239/pkl
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2016_mc_prodD2H/353_20200201-1929/pklsk
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2017_mc_prodD2H/353_20200201-1929/pklsk
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2018_mc_prodD2H/353_20200201-1929/pklsk
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_mc_prodD2H_evttot
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2016_data/350_20200201-1926/pklsk
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2017_data/351_20200202-0239/pklsk
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2018_data/352_20200202-0239/pklsk
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_data_evttot
rm -rf  /data/DerivedResultsJets/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2016_mc_prodD2H/353_20200201-1929/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2017_mc_prodD2H/353_20200201-1929/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2018_mc_prodD2H/353_20200201-1929/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7withJets/vAN-20200201_ROOT6-1/pp_mc_prodD2H/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2016_data/350_20200201-1926/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2017_data/351_20200202-0239/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2018_data/352_20200202-0239/resultsMBjetvspt
rm -rf  /data/DerivedResultsJets/LckINT7withJets/vAN-20200201_ROOT6-1/pp_data/resultsMBjetvspt
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2016_mc_prodD2H/353_20200201-1929/pklskdec
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2017_mc_prodD2H/353_20200201-1929/pklskdec
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2018_mc_prodD2H/353_20200201-1929/pklskdec
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2016_data/350_20200201-1926/pklskdec
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2017_data/351_20200202-0239/pklskdec
rm -rf  /data/Derived/LckINT7withJets/vAN-20200201_ROOT6-1/pp_2018_data/352_20200202-0239/pklskdec


DISPLAY="" python do_entire_analysis.py -r submission/default_complete.yml -d data/FFLc/database_ml_parameters_LcpK0spp_20200301.yml

