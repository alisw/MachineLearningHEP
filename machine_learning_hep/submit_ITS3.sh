rm -rf  /data/Derived/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/pklmc
rm -rf  /data/Derived/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/pkldata
rm -rf  /data/Derived/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/pklskmc
rm -rf  /data/Derived/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/pp_mc_evttot
rm -rf  /data/Derived/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/pklskdata
rm -rf  /data/Derived/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/pp_data_evttot
rm -rf  /data/Derived/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/pklmcdec
rm -rf  /data/Derived/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/pklskdatadec
rm -rf  /data/DerivedResults/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/resultsMBjetvsptmc
rm -rf  /data/DerivedResults/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/resultsMBjetvsptmctot
rm -rf  /data/DerivedResults/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/resultsMBjetvsptdata
rm -rf  /data/DerivedResults/LctopKpiITS/vAN-20200305_ROOT6-1/ITS2_19h1a2/359_20200306-0925/resultsMBjetvsptdatatot


DISPLAY="" python do_entire_analysis.py -r submission/default_complete.yml -d data/ITS3/database_ml_parameters_LcpKpippITS3.yml

