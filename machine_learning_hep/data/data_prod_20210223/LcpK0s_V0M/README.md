# Specifics for LcpK0s V0M analysis

The 2 directories contain

* `systematic_DBs/`: databases to estiamte systematic impact on the efficiency. To be run with the flag `--database-overwrite`
* `weights/`: weights for
    * tracklets
    * MC pT shape
    * zvtx

## MC pT shape

The nominal pT shape corresponds to Pythia8 Mode2 (colour reconnection mode). The systematic is estimated by changing the shape according to the FONLL shape found [here](https://twiki.cern.ch/twiki/pub/ALICE/FONLLDist/DmesonLcPredictions_13TeV_y05_FFptDepLHCb_BRpythia8_PDG2020.root).
