from machine_learning_hep.processer import Processer
from machine_learning_hep.logger import get_logger
import logging

class ProcesserJets(Processer): # pylint: disable=invalid-name, too-many-instance-attributes
    species = "processer"
    logger = get_logger()

    def __init__(self, case, datap, run_param, mcordata, p_maxfiles,
                d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                d_results, typean, runlisttrigger, d_mcreweights):
        super().__init__(case, datap, run_param, mcordata, p_maxfiles,
                        d_root, d_pkl, d_pklsk, d_pkl_ml, p_period, i_period,
                        p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                        p_frac_merge, p_rd_merge, d_pkl_dec, d_pkl_decmerged,
                        d_results, typean, runlisttrigger, d_mcreweights)
        self.logger.info("initialized processer for D0 jets")
