from machine_learning_hep.processer import Processer

class ProcesserD0jets(Processer): # pylint: disable=invalid-name, too-many-instance-attributes
    # Class Attribute
    species = "processer"

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
    print("initialized processer for D0 jets")
