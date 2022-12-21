"""bayesian program(main)
"""
from nsbayes.libs.bayes_sub import calc_free_energy
from nsbayes.libs.bayes_sub import plt_fit
from nsbayes.libs.bayes_sub import valu_write
from nsbayes.libs.bayes_sub import list_write
from nsbayes.libs.plt_sub import plt_par_log
from nsbayes.libs.rxmc_cls import Rxmc_ctrl
from multiprocessing import Pool
from nsbayes.libs.etc_sub import NUM_OF_THIS_PC_MULTIPLE_CPU_CORES as NUM_CORE
import numpy as np
import glob
import os


def run_bayesian(rx_cls:Rxmc_ctrl,sw_plt_finep_multi=True):
    """ Calc bayesian Programs

    Args:
        rx_cls (Rxmc_ctrl): Class Object

    Returns:
        _type_: _description_
    """

    ######## read raw data
    X = rx_cls.X
    Y = rx_cls.Y

    ######## deploy param to meaningful parameters
    func_dict = rx_cls.df["fpar"]
    run_dict = rx_cls.df["rpar"]
    ######## to list
    _p_init_ = rx_cls.get_param("par","init")
    
    ######## to float
    cycle = run_dict["cycle"]
    num_temp = run_dict["num_temp"]
    burn_in_length =run_dict["burn_in_length"]
    log_length = rx_cls.df["dpar"]["bayes_Efree_par_rxmc_loops"]
    data_len = len(X)
    par_num = len(_p_init_)
    ######## to str
    logfile_header = rx_cls.logfile_header
    picfile_header = rx_cls.picfile_header

    ### load datas temp
    temp_files = glob.glob(logfile_header+"*temp.npy")
    temp = np.load(temp_files[0],allow_pickle=True)
    ### load datas Error
    E_log_files = glob.glob(logfile_header+"_E_log_*.npy")
    E_log_files = sorted(E_log_files)
    E_log = np.empty((num_temp,0))
    for ifile in range(len(E_log_files)):
        tfile = E_log_files[ifile]
        tmp = np.load(tfile,allow_pickle=True)
        E_log = np.append(E_log,tmp,axis=1)
    ### load datas param
    p_log_files = glob.glob(logfile_header+"_p_log_*.npy")
    p_log_files = sorted(p_log_files)
    p_log = np.empty((num_temp,par_num,0))
    for ifile in range(len(p_log_files)):
        tfile = p_log_files[ifile]
        np_tmp = np.load(tfile,allow_pickle=True)
        p_log = np.append(p_log,np_tmp,axis=2)

    ####### calc free energy
    E_free, E_free_ave, mean_E_ave = calc_free_energy(E_log,temp,cycle,burn_in_length,log_length,data_len,logfile_header,picfile_header)

    ######## MAP
    iE_min_rep = np.argmin(E_free[:,1:-1],axis=-1)+1
    iE_min_rep_ave = np.argmin(E_free_ave[1:-1],axis=-1)+1
    Ef_min = np.min(E_free[:,1:-1],axis=-1)
    Ef_min_ave = np.min(mean_E_ave[1:-1],axis=-1)

    noise_result = 1/(temp[iE_min_rep_ave])**0.5
    print("最適な逆温度のインデックス",iE_min_rep)
    print("最適な逆温度のインデックス(平均)",iE_min_rep_ave)
    print("自由エネルギーの最小値",Ef_min)
    print("自由エネルギーの最小値(平均)",Ef_min_ave)
    print("推定されたノイズの標準偏差",noise_result)

    iE_min_cycle = np.argmin(E_log[iE_min_rep_ave])
    print(iE_min_cycle,"clcle")
    
    map_par = p_log[iE_min_rep_ave,:,iE_min_cycle]
    plt_fit(X,Y,map_par,func_dict,iE_min_rep_ave,iE_min_cycle,picfile_header)
    fout = open(logfile_header+"result.txt","w")
    valu_write(fout,"最適な逆温度のインデックス",noise_result)
    list_write(fout,"自由エネルギーの最小値",Ef_min)
    valu_write(fout,"自由エネルギーの最小値(平均)",Ef_min_ave)
    list_write(fout,"推定されたノイズの標準偏差",iE_min_rep)
    valu_write(fout,"推定されたノイズの標準偏差(平均)",iE_min_rep_ave)
    list_write(fout,"MAP_param",map_par)
    fout.close()

    ### plt fine p_log
    del_pics = glob.glob(picfile_header+"_finerep*")
    for delpic in del_pics:
        os.remove(delpic)
    args_p_log = []
    
    ### multi thled
    if sw_plt_finep_multi is True:
        pool_p_log = Pool(NUM_CORE)
        for ipar in range(par_num):
            tmp_p_log = p_log[iE_min_rep_ave][ipar][burn_in_length:]
            args_p_log.append((picfile_header+"_finerep",iE_min_rep_ave,ipar,tmp_p_log))
        _ = pool_p_log.starmap(plt_par_log, args_p_log)

    #### single
    else:
        for ipar in range(par_num):
            tmp_p_log = p_log[iE_min_rep_ave][ipar][burn_in_length:]
            plt_par_log(picfile_header+"_finerep",iE_min_rep_ave,ipar,tmp_p_log)
        

    res_MAP = {}
    res_MAP["Ef_min"] = Ef_min_ave
    res_MAP["noise_result"] = noise_result
    res_MAP["est_rep"] = iE_min_rep_ave
    res_MAP["MAP_param"] = map_par
    res_MAP["rawX"] = X
    res_MAP["rawY"] = Y
    res_MAP["func_dict"] = func_dict

    return res_MAP








