"""plot figures(main)
"""
from nsbayes.libs.plt_sub import plt_par_log, plt_step_temp
from nsbayes.libs.plt_sub import plt_E_log
from nsbayes.libs.plt_sub import plt_p_ratio
from nsbayes.libs.plt_sub import plt_exc_rep_log
from nsbayes.libs.rxmc_cls import Rxmc_ctrl
from nsbayes.libs.etc_sub import NUM_OF_THIS_PC_MULTIPLE_CPU_CORES as NUM_CORE
from multiprocessing import Pool, log_to_stderr
import glob
import numpy as np
import gc


def plt_figures(rx_cls:Rxmc_ctrl, sw_plt_all_p_log=False, sw_plt_all_E_log=False):

    df = rx_cls.df
    dir_dict = df["dpar"]
    func_dict = df["fpar"]
    run_dict = df["rpar"]
    ######## read dict
    _p_init_ = rx_cls.get_param("par","init")    
    cycle = run_dict["cycle"]
    num_temp = run_dict["num_temp"]
    burn_in_length =run_dict["burn_in_length"]
    model_name = dir_dict["model_name"]

    ######## calc dict
    par_num = len(_p_init_)
    logfile_header = rx_cls.logfile_header
    picfile_header = rx_cls.picfile_header
    
    ######### temp, temp_inv
    temp_files = glob.glob(logfile_header+"*temp.npy")
    temp = np.load(temp_files[0],allow_pickle=True)

    ######### p_log
    p_log_files = glob.glob(logfile_header+"_p_log_*.npy")
    p_log_files = sorted(p_log_files)
    if sw_plt_all_p_log:
        p_log = np.empty((num_temp,par_num,0))
        for i in range(len(p_log_files)):
            tfile = p_log_files[i]
            np_tmp = np.load(tfile,allow_pickle=True)
            p_log = np.append(p_log,np_tmp,axis=2)
        args_p_log = []
        pool_p_log = Pool(NUM_CORE)
        for irep in range(num_temp):
            for ipar in range(par_num):
                tmp_p_log = p_log[irep][ipar][burn_in_length:]
                args_p_log.append((picfile_header,irep,ipar,tmp_p_log))
                plt_par_log(picfile_header,irep,ipar,tmp_p_log)
        _ = pool_p_log.starmap(plt_par_log, args_p_log)
        del p_log
        del pool_p_log
    gc.collect()

    ######### E_log
    E_log_files = glob.glob(logfile_header+"_E_log_*.npy")
    E_log_files = sorted(E_log_files)
    if sw_plt_all_E_log:
        E_log = np.empty((num_temp,0))
        for ifile in range(len(E_log_files)):
            tfile = E_log_files[ifile]
            tmp = np.load(tfile,allow_pickle=True)
            E_log = np.append(E_log,tmp,axis=1)
        args_E_log = []
        pool_E_log = Pool(NUM_CORE)
        for irep in range(num_temp):
            tmp_E_log = E_log[irep][burn_in_length:]
            plt_E_log(picfile_header,irep,tmp_E_log)
            args_E_log.append((picfile_header,irep,tmp_E_log))
        _ = pool_E_log.starmap(plt_E_log,args_E_log)
        del pool_E_log
        del E_log
    gc.collect()

    ######### adopt ratio log(ratio_P_log) vs temp
    ratio_p_log_files = glob.glob(logfile_header+"_ratio_p_log_*.npy")
    ratio_p_log_files = sorted(ratio_p_log_files)
    ratio_p_log = np.empty((num_temp,par_num,0))
    for ifile in range(len(ratio_p_log_files)):
        tfile = ratio_p_log_files[ifile]
        np_tmp = np.load(tfile,allow_pickle=True)
        ratio_p_log = np.append(ratio_p_log,np_tmp,axis=2)
    sum_p_log = np.sum(ratio_p_log[:,:,burn_in_length:],axis=2)
    for ipar in range(par_num):
        ratio = sum_p_log[:,ipar]/(cycle-burn_in_length)
        plt_p_ratio(picfile_header,ipar,temp,ratio)
    
    ######### ratio_exc_rep_log vs temp
    ratio_exc_rep_files =  glob.glob(logfile_header+"*ratio_exc_rep_log*.npy")
    ratio_exc_rep_files = sorted(ratio_exc_rep_files)
    ratio_exc_rep_log = np.empty((num_temp,0))
    for ifile in range(len(ratio_exc_rep_files)):
        tfile = ratio_exc_rep_files[ifile]
        np_tmp = np.load(tfile,allow_pickle=True)
        ratio_exc_rep_log = np.append(ratio_exc_rep_log,np_tmp,axis=1)        
    sum_exc_rep = np.sum(ratio_exc_rep_log[:,burn_in_length:],axis=1)
    plt_exc_rep_log(picfile_header,model_name,temp,sum_exc_rep,cycle,burn_in_length)

    ########## stepsize vs temp
    stepsize_files = glob.glob(logfile_header+"*_stepsize.npy")
    stepsize = np.load(stepsize_files[0],allow_pickle=True)
    for ipar in range(len(stepsize[0])):
        plt_step_temp(picfile_header,ipar,temp,stepsize)


    return







