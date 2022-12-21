"""RXMC program(main)
"""

from nsbayes.libs.rxmc_sub import get_Y
from nsbayes.libs.rxmc_sub import get_Energy
from nsbayes.libs.rxmc_sub import get_fitting_container
from nsbayes.libs.rxmc_sub import exc_replica
from nsbayes.libs.rxmc_sub import renew_E_log
from nsbayes.libs.rxmc_sub import renew_p_log
from nsbayes.libs.rxmc_sub import update_param
from nsbayes.libs.rxmc_cls import Rxmc_ctrl
import numpy as np
import time
import gc




def run_rxmc(rx_cls:Rxmc_ctrl, sw_hyper_d_is_zero=False):
    """
    
    ## Calc RXMC Programs

    Args:
        rx_cls (Rxmc_ctrl): read json in RXMC instance
        sw_hyper_d_is_zero (bool, optional): dd(hyper param) parameter to zeros. Defaults to False.
    """
    ######## read raw data
    X = rx_cls.X
    Y = rx_cls.Y
    ######## deploy param to meaningful parameters
    func_dict = rx_cls.df["fpar"]
    run_dict = rx_cls.df["rpar"]
    ######## read dict
    _p_init_ = rx_cls.get_param("par","init")
    _step_ = rx_cls.get_param("hyper","step")
    _alpha_ = rx_cls.get_param("hyper","alpha")
    _d_ = rx_cls.get_param("hyper","dd")
    _p_min_ = rx_cls.get_param("limit","rng_min") 
    _p_max_ = rx_cls.get_param("limit","rng_max")
    log_length = rx_cls.df["dpar"]["log_par_rxmc_loops"]
    cycle = run_dict["cycle"]
    num_temp = run_dict["num_temp"]
    sigmaE = func_dict["noise"]["sigmaE"]    
    if sw_hyper_d_is_zero:
        _d_ = rx_cls.ctrl_peram_zeros(_d_)
    X_slice = rx_cls.xslice
    ######## get fitting container
    fit_cont, BG_cont = get_fitting_container(func_dict)


    ######## calc dict
    data_len = len(X)
    par_num = len(_p_init_)
    _p_min_tl_ = np.tile(_p_min_,(num_temp,1))
    _p_max_tl_ = np.tile(_p_max_,(num_temp,1))
    dt_log_name = rx_cls.logfile_header
    if func_dict["convolve"]["TF"]:
        conv_TF = True
    else:
        conv_TF = False

    ######## calc first ones
    Yraw = Y.reshape(1,-1)
    _p_now_ = np.tile(_p_init_,(num_temp,1))

    Ytmp = get_Y(X,_p_now_,fit_cont, BG_cont, conv_TF)

    _E_now_ = get_Energy(Yraw,Ytmp,data_len,sigmaE,X_slice)

    ######## cleate temp
    temp, _stepsize_p_ = rx_cls.get_temp_step(par_num,_step_,_alpha_,_d_)
    
    
    ######## define for RXMC
    p0_log = np.zeros((num_temp,par_num,log_length),dtype=np.int32)
    E0_log = np.zeros((num_temp,log_length),dtype=np.int32)
    _p_log_ = 1.0*p0_log
    _E_log_ = 1.0*E0_log
    _ratio_p_log_ = 0*p0_log
    _ratio_exc_rep_log_ = 0*E0_log
    tm_start = time.time()
    
    ######## event loop <----- MOST IMPORTANT!!!
    print("cycle =",cycle)
    for icycl in range(cycle):
        
        ########## print progress of RXMC
        if icycl%100==0:
            tm_stop = time.time()
            time_str = str(round(100/(tm_stop-tm_start+(10e-5)),1)) + "Hz"
            print("event loop =",rx_cls.model_name, icycl,time_str)
            tm_start = tm_stop
        log_mod_cycl = icycl%log_length

        ########## update all parameters
        _p_now_, _E_now_ = update_param(icycl,log_length,_E_now_,data_len,sigmaE,X,Yraw,
                        _p_now_,_stepsize_p_,_ratio_p_log_,num_temp,
                        _p_min_tl_,_p_max_tl_,temp,fit_cont, BG_cont,X_slice,conv_TF)
        _p_log_[:,:,log_mod_cycl] = _p_now_

        ########## exchange replica
        exc_replica(icycl,_p_now_,_E_now_,temp,data_len,num_temp,log_mod_cycl,_ratio_exc_rep_log_)
        _E_log_[:,log_mod_cycl] = _E_now_

        ######## save log pars
        if (icycl+1)%log_length == 0:
            loop_num_str = str(int((icycl+1)/log_length)).zfill(3)
            tmp_arg = [num_temp,par_num,log_length]
            _p_log_ = renew_p_log(dt_log_name,loop_num_str,"_p_log_",_p_log_,tmp_arg)
            _E_log_ = renew_E_log(dt_log_name,loop_num_str,"_E_log_",_E_log_,tmp_arg)
            _ratio_p_log_ = renew_p_log(dt_log_name,loop_num_str,"_ratio_p_log_",_ratio_p_log_,tmp_arg)
            _ratio_exc_rep_log_ = renew_E_log(dt_log_name,loop_num_str,"_ratio_exc_rep_log_",_ratio_exc_rep_log_,tmp_arg)
            print("saved log par",log_length,"events")
            gc.collect() #### Garbage Collector cleans memory
    print("event loop =",rx_cls.model_name,"F!!!!")
    return    



