from nsbayes.libs.fitting_functions import func_db
from nsbayes.libs.fitting_functions import smearing_gaus
import numpy as np

def get_fitting_container(fdict):
    fit_cont = []
    BG_cont = []

    # peak par
    ipar = 0
    peaks = fdict["peak"]
    for p_num in peaks:
        model_str = peaks[p_num]["model"]
        par_num = func_db[model_str]["parnum"]
        tmp_func = func_db[model_str]["func"]
        tmp_slice = slice(ipar,ipar+par_num)
        ipar += par_num
        fit_cont.append([tmp_func,tmp_slice])
    fit_cont = list(zip(*fit_cont))

    # BG par
    BG_dict = fdict["BG"]
    BG_model_str = BG_dict["model"]
    BG_par_num = func_db[BG_model_str]["parnum"]
    bg_func = func_db[BG_model_str]["func"]
    bg_slice = slice(ipar,ipar+BG_par_num)
    BG_cont.append(bg_func)
    BG_cont.append(bg_slice)

    return fit_cont, BG_cont

def get_Y(X:np.ndarray,p_now:np.ndarray,fit_cont:list, BG_cont:list, conv_TF=False)-> np.ndarray:
    """get Y np.array(2d)

    Todo:
        You should consider the argument types and contents,
        when you modify this function

    Args:
        X (np.array)1d: X np 1d array [Xlen]
        p_now (np.array): params np 2d array [nRep][nPar]
        fit_cont (list): fitting contents, [pyfunc obj [nPeak], pyslice obj [] ]
        BG_cont (list): back ground contents, [pyfunc obj [nBG], pyslice obj [] ]
        conv_TF (bool, optional): gaussian convolve Y data. Defaults to False.

    Returns:
        Y (np.array)2d: 


    """
    if conv_TF is False:
        tX = X*1
    else:
        # get conv gaus
        cc=(X[0]+X[-1])/2
        xstep = X[1]-X[0]
        tX = np.arange(2*X[0]-cc,2*X[-1]-cc,xstep)
        gaus_Y = smearing_gaus(X,np.tile([cc],(len(p_now),1)),p_now[:,-1]) * abs(xstep)

    # get peaks
    peak_Y = np.zeros((len(p_now),len(tX)),dtype=float)
    funcs = fit_cont[0]
    slices = fit_cont[1]
    for fn, sl in zip(funcs, slices):
        peak_Y += fn(tX,p_now[:,sl])

    # get BG 
    BG_func = BG_cont[0]
    BG_slice = BG_cont[1]
    peak_Y += BG_func(peak_Y,p_now[:,BG_slice])

    if conv_TF is False:
        return peak_Y

    return np.array([np.convolve(peak_Y[i],gaus_Y[i],"varid") for i in range(len(p_now))])



def get_Energy(y_raw,y_tmp,data_len,sigmaE,X_slice):
    E = 1/2/data_len/sigmaE**2*np.linalg.norm((y_raw-y_tmp)[:,X_slice],axis=1)**2
    return E

def update_param(
        ic:int, log_length:int, E_now:np.ndarray, data_len:int, 
        sigmaE:float, x:np.ndarray, y_raw:np.ndarray, p_now:np.ndarray,
        stepsize_p:np.ndarray, ratio_p_log:np.ndarray, num_temp:np.ndarray,
        p_min_tl:np.ndarray, p_max_tl:np.ndarray, temp:np.ndarray,
        fit_cont:list, BG_cont:list, X_slice:slice, conv_TF:bool) -> tuple:
    """RXMC sub routine

    Todo:
        It is better to fix this, if the current getY is insufficient, 


    Contents:

        for loop in (nPar times)

            0. shake param (one par) <- content def getY():
            1. calc Y,E
            2. calc & judge to adpt par
            3. update & log par

    Args:
        ic (int): loop num
        log_length (int): last 3 digits of loop num
        E_now (np.ndarray): Energy, [nRep]
        data_len (int): Xlen
        sigmaE (float): gaussian noise of data
        x (np.ndarray): Xraw [Xlen]
        y_raw (np.ndarray): Yraw [Xlen]
        p_now (np.ndarray): paramter, [nRep][nPar]
        stepsize_p (np.ndarray): param range, [nRep][nPar]
        ratio_p_log (np.ndarray): log ndarray (in bool), [nRep][nPar][1000]
        num_temp (np.ndarray): num of replica
        p_min_tl (np.ndarray): min param range
        p_max_tl (np.ndarray): max param range
        temp (np.ndarray): temprature of replica
        fit_cont (list): fitting contents, [pyfunc obj [nPeak], pyslice obj [] ]
        BG_cont (list): back ground contents, [pyfunc obj [nPeak], pyslice obj [] ]
        X_slice (slice): X range
        conv_TF (bool): False->don't convolv, True-> do convolv

    Returns:
        p_now (np.ndarray) : now param [nRep][nPar]
        E_now (np.ndarray) : now Energy [nRep]
    """

    log_mod_cycl = ic%log_length
    num_para = np.size(p_max_tl[0])
    p_now[0,:] = (p_max_tl[0]-p_min_tl[0])*np.random.rand()+p_min_tl[0]   ######## sampling (temp==0)
    adpt_par_nums = mk_shuff_turn(num_temp,num_para)
    for ishpar in range(num_para):
        p_next = 1*p_now
        rand = 2*np.random.random((num_temp,num_para)) - 1
        shake_TF = adpt_par_nums==ishpar
        p_next += (shake_TF*1.0) * stepsize_p * rand
        min_tf = p_min_tl < p_next
        max_tf = p_next < p_max_tl
        p_next = np.clip(p_next, p_min_tl, p_max_tl)

        ######## calc E_next
        y_tmp = get_Y(x,p_next,fit_cont, BG_cont, conv_TF)
        E_next = get_Energy(y_raw,y_tmp,data_len,sigmaE,X_slice)

        ######## calc porb for judge
        exp_cont = np.clip(-data_len*temp*(E_next-E_now), -708, 709)
        prob = np.exp(exp_cont) - np.random.rand(num_temp)

        ######## update & log parameter
        prob_tf = (prob > 0).reshape((num_temp,1))
        adpt_TF = prob_tf & min_tf & max_tf & shake_TF
        adpt_rep_TF = np.any(adpt_TF,axis=-1)
        p_now = np.where(adpt_TF==True,p_next,p_now) # Don't write "is", write "=="!!
        E_now = np.where(adpt_rep_TF==True,E_next,E_now)
        ratio_p_log[:,:,log_mod_cycl] += adpt_TF * 1

    ratio_p_log[0,:,log_mod_cycl]=0 
    E_now[0] = E_next[0]
    return p_now, E_now

def exc_replica(ic,p_now,E_now,temp,data_len,num_temp,log_mod_cycle,ratio_exc_rep_log):
    eo = np.mod(ic,2)
    if(eo==0):
        exp_cont = np.clip(data_len*(temp[eo+1::2]-temp[eo::2])*(E_now[eo+1::2]-E_now[eo::2]), -708, 709)
        prob = np.exp(exp_cont) - np.random.rand(np.int32(num_temp/2))
        J = 2*np.array(np.where(prob>0))
    else:
        exp_cont = np.clip(data_len*(temp[eo+1:num_temp-1:2]-temp[eo:num_temp-1:2])*(E_now[eo+1:num_temp-1:2]-E_now[eo:num_temp-1:2]), -708, 709)
        prob = np.exp(exp_cont) - np.random.rand(np.int32(num_temp/2)-1)
        J = 2*np.array(np.where(prob>0))+1
    for j in J[0]:
        s = 1*p_now[j,:]
        p_now[j,:] = 1*p_now[j+1,:]
        p_now[j+1,:] = s
        ratio_exc_rep_log[j][log_mod_cycle] = 1

    s = E_now[J]
    E_now[J] = E_now[J+1]
    E_now[J+1] = s
    return

def get_ratio_log(num_temp,par_num):
    ratio_log = np.zeros((num_temp,par_num))
    return ratio_log

def renew_p_log(log_name,loop_name,data_name,old_log,tmp_arg):
    num_temp,par_num,log_length = tmp_arg[0],tmp_arg[1],tmp_arg[2]
    np.save(log_name+data_name+loop_name,old_log)
    del old_log
    nw_log = np.zeros((num_temp,par_num,log_length))
    return nw_log

def renew_E_log(log_name,loop_name,data_name,old_log,tmp_arg):
    num_temp,_,log_length = tmp_arg[0],tmp_arg[1],tmp_arg[2]
    np.save(log_name+data_name+loop_name,old_log)
    del old_log
    nw_log = np.zeros((num_temp,log_length))
    return nw_log

def mk_shuff_turn(n_rep,n_par):
    tmp = np.arange(0,n_par)
    ret = np.zeros((n_rep,n_par),dtype=np.int64)
    for irep in range(n_rep):
        np.random.shuffle(tmp)
        ret[irep] = tmp
    return ret