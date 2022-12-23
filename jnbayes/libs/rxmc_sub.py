from jnbayes.libs.fitting_functions import func_db
from jnbayes.libs.fitting_functions import smearing_gaus
import numpy as np

def get_fitting_container(fdict):
    fit_cont = []
    BG_cont = []
    #### fittingの関数を呼び出して格納しておく
    #### "fitting_function.py"の中にある関数そのものを登録しておく

    # peak par
    ipar = 0
    peaks = fdict["peak"]
    for p_num in peaks:
        ## モデル名を取得
        model_str = peaks[p_num]["model"]
        ## パラメータ数を取得
        par_num = func_db[model_str]["parnum"]
        ## Pyhonの関数そのものを取得
        tmp_func = func_db[model_str]["func"]
        ## XYのスライスデータを取得
        tmp_slice = slice(ipar,ipar+par_num)
        ipar += par_num
        ## 追加
        fit_cont.append([tmp_func,tmp_slice])
    ### cont[[p1],[p2]]となっていたものを[[model],[npar],[func]]に転置
    fit_cont = list(zip(*fit_cont))

    ### 基本は上と処理は変わらない
    # BG par
    BG_dict = fdict["BG"]
    BG_model_str = BG_dict["model"]
    BG_par_num = func_db[BG_model_str]["parnum"]
    bg_func = func_db[BG_model_str]["func"]
    bg_slice = slice(ipar,ipar+BG_par_num)
    BG_cont.append(bg_func)
    BG_cont.append(bg_slice)

    return fit_cont, BG_cont

def get_Y(X:np.ndarray,p_now:np.ndarray,fit_cont:list, BG_cont:list, conv_TF=False):
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
    ### 特殊関数用、ふつうは使わない
    if conv_TF is False:
        tX = X*1
    else:
        # get conv gaus
        cc=(X[0]+X[-1])/2
        xstep = X[1]-X[0]
        tX = np.arange(2*X[0]-cc,2*X[-1]-cc,xstep)
        gaus_Y = smearing_gaus(X,np.tile([cc],(len(p_now),1)),p_now[:,-1]) * abs(xstep)

    ### 0で埋められたYの配列を生成
    peak_Y = np.zeros((len(p_now),len(tX)),dtype=float)
    ### peak部分を計算してYの配列に加算
    # get peaks
    funcs = fit_cont[0]
    slices = fit_cont[1]
    fn : function ## Python関数そのもの
    sl : slice    ## スライスオブジェクト
    for fn, sl in zip(funcs, slices):
        peak_Y += fn(tX,p_now[:,sl])

    # get BG 
    BG_func = BG_cont[0]
    BG_slice = BG_cont[1]
    peak_Y += BG_func(peak_Y,p_now[:,BG_slice])

    #### 特殊関数不使用系
    if conv_TF is False:
        return peak_Y  ## ふつうはこっちが返る

    #### 特殊関数用
    return np.array([np.convolve(peak_Y[i],gaus_Y[i],"varid") for i in range(len(p_now))])



def get_Energy(y_raw,y_tmp,data_len,sigmaE,X_slice):
    E = 1/2/data_len/sigmaE**2*np.linalg.norm((y_raw-y_tmp)[:,X_slice],axis=1)**2
    return E

def update_param(
        ic:int, log_length:int, E_now:np.ndarray, data_len:int, 
        sigmaE:float, x:np.ndarray, y_raw:np.ndarray, p_now:np.ndarray,
        stepsize_p:np.ndarray, ratio_p_log:np.ndarray, num_temp:np.ndarray,
        p_min_tl:np.ndarray, p_max_tl:np.ndarray, temp:np.ndarray,
        fit_cont:list, BG_cont:list, X_slice:slice, conv_TF:bool):
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
    # 現在がlog_length番目のサイクルかどうか判定
    log_mod_cycl = ic%log_length
    # パラメータ数を計算
    num_para = np.size(p_max_tl[0])
    # 0番目のレプリカのパラメータを全探索領域から振る
    p_now[0,:] = (p_max_tl[0]-p_min_tl[0])*np.random.rand()+p_min_tl[0]   ######## sampling (temp==0)
    # パラメータを振る順番を乱数で決める
    adpt_par_nums = mk_shuff_turn(num_temp,num_para)
    
    ### 全パラメータでくりかえす
    for ishpar in range(num_para):
        p_next = 1*p_now ## 参照避け
        # np.randomは0~1のため、-1~1にする（rand[nrep][npar]）
        rand = 2*np.random.random((num_temp,num_para)) - 1
        # bool型の配列を生成（sk_TF[npar]）、乱数を振るパラメータだけTrueになる
        shake_TF = adpt_par_nums==ishpar
        # 乱数を振るパラメータにのみ乱数が足され、それ以外に0が足される
        p_next += (shake_TF*1.0) * stepsize_p * rand
        # パラメータが最大最小に入っているか確認、(m_tf[nrep][npar])
        min_tf = p_min_tl < p_next
        max_tf = p_next < p_max_tl
        # 入っていなければサチらせる
        p_next = np.clip(p_next, p_min_tl, p_max_tl)

        ######## calc E_next
        y_tmp = get_Y(x,p_next,fit_cont, BG_cont, conv_TF)
        E_next = get_Energy(y_raw,y_tmp,data_len,sigmaE,X_slice)

        ######## calc porb for judge
        # numpy.exp()の計算範囲をE_nextが超えていればサチらせる
        exp_cont = np.clip(-data_len*temp*(E_next-E_now), -708, 709)
        # 乱数を振って値を採択するか計算(0以上で採択)
        prob = np.exp(exp_cont) - np.random.rand(num_temp)

        ######## update & log parameter
        # 採択かどうか判定
        prob_tf = (prob > 0).reshape((num_temp,1))
        # 全部の判定でTrueが通った場合のみ採択(adpt_TF[nrep][npar])
        adpt_TF = prob_tf & min_tf & max_tf & shake_TF
        # 採択が通ったレプリカかどうかanyで判定(adpt_rep_TF[npar])
        adpt_rep_TF = np.any(adpt_TF,axis=-1)
        # 採択が通ったパラメータ、エネルギーのみ更新
        p_now = np.where(adpt_TF==True,p_next,p_now) # Don't write "is", write "=="!!
        E_now = np.where(adpt_rep_TF==True,E_next,E_now)
        
        # ログに記入
        ratio_p_log[:,:,log_mod_cycl] += adpt_TF * 1

    # 0番目の配列は常に「採択されていない」にする
    ratio_p_log[0,:,log_mod_cycl]=0 
    # 0番目の配列のエネルギーも更新
    E_now[0] = E_next[0]
    return p_now, E_now

def exc_replica(ic,p_now,E_now,temp,data_len,num_temp,log_mod_cycle,ratio_exc_rep_log):
    ### 奇数か偶数か判定
    eo = np.mod(ic,2)
    # 偶数の時
    if(eo==0):
        # numpy.exp()の計算範囲をexp_contが超えていればサチらせる
        # [::2]で一つ飛ばしで読んでいる
        exp_cont = np.clip(data_len*(temp[eo+1::2]-temp[eo::2])*(E_now[eo+1::2]-E_now[eo::2]), -708, 709)
        # 交換するか乱数を振って判定(0以上で交換)
        prob = np.exp(exp_cont) - np.random.rand(np.int32(num_temp/2))
        ## 交換判定、判定用配列を１個飛ばしで作ってしまったので
        ## np.where(prob>0)で出てくるインデックスの数字を全部2倍にして元の配列に適用できるようにした
        J = 2*np.array(np.where(prob>0))
    # 奇数の時
    else:
        exp_cont = np.clip(data_len*(temp[eo+1:num_temp-1:2]-temp[eo:num_temp-1:2])*(E_now[eo+1:num_temp-1:2]-E_now[eo:num_temp-1:2]), -708, 709)
        prob = np.exp(exp_cont) - np.random.rand(np.int32(num_temp/2)-1)
        ## np.where(prob>0)で出てくるインデックスの数字を全部2倍して(奇数なので)1を足して元の配列に適用できるようにした
        J = 2*np.array(np.where(prob>0))+1
    
    ### 交換判定がTrueだったもののみ1つ下の階層(ステップ細かい側)と交換
    for j in J[0]:
        s = 1*p_now[j,:]
        p_now[j,:] = 1*p_now[j+1,:]
        p_now[j+1,:] = s
        ratio_exc_rep_log[j][log_mod_cycle] = 1

    ### 0番目と1番目は毎回必ず更新する
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