from jnbayes.libs.fitting_functions import func_db
from jnbayes.libs.fitting_functions import ngaus
from jnbayes.libs.etc_sub import COLOER_MAP
import matplotlib.pyplot as plt
import numpy as np

########## bayesian-sub ##########

def calc_free_energy(E_log,mean_E_log,temp,num_par_1000,data_len,logfile_header):
    num_temp = len(E_log)
    E_free = np.zeros(shape=(num_par_1000,num_temp))

    for irep in range(1,num_temp):
        tmp = 0
        for irep_sum in range(irep):
            tmp += (temp[irep_sum+1] - temp[irep_sum])*mean_E_log[:,irep_sum+1]
        E_free[:,irep] = (tmp - (np.log(temp[irep]/2/np.pi)/2 )) * data_len
    save_Efree(logfile_header,E_free)
    save_meanE_log(logfile_header,mean_E_log)

    return E_free


def save_Efree(log_header,E_free):
    log_name = log_header+"_Free_Energy.npy"    
    np.save(log_name,E_free)    
    return



def plt_Efree(pic_header,temp,E_free):
    pic_name = pic_header+"_Free_Energy.png"
    E_free_ave = np.mean(E_free,axis=0)
    E_free_std = np.std(E_free,axis=0)
    plt.plot(temp[1:-1],E_free_ave[1:-1])
    # plt.plot(temp[88:],1/(2*temp[88:]))
    plt.errorbar(temp[1:-1],E_free_ave[1:-1],yerr=E_free_std[1:-1],fmt='o', markersize=3)
    plt.xlabel("temp")
    plt.ylabel("E free")
    plt.xscale("log")
    plt.title("E_free")
    plt.ylim(min(E_free_ave[1:-1])-100,min(E_free_ave[1:-1])+1000)
    plt.savefig(pic_name)
    plt.close()
    print(pic_name,"saved!")
    return E_free_ave

def save_meanE_log(log_header,mean_E_log):
    log_name = log_header+"_meanE.npy"    
    np.save(log_name,mean_E_log)
    return


def plt_meanE_log(pic_header,temp,mean_E_log):
    mean_E_log_ave = np.mean(mean_E_log,axis=0)
    mean_E_log_std = np.std(mean_E_log,axis=0)
    mean_max = np.max(mean_E_log_ave[1:])
    mean_min = np.min(mean_E_log_ave[1:])
    pic_name = pic_header+"_meanE.png"
    plt.figure()
    plt.plot(temp[1:],mean_E_log_ave[1:])
    plt.errorbar(temp[1:],mean_E_log_ave[1:],yerr=mean_E_log_std[1:],fmt='o', markersize=3)
    plt.plot(temp[1:],1/(2*temp[1:]))
    plt.xlabel(r"$\beta$")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel(" $mean$_$E$")
    plt.ylim(top=mean_max*(1.3),bottom=mean_min*(0.7))
    plt.savefig(pic_name)
    plt.close()
    print(pic_name,"saved!")        
    return mean_E_log_ave


def get_each_Y(X,fdic,mpar):
    mpar = mpar.reshape((1,-1))
    _peak_ = fdic["peak"]
    num_peak = len(_peak_)
    conv_TF = fdic["convolve"]["TF"]
    if not conv_TF:
        tX = X*1
        Yfit = np.zeros((1,len(X)),dtype=float)
    else:
        # get conv gaus
        cc=(X[0]+X[-1])/2
        xstep = X[1]-X[0]
        tX = np.arange(2*X[0]-cc,2*X[-1]-cc,xstep)
        gaus_Y = ngaus(X,cc.reshape((1,1)),mpar[:,-1]) * abs(xstep)
        Yfit = np.zeros((1,len(tX)),dtype=float)

    # get each & total peak
    Ysngls = np.zeros((num_peak,len(X)),dtype=float)
    ipar = 0
    xc_li = []
    for ipeak, p_num in enumerate(_peak_):
        model_str = _peak_[p_num]["model"]
        tmp_par_num = func_db[model_str]["parnum"]
        tmp_arg = mpar[:,ipar:ipar+tmp_par_num]
        xc_li.append(func_db[model_str]["peak"](tmp_arg[0]))
        Ytmp = func_db[model_str]["func"](tX,tmp_arg)
        if not conv_TF:
            Yfit += Ytmp
            Ysngls[ipeak] = Ytmp[0]
        else:
            Yconv = np.convolve(Ytmp[0],gaus_Y[0],"varid")
            Ysngls[ipeak] = Yconv
            Yfit += Ytmp
        ipar += tmp_par_num
    
    # get BG
    BG_dict = fdic["BG"]
    BG_model_str = BG_dict["model"]
    BG_par_num = func_db[BG_model_str]["parnum"]
    BG_arg = mpar[:,ipar:ipar+BG_par_num]
    Ybg = func_db[BG_model_str]["func"](Yfit,BG_arg)
    Yfit += Ybg
    if not conv_TF:
        Yfit = Yfit[0]
        Ybg = Ybg[0]
    else:
        Ybg = np.convolve(Ybg[0],gaus_Y[0],"valid")
        Yfit = np.convolve(Yfit[0],gaus_Y[0],"valid")

    Ys = {}
    Ys["singls"] = Ysngls
    Ys["fit"] = Yfit
    Ys["BG"] = Ybg
    Ys["peaks"] = xc_li

    return Ys    




def plt_fit(X,Y,mpar,fdic,min_rep,min_cycle,pic_header,invertX=False):
    pic_name = pic_header+"_MAP_par.png"

    _peak_ = fdic["peak"]
    num_peak = len(_peak_)

    ### get Ys
    Ys = get_each_Y(X,fdic,mpar)
    Yfit = Ys["fit"]
    Ysngls = Ys["singls"]
    Ybg = Ys["BG"]
    xc_pos = Ys["peaks"]

    ### plot
    plt.scatter(X,Y,color="k",s=1)
    plt.title(f"MAP (rep={min_rep}, cycle={min_cycle})")
    plt.plot(X,Y,c="k")
    plt.plot(X,Yfit,c="r")
    plt.plot(X,Ybg,c="gray")    
    if invertX:
        plt.gca().invert_xaxis()
    for ipeak in range(num_peak):
        plt.plot(X,Ysngls[ipeak]+Ybg,c=COLOER_MAP[ipeak%10])
        if xc_pos[ipeak]:
            plt.vlines(xc_pos[ipeak],min(Y),max(Y),linestyles="dashed",colors=COLOER_MAP[ipeak%10])
    plt.savefig(pic_name)
    plt.close()
    print(pic_name,"saved!")
    print("MAP param ->",mpar)
    
    return



def list_write(fout, parname, par):
    fout.write(parname+" = \n")
    for i in range(len(par)):
        tmp_str = str(par[i])
        fout.write(str(i)+"\t"+tmp_str+"\n")    
    fout.write("\n")
    return
def valu_write(fout, parname, par):
    fout.write(parname+" = \n")
    fout.write("\t"+str(par)+"\n")
    fout.write("\n")
    return
