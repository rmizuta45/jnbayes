from jnbayes.libs.etc_sub import COLOER_MAP
from jnbayes.libs.bayes_sub import get_each_Y
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import math as mt

############ M O D E L   S E L E C T - s u b ############
def calc_prob(Efree,peak_model_strs,num_par_loops,pfname):
    minE = np.min(Efree[:,:,1:])
    cEfree = Efree[:,:,1:] - minE
    ex_cEfree = np.where(cEfree<707.0,np.exp(-cEfree),0)
    ex_cEfree_sum = np.sum(ex_cEfree,axis=-1)
    ex_cEfree_sum_sum = np.sum(ex_cEfree_sum,axis=0).reshape((1,len(ex_cEfree[0]),1))

    prob_b = ex_cEfree / ex_cEfree_sum_sum
    prob = np.sum(prob_b,axis=-1)

    round_prob_ave, round_prob_std = plt_prob_bar(prob,peak_model_strs,num_par_loops,pfname)   
    min_prob_num = np.argmin(round_prob_ave)
    return min_prob_num


def plt_prob_bar(prob,peak_model_strs,num_par_loops,pfname):
    nmodel = len(prob)
    nloop = len(prob[0])
    round_prob_ave = np.round(np.average(prob*100,axis=-1),decimals=1)
    round_prob_std = np.round(np.std(prob*100,axis=-1),decimals=1)
    print("-> peak probs %(average)=",round_prob_ave)

    ### plt
    pic_file_footer = "result_prob_peak_model.png"
    pic_name = pfname+pic_file_footer
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,2)
    x_loop = range(nloop)
    for im in range(nmodel):
        tmp_lbl = peak_model_strs[im]
        plt.plot(x_loop,prob[im]*100,c=COLOER_MAP[im],label=tmp_lbl)
        plt.scatter(x_loop,prob[im]*100,s=3,c=COLOER_MAP[im])
    plt.legend()
    plt.ylabel("prob [%]")
    plt.xlabel("nloops x "+str(num_par_loops))
    plt.ylim(0,100)
    plt.subplot(1,2,1)

    plt.bar(peak_model_strs,round_prob_ave)
    # plt.bar(peak_model_strs,round_prob_ave,yerr=round_prob_std)
    plt.xticks(rotation=30)
    plt.tick_params(labelsize=7)
    plt.xlabel("peak model")
    plt.ylabel("prob [%]")
    for i in range(len(round_prob_ave)):
        tmp_txt = str(round_prob_ave[i])+"%"
        # tmp_txt = str(round_prob_ave[i])+r"$\pm$"+str(round_prob_std[i])+"%"
        fsize = int(50/len(round_prob_ave))
        plt.text(i,round_prob_ave[i]+5,tmp_txt,horizontalalignment="center",fontsize=fsize)
    plt.ylim(0,100)
    plt.savefig(pic_name)
    plt.close()

    return round_prob_ave, round_prob_std




def plt_Efree_result(Efree,temp,peak_models,pfname):
    ### plt
    pic_file_footer = "result_Efree.png"
    pic_name = pfname+pic_file_footer
    plt.figure(figsize=(15,5))

    Efree_ave = np.mean(Efree,axis=1)
    Efree_std = np.std(Efree,axis=1)

    # all range plt
    plt.subplot(1,2,1)
    plt.xlabel("temp")
    plt.xscale("log")
    plt.ylabel("Efree")
    for im in range(len(Efree_ave)):
        plt.plot(temp[im,1:-1],Efree_ave[im,1:-1], c=COLOER_MAP[im],label=peak_models[im])
        # plt.errorbar(temp[im,1:-1],Efree_ave[im,1:-1],yerr=Efree_std[im,1:-1],color=COLOER_MAP[im],fmt='o', markersize=3)
    plt.legend()

    # few range plt
    plt.subplot(1,2,2)
    plt.xlabel("temp")
    plt.xscale("log")
    plt.ylabel("Efree")
    Y_LIM_LOW_MERGINE = 1  # index
    Y_LIM_HIGH_MERGINE = 7  # index
    X_LIM_LOW_MERGINE = 3  # index
    X_LIM_HIGH_MERGINE = 5  # index
    plt.ylim(np.min(Efree_ave[:,1:])-Y_LIM_LOW_MERGINE,np.min(Efree_ave[:,1:])+Y_LIM_HIGH_MERGINE)
    min_index = np.unravel_index(np.argmin(Efree_ave[:,1:]), Efree_ave[:,1:].shape)  # calc min index in 2d arr
    low_index = min_index[1]-X_LIM_LOW_MERGINE
    high_index = min_index[1]+X_LIM_HIGH_MERGINE
    x_lim_low_index  = low_index  if low_index  > 1 else 0
    x_lim_high_index = high_index if high_index < len(temp) else -1
    plt.xlim(temp[min_index[0],x_lim_low_index], temp[min_index[0],x_lim_high_index])
    for im in range(len(Efree_ave)):
        plt.plot(temp[im,1:-1],Efree_ave[im,1:-1], c=COLOER_MAP[im],label=peak_models[im])
        # plt.errorbar(temp[im,1:-1],Efree_ave[im,1:-1],yerr=Efree_std[im,1:-1],color="k",fmt='o', markersize=5)
    plt.legend()
    plt.savefig(pic_name)
    plt.close()


    return


def plt_adpt_result(picfile_headers,pfname,model_names):
    ### copy adpt fig
    for imodel in range(len(picfile_headers)):
        olds_png = glob.glob(picfile_headers[imodel]+"_adpt_ratio*")
        olds_png = sorted(olds_png)
        new_png = pfname+"adpt_res_"+model_names[imodel]+".png"
        HORI_PNG_NUM = 4
        VERT_PNG_NUM = mt.ceil(len(olds_png)/4)
        tmp_img = Image.open(olds_png[0])
        new_width = tmp_img.width * HORI_PNG_NUM
        new_height = tmp_img.height * VERT_PNG_NUM
        new_img = Image.new('RGB', (new_width, new_height),"white")

        for ipic in range(len(olds_png)):
            h_pos = (ipic % HORI_PNG_NUM) * tmp_img.width
            v_pos = mt.floor(ipic/HORI_PNG_NUM) * tmp_img.height
            tmp = Image.open(olds_png[ipic])
            new_img.paste(tmp, (h_pos,v_pos))
        new_img.save(new_png)

    return

def plt_fine_rep_result(picfile_headers,pfname,model_names):
    ### copy fine p_log
    for imodel in range(len(picfile_headers)):
        olds_png = glob.glob(picfile_headers[imodel]+"_finerep*")
        olds_png = sorted(olds_png)
        new_png = pfname+"finerep_"+model_names[imodel]+".png"
        HORI_PNG_NUM = 4
        VERT_PNG_NUM = mt.ceil(len(olds_png)/4)
        tmp_img = Image.open(olds_png[0])
        new_width = tmp_img.width * HORI_PNG_NUM
        new_height = tmp_img.height * VERT_PNG_NUM
        new_img = Image.new('RGB', (new_width, new_height),"white")

        for ipic in range(len(olds_png)):
            h_pos = (ipic % HORI_PNG_NUM) * tmp_img.width
            v_pos = mt.floor(ipic/HORI_PNG_NUM) * tmp_img.height
            tmp = Image.open(olds_png[ipic])
            new_img.paste(tmp, (h_pos,v_pos))
        new_img.save(new_png)

    return

def plt_stepsize_result(picfile_headers,pfname,model_names):
    ### copy adpt fig
    for imodel in range(len(picfile_headers)):
        olds_png = glob.glob(picfile_headers[imodel]+"_stepsize*")
        olds_png = sorted(olds_png)
        new_png = pfname+"stepsize_"+model_names[imodel]+".png"
        HORI_PNG_NUM = 4
        VERT_PNG_NUM = mt.ceil(len(olds_png)/4)
        tmp_img = Image.open(olds_png[0])
        new_width = tmp_img.width * HORI_PNG_NUM
        new_height = tmp_img.height * VERT_PNG_NUM
        new_img = Image.new('RGB', (new_width, new_height),"white")

        for ipic in range(len(olds_png)):
            h_pos = (ipic % HORI_PNG_NUM) * tmp_img.width
            v_pos = mt.floor(ipic/HORI_PNG_NUM) * tmp_img.height
            tmp = Image.open(olds_png[ipic])
            new_img.paste(tmp, (h_pos,v_pos))
        new_img.save(new_png)

    return


def plt_exc_ratio_result(picfile_headers,pfname,model_names):
    ### copy excange ratio picfile
    olds_png = [s + "_exc_ratio.png" for s in picfile_headers]
    new_png = pfname+"/result_exc_rep.png"

    HORI_PNG_NUM = 3
    VERT_PNG_NUM = mt.ceil(len(olds_png)/3)
    tmp_img = Image.open(olds_png[0])
    new_width = tmp_img.width * HORI_PNG_NUM
    new_height = tmp_img.height * VERT_PNG_NUM

    new_img = Image.new('RGB', (new_width, new_height),"white")
    for ipic in range(len(olds_png)):
        h_pos = (ipic % HORI_PNG_NUM) * tmp_img.width
        v_pos = mt.floor(ipic/HORI_PNG_NUM) * tmp_img.height
        tmp = Image.open(olds_png[ipic])
        new_img.paste(tmp, (h_pos,v_pos))
    new_img.save(new_png)

    return



def show_all_MAP(results):
    model_num = len(results)
    HORI_PLT_NUM = 3 
    VERT_PLT_NUM = mt.ceil(model_num/HORI_PLT_NUM)

    ### plt & show
    plt.figure()
    for imod, tmp_map_dict in enumerate(results):
        plt.subplot(VERT_PLT_NUM, HORI_PLT_NUM, imod+1)
        mpar = tmp_map_dict["MAP_param"]
        X =  tmp_map_dict["rawX"]
        Y =  tmp_map_dict["rawY"]
        fdic =  tmp_map_dict["func_dict"]
        
        ### get Ys
        _peak_ = fdic["peak"]
        num_peak = len(_peak_)
        Ys = get_each_Y(X,fdic,mpar)
        Yfit = Ys["fit"]
        Ysngls = Ys["singls"]
        Ybg = Ys["BG"]
        
        plt.scatter(X,Y,color="k",s=1)
        plt.plot(X,Y,c="k")
        plt.plot(X,Ybg,c="gray")    
        plt.plot(X,Yfit,c="r")
        for ipeak in range(num_peak):
            plt.plot(X,Ysngls[ipeak]+Ybg,c=COLOER_MAP[ipeak%10])


    plt.show()
        
            
    
    return None

