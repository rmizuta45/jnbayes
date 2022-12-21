import matplotlib.pyplot as plt
import numpy as np


############ plt-sub ############
def plt_raw(pic_header,x,ybg,ysig,invertX=False):
    plt.figure()
    plt.scatter(x,ybg+ysig,c="c",s=1)
    plt.plot(x,ybg,c="gray")
    plt.plot(x,ybg+ysig,c="c")
    if invertX:
        plt.gca().invert_xaxis()
    plt.savefig(pic_header+"_rawData.png")
    plt.close()
    print(pic_header+"_rawData.png saved!")
    return

def plt_par_log(pic_header,irep,ipar,tmp_log):
    par_str = "par"+str(ipar).zfill(2)
    rep_str = "rep"+str(irep).zfill(3)
    pic_name = pic_header+"_p_"+par_str+"_"+rep_str+".png"
    np_nums = np.arange(len(tmp_log))
    plt.figure(figsize=(15,5))
    plt.title(rep_str+" "+par_str)
    plt.subplot(1,2,1)
    plt.scatter(np_nums,tmp_log)
    plt.ylabel(par_str)
    plt.xlabel("Iteration number")
    plt.subplot(1,2,2)
    plt.hist(tmp_log,bins=40)
    # plt.yscale("log")
    plt.xlabel(par_str)
    plt.ylabel("Cnt")
    plt.savefig(pic_name)
    plt.close()
    print(pic_name,"saved!")
    return


def plt_p_ratio(pic_header,ipar,temp,ratio):
    par_str = "par"+str(ipar).zfill(2)
    pic_name = pic_header+"_adpt_ratio_"+par_str+".png"
    plt.hlines(1.0,min(temp[1:]),max(temp[1:]),colors="gray")
    plt.hlines(0.0,min(temp[1:]),max(temp[1:]),colors="gray")
    plt.hlines(0.8,min(temp[1:]),max(temp[1:]),colors="gray",linestyle="dashed")
    plt.hlines(0.2,min(temp[1:]),max(temp[1:]),colors="gray",linestyle="dashed")
    plt.plot(temp[1:],ratio[1:])
    plt.scatter(temp[1:],ratio[1:],s=3)
    plt.title("exc. "+par_str)
    plt.xlabel("temp")
    plt.ylabel("exc. ratio")
    plt.xscale("log")
    plt.ylim(-0.1,1.1)
    plt.savefig(pic_name)
    plt.close()
    print(pic_name,"saved!")
    return

def plt_E_log(pic_header,irep,tmp_log):
    rep_str = "rep"+str(irep).zfill(3)
    pic_name = pic_header+"_E_"+rep_str+".png"
    np_nums = np.arange(len(tmp_log))
    plt.figure(figsize=(15,5))
    plt.title(rep_str)
    plt.subplot(1,2,1)
    plt.scatter(np_nums,tmp_log)
    plt.ylabel("E")
    plt.xlabel("Iteration number")
    plt.subplot(1,2,2)
    plt.hist(tmp_log,bins=40)
    # plt.yscale("log")
    plt.xlabel("E")
    plt.ylabel("Cnt")
    plt.savefig(pic_name)
    plt.close()
    print(pic_name,"saved!")
    return


def plt_exc_rep_log(pic_header,model_name,temp,sum_exc_rep,cycle,burn_in_length):
    pic_name = pic_header+"_exc_ratio.png"
    plt.figure()
    plt.hlines(1.0,min(temp[1:-1]),max(temp[1:-1]),colors="gray")
    plt.hlines(0.0,min(temp[1:-1]),max(temp[1:-1]),colors="gray")
    # plt.plot(temp[1:-1],sum_exc_rep[1:-1]/((cycle-burn_in_length)/2))
    # plt.scatter(temp[1:-1],sum_exc_rep[1:-1]/((cycle-burn_in_length)/2),s=3)
    ratio = sum_exc_rep/((cycle-burn_in_length)/2)
    plt.plot(temp[1:-1],ratio[1:-1])
    plt.scatter(temp[1:-1],ratio[1:-1],s=3)
    min_rep = np.argmin(ratio[1:-1]) + 1
    tmp_txt = "min_rep = "+str(min_rep)
    plt.text(temp[min_rep],np.min(ratio[1:-1])-0.05,tmp_txt,horizontalalignment="center",fontsize=10)
    plt.title(model_name+" exc_rep_ratio")
    plt.xlabel("temp")
    plt.ylabel("ratio temp")
    plt.xscale("log")
    plt.ylim(-0.1,1.1)
    plt.savefig(pic_name)
    plt.close()
    print(pic_name,"saved!")
    return


def plt_step_temp(pic_header,ipar,temp,stepsize):
    pic_name = pic_header+"_stepsize_par"+str(ipar).zfill(2)+".png"
    plt.figure()
    plt.plot(temp[1:-1],stepsize[1:-1,ipar])
    plt.scatter(temp[1:-1],stepsize[1:-1,ipar],s=3)
    plt.xlabel("temp")
    plt.ylabel("stepsize")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(pic_name)
    plt.close()
    print(pic_name,"saved!")





    return