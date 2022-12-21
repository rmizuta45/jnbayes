import shutil
import numpy as np
from nsbayes.libs.slct_sub import calc_prob
from nsbayes.libs.slct_sub import plt_Efree_result
from nsbayes.libs.slct_sub import plt_adpt_result
from nsbayes.libs.slct_sub import plt_fine_rep_result
from nsbayes.libs.slct_sub import plt_stepsize_result
from nsbayes.libs.slct_sub import plt_exc_ratio_result
from nsbayes.libs.slct_sub import show_all_MAP
from nsbayes.libs.etc_sub import make_directory as mk_dir
from nsbayes.libs.rxmc_cls import Rxmc_ctrl


######## M O D E L   S E L E C T ########
def model_select(rx_list:list,MAP_results:list,show_map:bool):

    ### read jsons
    model_names = []
    logfile_headers = []
    picfile_headers = []
    for i in range(len(rx_list)):
        rx_tmp: Rxmc_ctrl = rx_list[i]
        model_names.append(rx_tmp.df["dpar"]["model_name"])
        logfile_headers.append(rx_tmp.logfile_header)
        picfile_headers.append(rx_tmp.picfile_header)

    ### mkdir
    pfname = rx_tmp.pic_dt_name+"result/"
    mk_dir(pfname)

    ### read Efree
    li_Efree = []
    li_temp = []
    for tfheader in logfile_headers:
        item_E = np.load(tfheader+"_Free_Energy.npy")
        item_temp = np.load(tfheader+"_temp.npy")
        li_Efree.append(item_E)
        li_temp.append(item_temp)
    Efree = np.array(li_Efree)
    temperature = np.array(li_temp)
    del li_Efree, li_temp
    
    #### get best model
    bayes_Efree_par_rxmc_loops = rx_tmp.df["dpar"]["bayes_Efree_par_rxmc_loops"]
    best_model_num = calc_prob(Efree,model_names,bayes_Efree_par_rxmc_loops,pfname)
    plt_Efree_result(Efree,temperature,model_names,pfname)
    best_model_rxcls: Rxmc_ctrl = rx_list[best_model_num]
    best_model_name = best_model_rxcls.df["dpar"]["model_name"]

    ### copy MAP fig
    for i in range(len(picfile_headers)):
        tmp_old = picfile_headers[i]+"_MAP_par.png"
        tmp_new = pfname+"MAP_par_"+model_names[i]+".png"
        shutil.copy(tmp_old,tmp_new)

    plt_adpt_result(picfile_headers,pfname,model_names)
    plt_fine_rep_result(picfile_headers,pfname,model_names)
    plt_stepsize_result(picfile_headers,pfname,model_names)
    plt_exc_ratio_result(picfile_headers,pfname,model_names)

    ### show all MAP
    if show_map is True:
        show_all_MAP(MAP_results)


    return best_model_name