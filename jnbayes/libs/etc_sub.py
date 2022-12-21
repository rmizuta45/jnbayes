import os
import numpy as np
from jnbayes.libs.rxmc_cls import Rxmc_ctrl


COLOER_MAP = ["darkorange","b","g","magenta","saddlebrown","c","olive","navy","lime","teal","pink","tan","gold",]

NUM_OF_THIS_PC_MULTIPLE_CPU_CORES = 2


def make_directory(filename):
    try:
        os.mkdir(filename)
    except:
        print(filename,"exist!")
    return

def read_xy_slice_data(rxcls:Rxmc_ctrl, check=False):
    """read & register raw XY data

    Args:
        rxcls (object): rxmc class
        check (bool, optional): check with graph. Defaults to False.

    Todo:
        1. load XY data as numpy array(1d)
        2. register XY data in Rxmc_ctrl

    """
    data = np.load(rxcls.dt_dt_name)
    Xraw = data[0]
    Ys = data[1:]
    Yraw = np.sum(Ys,axis=0)
    rxcls.register_reshape_XYslice(Xraw, Yraw)
    if check:
        print(Xraw,Yraw)
        rxcls.check_XYslice(invertX=False)
    return


