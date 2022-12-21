"""
    Todo: 

        1. X(1次元numpy arr)を引数にしてY(2次元numpy arr[params][Xlen])を
        戻り値にした関数を用意する
            引数はY(2次元)とかでも可、戻り値の型が合えば何でもいい。
            ブロードキャスト推奨。

        2. func_dbに作成した関数の情報を入れる
            main key : rxmc_params.jsonから呼び出す関数名 -> str
            "parnum" : 関数のパラメータ数 -> int
            "limit"  : 関数の理論上あり得る範囲 -> dict
                "rng_max" : 関数の上限値 -> list
                "rng_min" : 関数の下限値 -> list
    
"""



import numpy as np
import math as mt
import matplotlib.pyplot as plt
np_gamma = np.frompyfunc(mt.gamma, 1, 1)

def ngaus(x,mu,sig):
    return np.array(1/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/2/sig**2),dtype=np.float32)

def smearing_gaus(x,mu,sig):
    mu = mu.reshape(-1,1)
    sig = sig.reshape(-1,1)
    return np.array(1/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/2/sig**2),dtype=np.float32)






def p_liner_bc(x,pp):
    """ Peak Function (Broadcast)

    Args:
        x (np 1d array) : Binding Energy
        pp (np 2d array) : [[A, xc, gam, alp]*rep]
            -> A    : float, intensity of peak
            -> xc   : float, position of peak ( = energy )
            -> gam  : float, width of function
            -> alp  : float, Asymmetry index (specificity index)

    Returns:
        f(x) value as np 2d array : Y value [rep][np_Y]

    Note:
    """

    a   = pp.T[0].reshape(-1,1)
    b   = pp.T[1].reshape(-1,1)
    return x*a + b




def mk_shirley_BGp(y, a, b):
    b_ret = np.zeros_like(y)
    ab = a-b
    T = y.sum()
    for i in range(1,y.shape[0]):
        b_ret[i] = ab * y[:i].sum() / T
    return b_ret + b

def makeConstBG(y,c):
    if c is list:
        tmp = c[0]
    else:
        tmp = c
        y += tmp    
    return y




# ----- [ Sirley background from data ] ------
# y : np 1d array, peak intensity data
# a : background value of higher binding energy side
# b : background value of lower bindign energy side
# [return] : np 1d array, background values
def ShirleyBG(y, a, b, loop_max = 10):
    if y.sum() < 10:
        return np.zeros_like(y)
    if y[0]-y[-1]>0:
        y = y[::-1]
    b1 = np.full_like(y, b)
    b2 = np.zeros_like(y)
    ab = a-b
    yt = y - b1
    tt = T = int(yt.sum())

    
    
    for i in range(loop_max):
        b2[0] = b
        for j in range(1,yt.shape[0]):
            b2[j] = ab * yt[:j].sum() / T + b
        b1 = b2

        yt = y - b1
        T = int(yt.sum())
        if(T == tt):
            print("get shirly ("+str(i)+" loop)")
            break
        tt = T
    if y[0]-y[-1]>0:
        b2 = b2[::-1]
    return b2

# ----- [ get Sirley background from data and a,b with energy ] -----
#  x  : np 1d array, x (energy) value of data
#  y  : np 1d array, y (intensity) value of data
# e_a : float, energy of a point (see SirleyBG function)
# e_b : float, energy of b point
# avg : int, number of points to get a and b value by average
#[return] : np 1d array, background values
def get_bg(x, y, e_a, e_b, avg):

    nd = y.shape[0]
    ia = np.abs((x - e_a)).argmin()
    ib = np.abs((x - e_b)).argmin()
    iavg_diff = int(avg // 2)

    ias = ia - iavg_diff
    if(ias < 0):
        ias = 0
    iae = ia + iavg_diff
    if(iae > nd):
        iae = nd

    ibs = ib - iavg_diff
    if(ibs < 0):
        ibs = 0
    ibe = ib + iavg_diff
    if(ibe > nd):
        ibe = nd
    
    if(iae - ias == 0):
        da = y[ias]
    else:
        da = np.average(y[ias:iae])
    if(ibe - ibs == 0):
        db = y[ibs]
    else:
        db = np.average(y[ibs:ibe])

    b_ret = ShirleyBG(y, da, db)

    return b_ret
    # ===== [ End Function * get_bg ] =====
    
    
# ----- [ get Sirley background from data and a,b with energy ] -----
#  x  : np 1d array, x (energy) value of data
#  y  : np 1d array, y (intensity) value of data
# e_a : float, energy of a point (see SirleyBG function)
# e_b : float, energy of b point
# avg : int, number of points to get a and b value by average
#[return] : np 1d array, background values
def get_liner_bg(x, y, e_a, e_b, mode):
    np_bg = np.zeros_like(y)
    
    def find_near(x_list,val):
        min_diff=1e5
        near_point = -1
        for i in range(len(x_list)):
            dif = abs(val - x_list[i])
            if dif < min_diff:
                min_diff = dif
                near_point = i
        return near_point

    low_bord  = find_near(x,e_a)
    high_bord = find_near(x,e_b)
    low_ave  = np.average(y[0:low_bord])    
    high_ave = np.average(y[high_bord:-1])  

    for i in range(len(x)):
        if i < low_bord:
            np_bg[i] = low_ave
        elif i < high_bord:
            np_bg[i] = (high_ave-low_ave)/(high_bord-low_bord)*(i-low_bord) + low_ave
        else:
            np_bg[i] = high_ave
    return np_bg    




def pDniSun(x, A, xc, alp, gam):
    return A*np_gamma(1-alp)*(np.cos((np.pi*alp/2)+(1-alp)*np.arctan((x-xc)/gam)) ) / np.sqrt(((xc-x)**2+gam**2)**(1-alp))


def pDniSunBG(x, A, xc, alp, gam, BG):
    return A*np_gamma(1-alp)*(np.cos((np.pi*alp/2)+(1-alp)*np.arctan((x-xc)/gam)) ) / np.sqrt(((xc-x)**2+gam**2)**(1-alp)) + BG




def p_lorentz_bc(x, pp)-> np.ndarray:
    """ Peak Function (Broadcast)

    Args:
        x (np 1d array) : Binding Energy
        pp (np 2d array) : [[A, xc, gam]*rep]
            -> A  : float, intensity of peak
            -> xc : float, position of peak ( = energy )
            -> gam  : float, width of function

    Returns:
        f(x) value as np 2d array : Y value [rep][np_Y]

    Note:
    """

    A = pp.T[0].reshape(-1,1)
    xc = pp.T[1].reshape(-1,1)
    gam = pp.T[2].reshape(-1,1)
    return np.array(A*gam/((x-xc)**2+gam**2),dtype=float)




def p_viogt1_bc(x, pp):
    """ Peak Function (Broadcast)

    Args:
        x (np 1d array) : Binding Energy
        pp (np 2d array) : [[A, xc, w, r]*rep]
            -> A    : float, intensity of peak
            -> xc   : float, position of peak ( = energy )
            -> w  : FWHM of peak
            -> r  : ratio of Lorentzian (0.0 - 1.0)

    Returns:
        f(x) value as np 2d array : Y value [rep][np_Y]

    Note:
    """
    A = pp.T[0].reshape(-1,1)
    xc = pp.T[1].reshape(-1,1)
    w = pp.T[2].reshape(-1,1)
    r = pp.T[3].reshape(-1,1)
    tmp_lorentz = (1/np.pi) * (w / ((x-xc)**2 + w**2))
    tmp_gaus = np.sqrt(1/(2*np.pi*w)) * np.exp(-(x-xc)**2/(2*w**2))
    return A* ( (r * tmp_lorentz) + ((1.0-r) * tmp_gaus) )
    # return ( r * (2.0 / np.pi) * (w / ( 4.0 * (x - xc)**2 + w**2) ) + (1.0 - r) * np.sqrt(4.0 * np.log(2.0) / np.pi) / w * np.exp( -4.0 * np.log(2.0) / w**2 * (x-xc)**2) ) * A



def p_dnisun_bc(x,pp):
    """ Peak Function (Broadcast)

    Args:
        x (np 1d array) : Binding Energy
        pp (np 2d array) : [[A, xc, gam, alp]*rep]
            -> A    : float, intensity of peak
            -> xc   : float, position of peak ( = energy )
            -> gam  : float, width of function
            -> alp  : float, Asymmetry index (specificity index)

    Returns:
        f(x) value as np 2d array : Y value [rep][np_Y]

    Note:
    """

    A   = pp.T[0].reshape(-1,1)
    xc  = pp.T[1].reshape(-1,1) 
    gam = pp.T[2].reshape(-1,1)
    alp = pp.T[3].reshape(-1,1)
    return np.array(A*np_gamma(1-alp)*(np.cos((np.pi*alp/2)+(1-alp)*np.arctan((xc-x)/gam)) ) / np.sqrt(((x-xc)**2+gam**2)**(1-alp)),dtype=np.float32)

def sigmoid_bc(X,pp):    
    a = pp.T[0].reshape(-1,1)
    b = pp.T[1].reshape(-1,1)
    c = pp.T[2].reshape(-1,1)
    return a * (1 + np.exp( b*X + c ))



def mk_shirley_absBG_bc(y, p):
    """BackGround Function (Broadcast)

    Args:
        Y (np 2d array) : peak intensity data [[Y]*rep]
        pp (np 2d array) : [[a,b]*rep]
            -> a : background value of higher binding energy side
            -> b : background value of lower bindign energy side

    Returns:
        f(x) value as np 2d array : BG value [rep][np_BG]

    Note:
    """
    a = p[:,0]
    b = np.array(p[:,1],dtype=float)
    base_bg = (np.tile(b,(len(y[0]),1))).T
    gap_bg = (a-b).reshape(-1,1)
    sum_y = y.sum(axis=-1).reshape(-1,1)
    csum_y =y.cumsum(axis=-1)
    sum_y = np.where(abs(sum_y)<0.01,sum_y,0.01)
    return base_bg + gap_bg * csum_y / sum_y



def mk_shirley_relBG_bc(y,p):
    """BackGround Function (Broadcast)

    Args:
        Y (np 2d array) : peak intensity data [[Y]*rep]
        pp (np 2d array) : [[a,gap]*rep]
            -> a : background value of higher binding energy side
            -> gap : background value of energy gap

    Returns:
        f(x) value as np 2d array : BG value [rep][np_BG]

    Note:
    """
    b = p[:,0]
    gap = p[:,1].reshape(-1,1)
    base_bg = (np.tile(b,(len(y[0]),1))).T
    sum_y = y.sum(axis=-1).reshape(-1,1)
    csum_y =y.cumsum(axis=-1)
    sum_y = np.where(abs(sum_y)>0.01,sum_y,0.01)
    return base_bg + gap * csum_y / sum_y

def mk_const_BG_bc(y, p):
    """ BackGround Function (Broadcast)

    Args:
        x (np 1d array) : Binding Energy
        pp (np 2d array) : [[C]*rep]
            -> C    : float, intensity of BG

    Returns:
        f(x) value as np 2d array : BG value [rep][np_BG]

    Note:
    """
    d = p.T.reshape(-1,1)
    return np.full(y.shape,d)

def mk_nothing_BG_bc(y,p):    
    """ Peak Function (Broadcast)

    Args:
        x (np 1d array) : Binding Energy
        pp (np 2d array) : [[0]*rep]

    Returns:
        ZEROs 2d array : np.full(y.shape,0)

    Note:
    """
    return np.full(y.shape,0)




def c_dnisun_gaus_bc(x,pp):
    """ Peak Function (Broadcast, Convolution)
    Args:
        x (np 1d array) : Binding Energy
        pp (np 2d array) : [[A, xc, alp, gam]*rep]
            -> A    : float, intensity of peak
            -> xc   : float, position of peak ( = energy )
            -> alp  : float, Asymmetry index (specificity index)
            -> gam  : float, width of function
            -> sig  : float, width of gaussian to convolve dnisun

    Returns:
        f(x) value as np 2d array : Y value [rep][np_Y]

    Note:
    """
    A   = pp.T[0].reshape(-1,1)
    xc  = pp.T[1].reshape(-1,1) 
    gam = pp.T[2].reshape(-1,1)
    alp = pp.T[3].reshape(-1,1)
    sig = pp.T[4].reshape(-1,1)
    cc=(x[0]+x[-1])/2
    xstep = x[1]-x[0]
    gy = ngaus(x,cc,sig) * abs(xstep)
    lx = np.arange(2*x[0]-cc,2*x[-1]-cc,xstep)
    ly = np.array(A*np_gamma(1-alp)*(np.cos((np.pi*alp/2)+(1-alp)*np.arctan((xc-lx)/gam)) ) / np.sqrt(((xc-lx)**2+gam**2)**(1-alp)),dtype=float)
    return np.array([np.convolve(ly[i],gy[i],"varid") for i in range(len(A))])
    



def c_lorentz_gaus_bc(x,pp):
    """ Peak Function (Broadcast, Convolution)
    Args:
        x (np 1d array) : Binding Energy
        pp (np 2d array) : [[A, xc, alp, gam]*rep]
            -> A    : float, intensity of peak
            -> xc   : float, position of peak ( = energy )
            -> gam  : float, width of function
            -> sig  : float, width of gaussian to convolve lorentz

    Returns:
        f(x) value as np 2d array : Y value [rep][np_Y]

    Note:
    """
    A = pp.T[0].reshape(-1,1)
    xc = pp.T[1].reshape(-1,1)
    gam = pp.T[2].reshape(-1,1)
    sig = pp.T[3].reshape(-1,1)
    cc=(x[0]+x[-1])/2
    xstep = x[1]-x[0]
    gy = ngaus(x,cc,sig) * abs(xstep)
    lx = np.arange(2*x[0]-cc,2*x[-1]-cc,xstep)
    ly = np.array(A*gam/((lx-xc)**2+gam**2),dtype=float)
    return np.array([np.convolve(ly[i],gy[i],"varid") for i in range(len(A))])






#############################
# set 1d params
def dnisun_peak(p):
    """Get peak position

    Args:
        p : numpy 1d ndarray

    Returns:
        Eb : float, peak position
    """
    return p[1]

def gaus_peak(p):
    """Get peak position

    Args:
        p : numpy 1d ndarray

    Returns:
        Eb : float, peak position
    """
    return p[1]

def voigt_peak(p):
    """Get peak position

    Args:
        p : numpy 1d ndarray

    Returns:
        Eb : float, peak position
    """
    return p[1]

def nothing(p):
    return None



func_db={
    "sigmoid":{
        "parnum":3,
        "func":sigmoid_bc,
        "peak":nothing,
        "limit":{
            "rng_min":[
                0.0,
                0.0,
                -100,
            ],
            "rng_max":[
                100,
                10,
                100,
            ]            
        }        
    },
    
    
    "liner":{
        "parnum":2,
        "func":p_liner_bc,
        "peak":nothing,
        "limit":{
            "rng_min":[
                -1000000.0,
                -1000000.0,
            ],
            "rng_max":[
                1000000.0,
                1000000.0
            ]            
        }
    },
    "voigt":{
        "parnum":4,
        "func":p_viogt1_bc,
        "peak":voigt_peak,
        "limit":{
            "rng_min":[
                0.0,
                0.0,
                0.0,
                0.0
            ],
            "rng_max":[
                10000000000.0,
                100000.0,
                100.0,
                1.0
            ]            
        }
    },
    "lorentz":{
        "parnum":3,
        "func":p_lorentz_bc,
        "peak":gaus_peak,
        "limit":{
            "rng_min":[
                0.0,
                0.0,
                0.001
            ],
            "rng_max":[
                10000000000.0,
                100000.0,
                100.0
            ]            
        }
    },
    "dnisun":{
        "parnum":4,
        "func":p_dnisun_bc,
        "peak":dnisun_peak,
        "limit":{
            "rng_min":[
                0.0,
                0.0,
                0.00001,
                0.00001
            ],
            "rng_max":[
                10000000000.0,
                100000.0,
                100,
                0.5
            ]
        }        
    },
    "c_dnisun_gaus":{
        "parnum":5,
        "func":c_dnisun_gaus_bc,
        "peak":dnisun_peak,
        "limit":{
            "rng_min":[
                0.0,
                0.0,
                0.00001,
                0.00001,
                0
            ],
            "rng_max":[
                10000000000.0,
                100000.0,
                100.0,
                0.5,
                100.0
            ]
        }        
    },
    "c_lorents_gaus":{
        "parnum":4,
        "func":c_lorentz_gaus_bc,
        "peak":voigt_peak,
        "limit":{
            "rng_min":[
                0.0,
                0.0,
                0.00001,
                0.00001
            ],
            "rng_max":[
                10000000000.0,
                100000.0,
                100.0,
                100.0,
                100.0
            ]
        }        
    },
    
    "abs_shirley":{
        "parnum":2,
        "func":mk_shirley_absBG_bc,
        "limit":{
            "rng_min":[
                0.0,
                0.0
            ],
            "rng_max":[
                100000000.0,
                100000000.0
            ]            
        }
    },
    "rel_shirley":{
        "parnum":2,
        "func":mk_shirley_relBG_bc,
        "limit":{
            "rng_min":[
                0.0,
                -50000000.0
            ],
            "rng_max":[
                100000000.0,
                50000000.0
            ]            
        }
    },
    "const":{
        "parnum":1,
        "func":mk_const_BG_bc,
        "limit":{
            "rng_min":[
                0.0
            ],
            "rng_max":[
                100000000.0
            ]            
        }
    },
    "nothing":{
        "parnum":0,
        "func":mk_nothing_BG_bc,
        "limit":{
            "rng_min":[
                0.0
            ],
            "rng_max":[
                1.0
            ]            
        }
    }
}


# def cov_voigt():
#     def gaus(x,mu,sigma):
#         return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2/(2*sigma**2))

#     X = np.linspace(-5,5,100)
#     filt_x = np.linspace(-1,1,49)
#     gaus_filter = gaus(filt_x,0,0.2)
#     gaus_filter /= sum(gaus_filter)
#     gaus_ccum  = np.cumsum(gaus_filter)[24:]
#     ratio = np.full(100,1.0)
#     ratio[:len(gaus_ccum)] = gaus_ccum
#     ratio[-len(gaus_ccum):] = gaus_ccum[::-1]
#     print(ratio)
#     Yorg = np.full(100,3)
#     Ycmb = np.convolve(Yorg,gaus_filter,"same")
#     Ynorm = Ycmb / ratio

#     plt.plot(X,Yorg,c="r")
#     plt.plot(X,Ycmb,c="g")
#     plt.plot(X,Ynorm+0.1,c="b")
#     plt.plot(X,ratio,c="k")
#     # plt.plot(filt_x,gaus_filter)

#     plt.show()
