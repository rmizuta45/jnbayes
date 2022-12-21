import math

def gaussian_func(x, constant, mean, sigma, BG=0):
    import numpy as np
    return constant * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2)) + BG

def align_digit(f, digit=None,shift=3):
    if digit==None:
        digit = int("{:e}".format(f).split("e")[1])-shift
    if digit<=0: s = str("{:."+str(-digit)+"f}").format(f)
    else: s = str(f)
    return s, digit
def gaussian_plot_fit(arr_x,arr_y, constant, mean, sigma, BG=False):
    import numpy as np
    from scipy.optimize import curve_fit
    if BG:
        parameter_initial = [constant, mean, sigma, BG]
        popt, pcov = curve_fit(gaussian_func, arr_x, arr_y, absolute_sigma =True, p0=parameter_initial)
        arr_fitted_y = gaussian_func(np.array(arr_x), popt[0], popt[1], popt[2], popt[3]) 
    else:
        parameter_initial = [constant, mean, sigma]
        popt, pcov = curve_fit(gaussian_func, arr_x, arr_y, absolute_sigma =True, p0=parameter_initial)
        arr_fitted_y = gaussian_func(np.array(arr_x), popt[0], popt[1], popt[2]) 
    stderr = np.sqrt(np.diag(pcov)) #対角行列を取って平方根

    from scipy.stats import chisquare
    chisq, p = chisquare(f_exp=arr_y, f_obs=arr_fitted_y, ddof = 2)
    ndf = len(arr_x) - 3


    str_popt0, digit_popt0 = align_digit(popt[0])
    str_popt1, digit_popt1 = align_digit(popt[1])
    str_popt2, digit_popt2 = align_digit(popt[2])
    str_serr0, _ = align_digit(stderr[0], digit_popt0)
    str_serr1, _ = align_digit(stderr[1], digit_popt1)
    str_serr2, _ = align_digit(stderr[2], digit_popt2)
    label=str("Fitted\n$\\chi^2$/ndf: {0:.2f}/{1}\n"+
                "Constant: {2}$\\pm${3}\n"+
                "Mean: {4}$\\pm${5}\n"+
                "Sigma: {6}$\\pm${7}").format(chisq,ndf,
                                        str_popt0,str_serr0,
                                        str_popt1,str_serr1,
                                        str_popt2,str_serr2)


    
    obj = {}
    obj["chi2"]    = chisq
    obj["constant"]= [popt[0],stderr[0]]
    obj["mean"]    = [popt[1],stderr[1]]
    obj["sigma"]   = [popt[2],stderr[2]]
    if BG: 
        obj["BG"]  = [popt[3],stderr[3]]   
    obj["ndf"]     = ndf
    obj["pvalue"]  = p
    obj["label"]   = label
    return obj

