import math

def gaussian_func(x, constant, mean, sigma):
    import numpy as np
    return constant * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2))

def align_digit(f, digit=None,shift=3):
    if digit==None:
        digit = int("{:e}".format(f).split("e")[1])-shift
    if digit<=0: s = str("{:."+str(-digit)+"f}").format(f)
    else: s = str(f)
    return s, digit

def fit_gaussian_impl(arr_x, arr_y, mean_value, stdev_value, show_result):

    '''
    return chi2, constant, error, mean, error, sigma, error, degree of freedom, p-value
    '''

    arr_yerror = [y**0.5 for y in arr_y]

    import numpy as np
    parameter_initial = np.array([max(arr_y), mean_value, stdev_value])
    
    arr_x2 = []
    arr_y2 = []
    arr_yerror2 = []
    for x, y, yerror in zip(arr_x, arr_y, arr_yerror): #entry=0のデータを除外
        if y == 0:continue
        arr_x2.append(x)
        arr_y2.append(y)
        arr_yerror2.append(yerror)

    if len(arr_x2) < 4:return {}

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(gaussian_func, arr_x2, arr_y2, sigma=arr_yerror2, absolute_sigma =True, p0=parameter_initial)
    stderr = np.sqrt(np.diag(pcov)) #対角行列を取って平方根

    arr_fitted_y = gaussian_func(np.array(arr_x2), popt[0], popt[1], popt[2]) 

    from scipy.stats import chisquare
    #chisq, p = chisquare(f_exp=arr_y2, f_obs=arr_fitted_y, ddof = 2)
    ndf = len(arr_x2) - 3

    mat = np.vstack((popt,stderr)).T
    if show_result:
        import pandas as pd
        df = pd.DataFrame(mat,index=("Constant", "Mean", "Sigma"), columns=("Estimate", "Std. error"))
        print(df)


    
    obj = {}
    obj["constant"]=[popt[0],stderr[0]]
    obj["mean"]    =[popt[1],stderr[1]]
    obj["sigma"]   =[popt[2],stderr[2]]
    obj["ndf"]     =ndf
    return obj

def fit_gaussian(vx, nbin, min_x, max_x,*,mean_value=None,stdev_value=None,show_plot=False,show_result=True):
    '''
    return chi2, constant, error, mean, error, sigma, error, degree of freedom, p-value
    '''
    if len(vx) == 0:return {}

    arr_x = [0 for i in range(nbin)]
    arr_y = [0 for i in range(nbin)]

    wbin = (max_x - min_x) / nbin
    if wbin <= 0:return {}

    for i in range(nbin): arr_x[i] = min_x + wbin * (i + 0.5)

    for x in vx:
        bin = math.floor((x - min_x) / wbin)
        if bin < 0:continue
        if bin >= nbin:continue
        arr_y[bin]+=1

    if mean_value == None:
        from statistics import mean
        mean_value = mean(vx)
    if stdev_value == None:
        from statistics import stdev
        stdev_value = stdev(vx)

    if show_plot:
        import matplotlib.pyplot as plt
        arr_yerror = [y**0.5 for y in arr_y]
        str_mean_value, _ = align_digit(mean_value)
        str_stdev_value, _ = align_digit(stdev_value)
        plt.errorbar(arr_x,arr_y,linestyle="None",marker="+",yerr=arr_yerror,color="tab:blue",
                    label="Data\nEntries: {0}\nMean: {1}\nStdev: {2}".format(len(vx),str_mean_value,str_stdev_value))
        print(mean_value,stdev_value)
        plt.legend(bbox_to_anchor=(1.12, 1.15), loc='upper right', borderaxespad=0)
        plt.show()

        
    return fit_gaussian_impl(arr_x, arr_y, mean_value, stdev_value, show_plot)



