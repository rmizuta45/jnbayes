from cProfile import label
import numpy as np
import math as mt
import tifffile as tif
import glob
import matplotlib.pyplot as plt
import gaus_plot_fit as gpf


fli = glob.glob("dt/raw/*.tif")
print(fli)


li = []
for fin in fli:
    tmp_img = tif.imread(fin)

    # X
    tmp_x_sum = np.sum(tmp_img,axis=0)
    x_axis = np.array(range(len(tmp_x_sum)))
    fit_x_ret = gpf.gaussian_plot_fit(x_axis,tmp_x_sum,np.max(tmp_x_sum),np.argmax(tmp_x_sum),50,BG=True)
    fit_x = gpf.gaussian_func(x_axis,fit_x_ret["constant"][0],fit_x_ret["mean"][0],fit_x_ret["sigma"][0],fit_x_ret["BG"][0])
    li.append(fit_x_ret["sigma"][0]*5)

    # # # Y
    # tmp_y_sum = np.sum(tmp_img,axis=1)
    # plt.subplot(221)
    # plt.imshow(tmp_img,cmap="gray")
    # plt.subplot(222)
    # plt.plot(x_axis,tmp_x_sum)
    # # plt.plot(x_axis,fit_x)
    # # plt.plot(x_axis,fit_x,label=fit_x_ret["label"])
    # # plt.legend()
    # plt.subplot(223)
    # plt.plot(tmp_y_sum)
    # plt.show()


np_li = np.array(li)
plt.hist(np_li,bins=50)

plt.show()



