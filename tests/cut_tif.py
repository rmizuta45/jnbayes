import numpy as np
import tifffile as tif
import glob
import matplotlib.pyplot as plt
import gaus_plot_fit as gpf


fli = glob.glob("dt/raw/*.tif")
print(fli)

fout_log = open("tests/cut_log.dat","w")
fout_log.write("# Center[pix]\n")

CUT_POS_RANGE = int(200 / 2)


li = []
for fin in fli:
    fout_tif_name = fin.replace("raw/","cut/c")
    fout_png_name = fin.replace("dt/raw/","pic/cut/")
    fout_png_name = fout_png_name.replace(".tif",".png")
    fout_np_name = fin.replace("dt/raw/","dt/np/c")
    fout_np_name = fout_np_name.replace(".tif","")
    raw_img = tif.imread(fin)
    # print("shape =",raw_img.shape)

    ##### pos axis
    raw_pos_sum = np.sum(raw_img,axis=0)
    x_axis = np.array(range(len(raw_pos_sum)))
    fit_pos_ret = gpf.gaussian_plot_fit(x_axis,raw_pos_sum,np.max(raw_pos_sum),np.argmax(raw_pos_sum),50,BG=True)
    fit_pos = gpf.gaussian_func(x_axis,fit_pos_ret["constant"][0],fit_pos_ret["mean"][0],fit_pos_ret["sigma"][0],fit_pos_ret["BG"][0])
    li.append(fit_pos_ret["sigma"][0]*3)
    cut_center_pix = int(fit_pos_ret["mean"][0])
    cut_min = cut_center_pix - CUT_POS_RANGE
    cut_max = cut_center_pix + CUT_POS_RANGE + 1
    cut_img = raw_img[:,cut_min:cut_max]


    ##### Energy axis
    raw_E_sum = np.sum(raw_img,axis=1)
    cut_E_sum = np.sum(cut_img,axis=1)

    ##### write log
    fout_log.write(f"{fin} {cut_center_pix}\n")

    ##### tiff
    # print(fout_name, "save!")
    # tif.imsave(fout_name,cut_img)


    ##### plt
    # plt.subplot(221)
    # plt.imshow(raw_img,cmap="gray")
    # plt.vlines(cut_min,0,len(raw_img))
    # plt.vlines(cut_max,0,len(raw_img))
    # plt.subplot(222)
    # plt.imshow(cut_img,cmap="gray")
    # plt.subplot(223)
    # plt.plot(raw_E_sum)
    # plt.subplot(224)
    # plt.plot(cut_E_sum)
    # plt.savefig(fout_png_name)
    # plt.close()
    # print(fout_png_name,"saved!")
    # # plt.show()

    ###### numpy
    # X = np.array(range(len(cut_E_sum)))
    # Y = cut_E_sum
    # np_dt = np.array([X,Y])
    # print(fout_np_name,"saved!")
    # np.save(fout_np_name,np_dt)




### 3sigma hist
# np_li = np.array(li)
# plt.hist(np_li,bins=50)
# plt.show()




### close logfile
fout_log.close()