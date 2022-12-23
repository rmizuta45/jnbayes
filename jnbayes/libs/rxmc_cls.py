import numpy as np
import json
import matplotlib.pyplot as plt

class Rxmc_ctrl:
    """
    # Control & Register RXMC parametr & simple func

    Args:
        fin_name (str): data json path (relative)



    Returns:
        None: None
    """
    fin_name : str
    peak_num    : int
    dt_name     : str
    log_dt_name : str
    pic_dt_name : str
    dt_dt_name  : str
    model_name  : str
    logfile_header : str
    picfile_header : str
    
    def __init__(self,fin_name):
        ## コンストラクタ(クラスを読んだときに自動でロードされる関数)
        self.json_name = fin_name
        with open(fin_name) as f:  ## jsonデータを読み込み
            self.df : dict = json.load(f)
        f.close()
        ## 各種パラメータをクラス内変数に格納
        peak_num =  len(self.df["fpar"]["peak"])
        dt_name = self.df["dpar"]["dt_name"]
        log_dt_name = self.df["dpar"]["log_file"]+self.df["dpar"]["dt_name"]+"/"
        pic_dt_name = self.df["dpar"]["pic_file"]+self.df["dpar"]["dt_name"]+"/"
        dt_dt_name = self.df["dpar"]["dt_file"] +self.df["dpar"]["dt_name"]+self.df["dpar"]["dt_extension"]
        model_name = self.df["dpar"]["model_name"]
        logfile_header = log_dt_name + dt_name + "_" + model_name
        picfile_header = pic_dt_name + dt_name + "_" + model_name

        self.peak_num = peak_num
        self.dt_name = dt_name
        self.log_dt_name = log_dt_name
        self.pic_dt_name = pic_dt_name
        self.dt_dt_name = dt_dt_name
        self.model_name = model_name
        self.logfile_header = logfile_header
        self.picfile_header = picfile_header    
        

    def register_reshape_XYslice(self,Xold:np.ndarray,Yold:np.ndarray):
        """ 
        ## Registr X, Y data

        Contents:

            1. Jdge order of array 
            2. Read slice object for Xrange in json
            3. Registr newX and newY

        Args:
            Xraw (np.ndarray): np.array (1d)[Xlen]
            Yraw (np.ndarray): np.array (1d)[Ylen]
        """
        
        ## 型ヒントを与える
        ## 複数回型ヒントは与えられないので条件分岐するときはこのように書く
        self.X :np.ndarray
        self.Y :np.ndarray
        self.xslice :slice
        
        ## Xの配列が小さい順になっているか確認
        if Xold[-1] - Xold[0] < 0:
            self.X  = Xold[::-1]
            self.Y  = Yold[::-1]
        else:
            self.X = Xold
            self.Y = Yold
        ## jsonでデータ制限(xlim)が設定されていたらそれを反映
        try:
            xlim = self.df["fpar"]["xlim"]
            self.xslice = slice(xlim[0],xlim[1])
        except:
            self.xslice = slice(None,None)

    def check_XYslice(self, invertX=False):
        """
        ## check X, Y with graph

        Args:
            invertX (bool, optional): _description_. Defaults to False.

        Note:
            The contents of the array do not invert even if invertX == True

        """
        ### XYデータを確認できる
        plt.plot(self.X,self.Y)
        plt.plot(self.X[self.xslice],self.Y[self.xslice])
        if invertX:
            plt.gca().invert_xaxis()
        plt.show()
        return
    
        
    def json_dump(self,fout_name=None):
        if fout_name == None:
            fout_name = self.json_name
        with open(fout_name, 'w') as f:
            json.dump(self.df, f, indent=4)
        f.close()


    def get_param(self,par_tree:str,par_name:str):
        """
        ## Get param from json dict

        Args:
            par_tree (str): json key ("par", "hyper", "limit")
            par_name (str): json key ("init","step""alpha""dd","rng_min""max")

        Returns:
            np_par (ndarray(1d)): 1d parameter
        """
        ### 呼び出したjsonのデータをもとにnumpyでデータを返す
        ### 引数の辞書式keyを参照してデータの探索をする
        ### ここではパラメータのみに対して行う
        
        li_par = []
        # read peak par
        _peak_ = self.df["fpar"]["peak"]
        for p_num in _peak_:
            li_par.extend(_peak_[p_num][par_tree][par_name])

        # read BG par (w/ model)
        BG_model = self.df["fpar"]["BG"]["model"]
        if BG_model != "nothing":
            li_par.extend(self.df["fpar"]["BG"][par_tree][par_name])

        # read conv par
        ### 特殊な関数のみ使用、ふつうは使わない
        if self.df["fpar"]["convolve"]["TF"]:
            li_par.append(self.df["fpar"]["convolve"][par_tree][par_name])
        np_par = np.array(li_par)
        self.parnum = len(np_par)
        return np_par


    def get_temp_step(self,par_num:int,step:np.ndarray,alpha:np.ndarray,dd:np.ndarray):
        """
        ## Get 'reverse temperature' & 'step size'

        Args:
            par_num (int): num of parameter
            step (np.ndarray): np array (2d), hyper parameter
            alpha (np.ndarray): np array (2d), hyper parameter
            dd (np.ndarray): np array (2d), hyper parameter

        Returns:
            temp (np.ndarray): np.array (1d), reverse temprature
            stepsize_p (np.ndarray): np.array (2d(1d)) [0][nRep], range of parameter
        """
        
        #### 逆温度とRXMCの乱数振れ幅(ステップサイズ)を生成
        num_temp = self.df["rpar"]["num_temp"]
        proportion = self.df["rpar"]["proportion"]
        sigmaR = self.df["fpar"]["noise"]["sigmaR"]
        
        ### 逆温度の生成、等比数列
        temp = np.array([proportion**(j-num_temp+1)/sigmaR**2 for j in range(num_temp)])
        
        ### ステップサイズの生成
        tmp_stepsize_p = []
        ## 各パラメータ毎にステップサイズを生成
        for i in range(par_num):
            tmp_ = np.where(temp<=1/alpha[i],step[i],step[i]/(alpha[i]*temp)**dd[i])
            ## 必ず0番目のステップサイズは全領域探索にする
            tmp_[0] = step[i]
            tmp_stepsize_p.append(tmp_)
        stepsize_p= np.array(tmp_stepsize_p)
        stepsize_p = stepsize_p.T
        ## 必ず0番目の逆温度は0にしておく
        temp[0] = 0.0
        
        np.save(self.logfile_header+"_temp.npy",temp)
        np.save(self.logfile_header+"_stepsize.npy",stepsize_p)
        
        return temp, stepsize_p
    
    
    
    def ctrl_peram_zeros(self,old_par:np.ndarray):
        """
        ## Set all elements to 0

        For:
            Hyper dd param for calib para

        Args:
            old_par (np.ndarray): any shape

        Returns:
            np.ndarray: Shape is same as old par 
        """
        new_par = np.zeros(old_par.shape,dtype=float)
        del old_par
        return new_par


    # def diff_shirley_BG(self):
    #     ab = self.df["fpar"]["BG"]["pre_shirley"]
    #     a = ab[0]
    #     b = ab[1]
    #     Yraw = self.Y
    #     Ybg = rf.ShirleyBG(Yraw,a,b)
    #     self.Y = Yraw - Ybg
    #     return 



