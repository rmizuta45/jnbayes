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
    def __init__(self,fin_name):
        self.json_name = fin_name
        with open(fin_name) as f:
            self.df : dict = json.load(f)
        f.close()
        self.peak_num    : int =  len(self.df["fpar"]["peak"])
        self.dt_name     : str = self.df["dpar"]["dt_name"]
        self.log_dt_name : str = self.df["dpar"]["log_file"]+self.df["dpar"]["dt_name"]+"/"
        self.pic_dt_name : str = self.df["dpar"]["pic_file"]+self.df["dpar"]["dt_name"]+"/"
        self.dt_dt_name  : str = self.df["dpar"]["dt_file"] +self.df["dpar"]["dt_name"]+self.df["dpar"]["dt_extension"]
        self.model_name  : str = self.df["dpar"]["model_name"]
        self.logfile_header : str = self.log_dt_name+self.dt_name+"_"+self.model_name
        self.picfile_header : str = self.pic_dt_name+self.dt_name+"_"+self.model_name

    def register_reshape_XYslice(self,Xold:np.ndarray,Yold:np.ndarray) -> None:
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
        if Xold[-1] - Xold[0] < 0:
            self.X :np.ndarray= Xold[::-1]
            self.Y :np.ndarray= Yold[::-1]
        else:
            self.X :np.ndarray= Xold
            self.Y :np.ndarray= Yold
        try:
            xlim = self.df["fpar"]["xlim"]
            self.xslice :slice= slice(xlim[0],xlim[1])
        except:
            self.xslice :slice= slice(None,None)
        return

    def check_XYslice(self, invertX=False):
        """
        ## check X, Y with graph

        Args:
            invertX (bool, optional): _description_. Defaults to False.

        Note:
            The contents of the array do not invert even if invertX == True

        """
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


    def get_param(self,par_tree:str,par_name:str) -> np.ndarray:
        """
        ## Get param from json dict

        Args:
            par_tree (str): json key ("par", "hyper", "limit")
            par_name (str): json key ("init","step""alpha""dd","rng_min""max")

        Returns:
            np_par (ndarray(1d)): 1d parameter
        """
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
        if self.df["fpar"]["convolve"]["TF"]:
            li_par.append(self.df["fpar"]["convolve"][par_tree][par_name])
        np_par = np.array(li_par)
        self.parnum = len(np_par)
        return np_par


    def get_temp_step(self,par_num:int,step:np.ndarray,alpha:np.ndarray,dd:np.ndarray) -> tuple:
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
        num_temp = self.df["rpar"]["num_temp"]
        proportion = self.df["rpar"]["proportion"]
        sigmaR = self.df["fpar"]["noise"]["sigmaR"]
        temp = np.array([proportion**(j-num_temp+1)/sigmaR**2 for j in range(num_temp)])
        tmp_stepsize_p = []
        for i in range(par_num):
            tmp_ = np.where(temp<=1/alpha[i],step[i],step[i]/(alpha[i]*temp)**dd[i])
            tmp_[0] = step[i]
            tmp_stepsize_p.append(tmp_)
        stepsize_p= np.array(tmp_stepsize_p)
        stepsize_p = stepsize_p.T
        temp[0] = 0.0
        
        np.save(self.logfile_header+"_temp.npy",temp)
        np.save(self.logfile_header+"_stepsize.npy",stepsize_p)
        
        return temp, stepsize_p
    
    
    
    def ctrl_peram_zeros(self,old_par:np.ndarray) -> np.ndarray:
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



