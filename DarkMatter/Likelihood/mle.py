import numpy as np

import matplotlib.pyplot as plt

import os

import ctypes

from array import array

from ROOT import TFile, TH1D, TH2D

from ROOT import TMinuit, Math

from .signal import calcSignal

from .fcn import fcn, mfcn, binnedfcn, simplefcn, fcn_bkg, getLikelihood

from . import vegas, eventdisplay

from .check import convert2Dsignal

from ..utils import getArray, plot2D, convertToPDF, defineTheta2Cut, listOfVersions

from ..const import OUTPUT_DIR

from .. import ResponseFunction

from scipy.stats import poisson

from scipy.interpolate import interp1d

from astropy.table import Table

from ..ResponseFunction.eventdisplay import th2cut_ext

class MLE:

    def __init__(self, dwarf, M, package="EventDisplay", channel="tt", irf = None, jProfile = None, jArray=True,
                 addTheta = False, averagedIRF = True, version="all", 
                 th2Cut = 0, ext=False, eLowerCut = None, eUpperCut=None, 
                 bkgModel=None, seed=0, jSeed=-1, verbose=True, events = [], filter_he=False,
                 ideal=False, test=False, expectedLimit=False, tau=[1], 
                 statistic="unbinned", DM_spectra="PPPC", **kwargs):

        if package=="":
            print("[Error] The package is not specified. You need to set package='VEGAS' or package='EventDisplay'")
            return

        if ext and (th2Cut == 0):
            th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
        else:
            th2Cut = defineTheta2Cut(package, th2Cut)

        self._verbose=verbose
        self._package=package

        if package == "VEGAS" and addTheta:
            print("[Error] The VEGAS package does not allow to perform the 2D analysis.")
            return

        self._channel = channel

        self._signu0 = kwargs.pop("signu", -23)

        self._dwarf = dwarf
        self._M = M
        
        self._th2Cut = th2Cut
        self._eLowerCut = eLowerCut
        self._eUpperCut = eUpperCut
        self._addTheta = addTheta
        if addTheta == 2:
            self._semi = True
        else:
            self._semi = False
        self._averagedIRF = averagedIRF
        self._version=version
        self._test = test
        self._expectedLimit = expectedLimit
        self._singleIRF = bool((self.averagedIRF) + (self.version != "all"))
        self._listOfVersions = listOfVersions(self.dwarf)
        self._normDisp = kwargs.pop("normDisp", False)
        self._useBias = kwargs.pop("useBias", True)
        self._args = {}
        self._jSeed = jSeed
        self._statistic = statistic

        self._core = kwargs.pop("core", False)
        
        self._ext = ext

        if verbose: print("[Log] Initializing... ({})".format(package))
        if package=="EventDisplay":
            if seed ==0: seed = 100
            else: seed *= 100
            self._pn = int(seed+np.random.rand(1)*100)
        elif package=="VEGAS":
            if seed ==0: seed = 200
            else: seed *= 100
            self._pn = int(seed+np.random.rand(1)*100)

        # Loading IRF
        if verbose: print("[Log] Step 1: Importing instrument response functions...", end="\r")
        if irf is None:
            if package=="EventDisplay":
                if self.singleIRF:
                    try:
                        irf = ResponseFunction.EventDisplay.readIRFs(self.dwarf, version=self.version, norm=self.normDisp, ext=self.ext)
                        err = ResponseFunction.checkIRF(irf)
                    except:
                        err = True

                    if err:
                        irf = ResponseFunction.EventDisplay.averagedIRFs(self.dwarf, version=self.version, export=False, norm=self.normDisp, ext=self.ext)
                    tau = [1]
                else:
                    irf = {}
                    tau = []
                    for v in self.listOfVersions:
                        try:
                            irf[v] = ResponseFunction.EventDisplay.readIRFs(dwarf, version=v,  ext=ext)
                            err = ResponseFunction.checkIRF(irf[v])
                        except:
                            
                            err = True    
                        if err:
                            irf[v] = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=v, export=False, verbose=(verbosity>1), ext=ext)
                        tau.append(irf[v].exposure)
                    tau = np.asarray(tau)/sum(tau)

            elif package=="VEGAS":
                try:
                    irf = ResponseFunction.VEGAS.readIRFs(self.dwarf, verbose=False)
                except:
                    irf = ResponseFunction.VEGAS(self.dwarf, mode=0, verbose=False)
        
        self.irf = irf
        self.tau = tau

        if verbose: print("[Log] Step 1: Instrument response functions are imported.")
        
        # Loading Events
        if verbose: print("[Log] Step 2: Loading events...", end="\r")    
        
        if filter_he:
            self._eUpperCut = 1e4

        if package=="EventDisplay":
            self.hOn, self.hOff, self.N_on, self.N_off, raw_events, self.alpha = eventdisplay.readData(self.dwarf, 
                addTheta=self.addTheta, 
                full_output=True, bkgModel=bkgModel, th2Cut=self.th2Cut, 
                eLowerCut=self.eLowerCut, eUpperCut=self.eUpperCut, 
                version=self.version, ext=self.ext, **kwargs)
            self.events = raw_events[raw_events[:,2]==1.0]
            self.bkgs = raw_events[raw_events[:,2]==0.0]
            self._eLowerCut = min(self.events[:,0])

        elif package=="VEGAS":
            self._addTheta = False
            self.hOn, self.hOff, self.N_on, self.N_off, raw_events, self.alpha = vegas.readData(self.dwarf, full_output=True, **kwargs)
            self.events = raw_events[raw_events[:,2]==1.0]
            self.bkgs = raw_events[raw_events[:,2]==0.0]
            self._eLowerCut = min(self.events[:,0])

        if self.test or self.expectedLimit:
            if verbose: print("[Log] Step 2: Loading events from inputs")
            self.events = events
            self.events = events[events[:,2]==1.0]
            self.N_on = kwargs.pop("N_on", self.N_on)
            self.N_off = kwargs.pop("N_off", self.N_off)
            self.hOn = kwargs.pop("hOn", self.hOn)
            self.hOff = kwargs.pop("hOff", self.hOff)
            self.alpha = kwargs.pop("alpha", self.alpha)
        
        if not(self.singleIRF) and (package=="EventDisplay"):
            self.hOff = {}
            self.N_off_v = {}
            for v in self.listOfVersions:
                hOn, hOff, N_on, N_off, events, alpha = eventdisplay.readData(self.dwarf, addTheta=self.addTheta, full_output=True, bkgModel=bkgModel, th2Cut=self.th2Cut, eLowerCut=self.eLowerCut, eUpperCut=self.eUpperCut, version=v, ext=self.ext, **kwargs)
                self.hOff[v] = hOff
                self.N_off_v[v] = N_off
        

        if verbose: print("[Log] Step 2: Events are loaded.")
        
        # Loading a model
        if verbose: print("[Log] Step 3: Loading a source model...", end="\r")
        hSignal = kwargs.get("hSignal", None)
        if hSignal is None:
            if self.channel == "gamma" or self.channel == "delta" :
                if self.M > 1e5:
                    self._signu0 = -21
                else:
                    self._signu0 = -25
            elif self.channel == "wino" or self.channel == "quintuplet":
                self._signu0 = -24


            if package=="EventDisplay":
                if self.singleIRF:
                    self.hSignal = calcSignal(self.dwarf, self.M, self.irf, jProfile = jProfile, package=self.package, 
                        channel=self.channel, addTheta=self.addTheta, sigma=self.signu0, ideal=ideal, 
                        th2Cut=self.th2Cut, eLowerCut=self.eLowerCut,  version=self.version, 
                        normDisp=self.normDisp, useBias=self.useBias, jArray=jArray, 
                        jSeed=self.jSeed, ext=self.ext, DM_spectra=DM_spectra, **kwargs)
                    self.hSignal.SetDirectory(0)
                else:
                    self.hSignal = {}
                    if jProfile == None:
                        jProfile = {}
                    for v in self.listOfVersions:
                        if v not in jProfile.keys():
                            jProfile[v] = None
                        
                        if jArray:
                            self.hSignal[v] = calcSignal(self.dwarf, self.M, self.irf[v], jProfile = jProfile[v], package=self.package, channel=self.channel, addTheta=self.addTheta, sigma=self.signu0, ideal=ideal, th2Cut=self.th2Cut, eLowerCut=self.eLowerCut, version=v, normDisp=self.normDisp, jArray=jArray, jSeed=self.jSeed, ext=self.ext, DM_spectra=DM_spectra)
                        else:
                            self.hSignal[v] = calcSignal(self.dwarf, self.M, self.irf[v], jProfile = jProfile, package=self.package, channel=self.channel, addTheta=self.addTheta, sigma=self.signu0, ideal=ideal, th2Cut=self.th2Cut, eLowerCut=self.eLowerCut, version=v, normDisp=self.normDisp, jArray=jArray, jSeed=self.jSeed, ext=self.ext, DM_spectra=DM_spectra)
                        self.hSignal[v].SetDirectory(0)
                        
            elif package=="VEGAS":
                self.hSignal = calcSignal(self.dwarf, self.M, self.irf, jProfile = jProfile, package=self.package, channel=self.channel, addTheta=self.addTheta, sigma=self.signu0, ideal=ideal, th2Cut=self.th2Cut, eLowerCut=self.eLowerCut, version=self.version, jArray=jArray, jSeed=self.jSeed, useBias=False, DM_spectra=DM_spectra)
                self.hSignal.SetDirectory(0)
        else:
            self.hSignal = hSignal
            if self.singleIRF:
                if type(self.hSignal) == TH1D or type(self.hSignal) == TH2D:
                    self.hSignal.SetDirectory(0)
            else:
                for v in self.listOfVersions:
                    self.hSignal[v].SetDirectory(0)

        if verbose: print("[Log] Step 3: A source model is loaded")

        # Assign probability
        if verbose:
            print("[Log] Step 4: Assigning probabilities for each event...", end="\r")
        
        self.updateArgs(**kwargs)
        if verbose: print("[Log] Step 4: A temporary file for minimization is generated.")

        if verbose: print("[Log] Initialization is done. Run minuite.")
        
    @property
    def verbose(self):
        return self._verbose

    @property
    def statistic(self):
        return self._statistic
    
    @property
    def jSeed(self):
        return self._jSeed

    @property
    def package(self):
        return self._package
    
    @property
    def signu0(self):
        return self._signu0

    @property
    def dwarf(self):
        return self._dwarf

    @property
    def M(self):
        return self._M

    @property
    def channel(self):
        return self._channel

    @property
    def th2Cut(self):
        return self._th2Cut

    @property
    def eLowerCut(self):
        return self._eLowerCut

    @property
    def eUpperCut(self):
        return self._eUpperCut
    
    @property
    def eLowerCut(self):
        return self._eLowerCut

    @property
    def addTheta(self):
        return self._addTheta

    @property
    def semi(self):
        return self._semi

    @property
    def averagedIRF(self):
        return self._averagedIRF

    @property
    def expectedLimit(self):
        return self._expectedLimit

    @property
    def normDisp(self):
        return self._normDisp

    @property
    def useBias(self):
        return self._useBias

    @property
    def test(self):
        return self._test
    
    @property
    def version(self):
        return self._version

    @property
    def singleIRF(self):
        return self._singleIRF

    @property
    def args(self):
        return self._args

    @property
    def listOfVersions(self):
        return self._listOfVersions

    @property
    def ext(self):
        return self._ext

    @property
    def core(self):
        return self._core
    
    @property
    def likelihood(self):
        if hasattr(self, "_likelihood"):
            return self._likelihood
        else:
            gLb, gLSignu = self.__scanProfile__()
            x_signu, y_signu = getArray(gLSignu)
            return np.asarray([10**(x_signu+self.signu0), y_signu-self.logl0]).T
    
    def __signalProb__(self):

        if self.singleIRF:
            self._prob_signal = convertToPDF(self.hSignal, norm=True)

            self._prob_signal.SetDirectory(0)
            hg = convertToPDF(self.hSignal, norm=False)
            
            if self.addTheta:
                #z, x, y = getArray(self.hSignal)
                #i = len(x[x<min(self.events[:,0])])
                self.g = [hg.Integral(1, -1, 1, -1, "width")]
                #self.g0 = hg.Integral(i, hg.GetNbinsX(), 1, hg.GetNbinsY(), "width")
            else:
                self.g = [hg.Integral(1, -1, "width")]
        else:
            self._prob_signal = {}
            self.g = []
            
            for v in self.listOfVersions:
                self._prob_signal[v] = convertToPDF(self.hSignal[v], norm=True)
                self._prob_signal[v].SetDirectory(0)
                hg = convertToPDF(self.hSignal[v], norm=False)
                if self.addTheta:
                    g = hg.Integral(1, -1, 1, -1, "width")
                    self.g.append(g)
                else:
                    g = hg.Integral(1, -1, "width")
                    self.g.append(g)

        p_on = []
        
        for evt in self.events:

            if self.singleIRF:
                if self.addTheta:
                    val = self._prob_signal.Interpolate(evt[0], evt[1])
                    if val < 0:
                        val = 0
                else:
                    if self.core:
                        e, v = getArray(self.hSignal)
                        filt = v>0
                        minE = min(e[filt])
                        maxE = max(e[filt])
                        if (evt[0] < minE) or (evt[0]> maxE):
                            continue

                    val = self._prob_signal.Interpolate(evt[0])
                    
                    if val < 0:
                        val = 0
                p_on.append(val)
    
            else:
                evt_version = "v"+str(int(evt[4]))
                if self.addTheta:
                    val = self._prob_signal[evt_version].Interpolate(evt[0], evt[1])
                    if val < 0:
                        val = 0
                else:
                    val = self._prob_signal[evt_version].Interpolate(evt[0])
                    if val < 0:
                        val = 0
                p_on.append([val, evt_version])
            
        if self.singleIRF:
            p_on = [np.asarray(p_on)]
        else:
            p_on = np.asarray(p_on)
            p_on_t = []
            for v in self.listOfVersions:
                p_on_t.append(p_on[:,0][p_on[:,1] == v].astype("float"))
            p_on = p_on_t
        
        self.p_on = np.asarray(p_on)

    def __bgProb__(self, **kwargs):

        self._prob_bg = kwargs.get("hOff", None)
        
        if self._prob_bg == None:
            if self.singleIRF:
                self._prob_bg = convertToPDF(self.hOff, norm=True, apply_gp=kwargs.pop("apply_gp", False))
                self._prob_bg.SetDirectory(0)
            else:
                self._prob_bg = {}
                for v in self.listOfVersions:
                    self._prob_bg[v] = convertToPDF(self.hOff[v], norm=True, apply_gp=kwargs.pop("apply_gp", False))
                    self._prob_bg[v].SetDirectory(0)
        

        p_off = []
        p_off_err = []

        if self.core:
            e, v = getArray(self.hSignal)
            filt = v>0
            
            self.N_off = sum(getArray(self.hOff)[1])/self.alpha
            self.N_on = sum(getArray(self.hOn)[1])

        for evt in self.events:
            if self.singleIRF:
                if self.addTheta:
                    val = self._prob_bg.Interpolate(evt[0], evt[1])
                    if val <= 0:
                        val = 0
                        p_off_err.append([evt[0], evt[1]])

                else:    
                    if self.core:
                        e, v = getArray(self.hSignal)
                        filt = v>0
                        minE = min(e[filt])
                        maxE = max(e[filt])
                        if (evt[0] < minE) or (evt[0]> maxE):
                            continue

                    val = self._prob_bg.Interpolate(evt[0])
                    if val <= 0:
                        val = 0
                        p_off_err.append([evt[0]])

                p_off.append(float(val))
                
            else:
                evt_version = "v"+str(int(evt[4]))
                if self.addTheta:
                    val = self._prob_bg[evt_version].Interpolate(evt[0], evt[1])
                    if val <= 0:
                        val = 0
                        p_off_err.append([evt[0], evt[1]])
                else:
                    val = self._prob_bg[evt_version].Interpolate(evt[0])
                    if val <= 0:
                        val = 0 
                        p_off_err.append([evt[0]])

                p_off.append([val, evt_version])

        if kwargs.pop("correction", False):
            p_off = np.asarray(p_off)
            if len(p_off[p_off!=0]) == 0:
                print("[Error] There is an issue in calculating the p_off.")
                minp = 0
            else:
                minp = min(p_off[p_off!=0])

            p_off[p_off==0] = minp
            self._min_p = minp

        if self.singleIRF:
            p_off = [np.asarray(p_off)]
        else:
            p_off = np.asarray(p_off)
            p_off_t = []
            for v in self.listOfVersions:
                p_off_t.append(p_off[:,0][p_off[:,1] == v].astype("float"))
            p_off = p_off_t
        
        self.p_off = np.asarray(p_off)
        self.p_off_err = np.asarray(p_off_err)

    
    def __updateOnRegion__(self, **kwargs):
        if kwargs.get("hOn", None) is None:
            hOn_new = self.hOn.Clone()
            hOn_new.Reset()
            for evt in self.events:
                if self.addTheta:
                    hOn_new.Fill(evt[0], evt[1])
                else:
                    hOn_new.Fill(evt[0])
            if len(self.events) != 0:
                if self.N_on != len(self.events):
                    print("[Warning] The number of events and the pre-defined N_on are different.")
                self.N_on = len(self.events)
            self.hOn = hOn_new
            self.hOn.SetDirectory(0)

    def __manualScan__(self):
        sig_min = max((self.signu[0]-self.signu0), -1)
        signu = np.linspace(-2, +5, 1000)
        gLSignu = np.asarray([getLikelihood(sig, self.b[0], seed=self._pn) for sig in signu])
        
        return signu, gLSignu


    def __scanProfile__(self, skip_b=False, skip_sig=False):

        istat = ctypes.c_int(12)
        ierflg = ctypes.c_int(7)
        sig_min = max((self.signu[0]-self.signu0), -1)
        
        self.fit.mnparm(0,"b", self.b[0],     0.01,  self.b[0]-3*self.b[1],    self.b[0]+3*self.b[1],  ierflg)
        self.fit.mnparm(1,"signu", sig_min,     0.01,    sig_min-5,     sig_min+10,  ierflg)

        if not(skip_b):
            self.fit.mncomd("scan 1 100",istat)
            gLb = self.fit.GetPlot()
        else:
            gLb = None

        if not(skip_sig):
            self.fit.mncomd("scan 2 100",istat)
            gLSignu = self.fit.GetPlot()
        else:
            gLSignu = None

        return gLb, gLSignu


    def nullHypothesis(self, b_null, **kwargs):
        if self.statistic == "unbinned":
            P =[]
            for p, tau in zip(self.p_off, self.tau):               
                P += list(self.alpha*b_null*p*tau)
            P = np.asarray(P)
            P = P[P>0]
            logl0 = self.N_off*np.log(b_null) - (self.alpha+1)*b_null+sum(np.log(P))
            
        elif self.statistic == "binned":
            alpha_arr = kwargs.get("alpha_array", None)
            b_arr = self.args["hOff"]/sum(self.args["hOff"])*b_null
            valid = (b_arr!=0)
            b_arr = b_arr[valid]
            if alpha_arr is None:
                logl0 = sum(self.args["hOff"][valid]*np.log(b_arr)) - (self.args["alpha"]+1)*b_null + sum(self.args["hOn"][valid]*np.log((self.args["alpha"]*b_arr)))
            else:
                alpha_arr = np.asarray(alpha_arr)
                logl0 = sum(self.args["hOff"][valid]*np.log(b_arr)) - sum((alpha_arr[valid]+1)*b_arr) + sum(self.args["hOn"][valid]*np.log((alpha_arr[valid]*b_arr)))
        elif self.statistic == "simple":
            logl0 = self.args["N_off"]*np.log(b_null) - (self.args["alpha"]+1)*b_null + self.args["N_on"]*np.log(self.args["alpha"]*b_null)
        return -logl0

    def updateArgs(self, verbose=False, forced=False, **kwargs):
        if (self.test or self.expectedLimit or forced):
            self.__updateOnRegion__(**kwargs)

        if self.statistic == "unbinned":
            self.__signalProb__()
            self.__bgProb__(**kwargs)

            err_evt = len(self.p_off_err)

            if self.addTheta:
                tab = Table(self.p_off_err, names=["Energy", "Theta2"])
            else:
                tab = Table(self.p_off_err, names=["Energy"])
            if err_evt > 0:
                if self.verbose:
                    print("\n")
                    if kwargs.pop("correction", False):
                        print("[Warning] {:.0f} events have p_bkg of 0. Set a minimum probability of 10^{:.3f}.".format(err_evt, np.log10(self._min_p)))
                    else:
                        print("[Warning] {:.0f} events have p_bkg of 0. They are ignored when calculating the likelihood.".format(err_evt))
                if self.verbose:
                    print(tab)
                    print("\n")

        else:
            self.p_on = None
            self.p_off = None
            self.g = [sum(getArray(self.hSignal)[1])]
        
        self.args["dwarf"]  = self.dwarf
        self.args["events"] = self.events
        self.args["alpha"]  = self.alpha
        self.args["alpha_array"]  = kwargs.get("alpha_array", None)
        self.args["jSeed"]  = self.jSeed
        self.args["mass"]   = self.M
        self.args["N_on"]   = self.N_on
        self.args["N_off"]  = self.N_off
        self.args["g"]      = self.g
        #self.args["g0"]     = self.g0
        self.args["p_on"]   = self.p_on
        self.args["p_off"]  = self.p_off
        self.args["tau"]    = self.tau
        self.args["exposure"]    = self.irf.exposure
        self.args["package"] = self.package
        self.args["energies"] = getArray(self.hOn)[0]
        
        if self.singleIRF:
            if self.addTheta:

                self.args["hOn"]    = getArray(self.hOn)[0].flatten()
                self.args["hSig"]   = getArray(self.hSignal)[0].flatten()
                if kwargs.get("alpha_corrected", True):
                    self.args["hOff"]   = getArray(self.hOff)[0].flatten()/self.alpha
                else:
                    self.args["hOff"]   = getArray(self.hOff)[0].flatten()
            elif self.core:
                e, v = getArray(self.hSignal)
                filt = v>0
                    
                self.args["hOn"]    = getArray(self.hOn)[1][filt]
                self.args["hSig"]   = getArray(self.hSignal)[1][filt]
                if kwargs.get("alpha_corrected", True):
                    self.args["hOff"]   = getArray(self.hOff)[1][filt]/self.alpha
                else:
                    self.args["hOff"]   = getArray(self.hOff)[1][filt]

            else:
                self.args["hOn"]    = getArray(self.hOn)[1]
                self.args["hSig"]   = getArray(self.hSignal)[1]
                if kwargs.get("alpha_corrected", True):
                    self.args["hOff"]   = getArray(self.hOff)[1]/self.alpha
                else:
                    self.args["hOff"]   = getArray(self.hOff)[1]
        
        np.save(OUTPUT_DIR+"/__temp__/temp_args_{}".format(self._pn), self.args)

        if self.verbose and verbose:
            print("[Log] Arguments are successfully updated.")
    
    def minuit(self, fix_b = False, fix_b_value=None, upperLimit=True, method = 1, verbose=None, **kwargs):            

        if verbose is None:
            verbose=self.verbose
            printLevel = -1
        elif verbose == 3:
            printLevel = 1
        else:
            printLevel = -1
        
        Math.MinimizerOptions.SetDefaultMinimizer("Minuit2");

        fit = TMinuit(3)
        
        fit.SetPrintLevel(printLevel=printLevel)
        
        # Parameter setting
        fit.mnrset(1)
        
        ierflg = ctypes.c_int(199)
        istat = ctypes.c_int(12)

        if fix_b_value==None:
            self.b_null = (self.N_on + self.N_off)/(1.0 + self.alpha)
        else:
            self.b_null = fix_b_value

        arglist = array( 'd', 10*[0.] )
        arglist[0] = 10000
        arglist[1] = 1.

        # Import a model
        if self.statistic == "unbinned":
            fit.SetFCN(fcn)
        elif self.statistic == "binned":
            fit.SetFCN(binnedfcn)
            fix_b = True
        elif self.statistic == "simple":
            fit.SetFCN(simplefcn)
        
        fit.mnparm(0,"b    ", self.b_null,   0.1,   self.b_null*0.7,   max(1e7, self.b_null*1.5),  ierflg)
        fit.mnparm(1,"signu", 1,     0.01,    -20,     5,  ierflg)

        # This is for running multiple MLE.
        fit.mnparm(2,"package", self._pn,     0,    self._pn,     self._pn,  ierflg)
        fit.FixParameter(2)

        if fix_b:
            fit.FixParameter(0)

        # Initial minimization
        fit.SetErrorDef(1)
        fit.mnexcm("MIGRAD", arglist, 2, ierflg);
        fit.mnmnos()
        
        if ierflg.value == 0:
            if verbose==2:
                print("[Log] MINUIT finds a minimum successfully (MIGRAD is converged).")

        elif ierflg.value == 4:
            for i in range(5):
                if verbose ==2:
                    print("[Warning] MIGRAD is NOT converged. Try again (trials: {}).".format(i+1), end="\r")
                fit.mnparm(1,"signu", (np.random.rand(1)[0]-0.5)*2,     0.1,    -20,     3,  ierflg)
                fit.mnexcm("MIGRAD", arglist, 2, ierflg);
                
                if ierflg.value == 0:
                    break
            
            if ierflg.value == 4:
                print("[Error] MIGRAD is NOT converged. Check initial parameters (minuit in mle.py).")
                
            else:
                if verbose==2:
                    print("[Log] MINUIT finds a minimum successfully (MIGRAD is converged).")

        else:
            print("[Error] An error occurs (type={}, https://root.cern.ch/doc/master/classTMinuit.html#ab48dd4b48800edc090396e35cb465fb9)".format(int(ierflg.value )))
        
        fit.mnimpr()
        
        logl, edm, errdef = map(ctypes.c_double, (0.18, 0.19, 0.20))
        nvpar, nparx, icstat = map(ctypes.c_int, (1983, 1984, 1985))
        b_min, b_err = map(ctypes.c_double, (0.50, 0.51))
        signu_min, signu_err = map(ctypes.c_double, (0.40, 0.41))
        
        fit.mnstat(logl, edm, errdef, nvpar, nparx, icstat )        
        fit.GetParameter(0, b_min, b_err)
        fit.GetParameter(1, signu_min, signu_err)
        
        self.b = (float(b_min.value), float(b_err.value))
        self.signu = (float(signu_min.value)+self.signu0, float(signu_err.value))
        self.flag = ierflg.value
        self.fit = fit

        self.logl0 = self.nullHypothesis(self.b_null, **kwargs)
        self.logl2 = self.nullHypothesis(self.b[0], **kwargs)
        self.logl = logl.value
        
        self.ts = 2*(self.logl0-self.logl)
        
        signu_ul = np.nan
        self.gL = None
        # For calculating an upper limit
        if upperLimit:
            fit.SetErrorDef(2.70/2)
            fit.mnexcm("MIGRAD", arglist, 2, ierflg);
            if verbose==2:
                print("[Log] Since TS <25 and upperLimit=True, an upper limit (95%) will be estimated.")
            
            # Estimate an upper limit with contour
            if (method == 1 or method == 3) and not(fix_b):
                ctr = fit.Contour(40, 0, 1)
                
                if ctr != None:
                    x_95, y_95 = getArray(ctr)
                    signu_ul = max(y_95+self.signu0)
                    if signu_ul >= -19.0:
                        signu_ul = np.nan
                    if method == 3:
                        signu_ul_ctr = signu_ul
                else:
                    signu_ul_ctr = np.nan
                    if verbose==2:
                        print("[Warning] Unable to get an upper limit with MINOS. Use an alternative method (profile likelihood).")

            # Estimate an upper limit with profile likelihood
            if (np.isnan(signu_ul)) or (ctr == None) or (method==2) or (method==3):
                gLb, gLSignu = self.__scanProfile__(skip_b=(fix_b or self.expectedLimit))
                
                x_signu, y_signu = getArray(gLSignu)
                self.gL = np.asarray([[x+self.signu0, y-self.logl] for x, y in zip(x_signu, y_signu)])
                aboveMax = (y_signu>min(y_signu))
                st_idx = list(aboveMax).index(False)
                try:
                    logl_int = interp1d(y_signu[st_idx:], x_signu[st_idx:], kind='linear')
                    signu_ul = logl_int(self.logl+2.71/2)+self.signu0
                    if method == 3:
                        signu_ul_lp = signu_ul
                except:
                    self.plotProfileLikelihood(upperLimit=False, invert=True)
                    plt.show(block=False)
                    signu_ul = np.nan
                    if method == 3:
                        signu_ul_lp = np.nan
                    self.signu_ul = signu_ul
                    print("[Log, Error] The upper limit is not estimated (M={:.3f} TeV).".format(self.M/1000.))
                    
            if verbose==2:
                print("[Log] An upper limit is estimated.")
            
        if (method == 3) and (np.isfinite(signu_ul_ctr)) and (np.isfinite(signu_ul_lp)):
            self.signu_ul = min(signu_ul_ctr, signu_ul_lp)
        else:
            self.signu_ul = signu_ul

        if verbose:
            if self.ts < 1 and self.ts > 0:
                print("[Log, Result] The signal (M={:.0f} GeV) is not significant (TS < 1).".format(self.M))
            elif self.ts< -1:
                print("[Log, Error] For M={:.0f} GeV, TS value is negative (TS = {:.2f})".format(self.M, self.ts))
            else:
                print("[Log, Result] TS = {:.2f} (M={:.0f} GeV)".format(self.ts, self.M))

            if self.ts<25 and upperLimit:
                print(u"[Log, Result] <\u03C3\u03BD> (95& upper limit) = 10^({:.3f}) (equivalent to {:.2e})".format(self.signu_ul, 10**self.signu_ul))
            else:
                print(u"[Log, Result] <\u03C3\u03BD> = 10^({:.3f} +/- {:.3f})".format(*self.signu))
                
            if fix_b:
                print("[Log, Result] b = fixed")
            else:
                print("[Log, Result] b = {:.0f} +/- {:.0f} (null value: {:.0f})".format(*self.b, self.b_null))
            
            if verbose == 2:
                if self.singleIRF:
                    hSignal_fit = self.hSignal.Clone()
                    hSignal_fit.Scale(10**(self.signu[0]-self.signu0))
                    if hSignal_fit.Class_Name() == "TH2D":
                        hSignal_fit = convert2Dsignal(hSignal_fit)
                    x_s, y_s = getArray(hSignal_fit)
                else:
                    y_s = 0
                    for v in self.listOfVersions:
                        hSignal_fit = self.hSignal[v].Clone()
                        hSignal_fit.Scale(10**(self.signu[0]-self.signu0))
                        if hSignal_fit.Class_Name() == "TH2D":
                            hSignal_fit = convert2Dsignal(hSignal_fit)
                        x_s, y_s_temp = getArray(hSignal_fit)
                        y_s+=y_s_temp
                print("[Log, Result] N_on = {:.0f}, N_off = {:.0f}, N_dm = {:.0f}".format(self.N_on, self.b[0]*self.alpha, sum(y_s)))
        
        elif self.ts<-1:
                print("[Log, Error] For M={:.0f} GeV, TS value is negative (TS = {:.2f})".format(self.M, self.ts))

    def bkg_minuit(self, model="powerlaw", fix_idx=False, index=-3.5, verbose=None):
        if verbose==None:
            verbose=self.verbose
            printLevel = -1
        elif verbose == 3:
            printLevel = 1
        else:
            printLevel = -1
        
        Math.MinimizerOptions.SetDefaultMinimizer("Minuit2");

        fit = TMinuit(3)
        
        fit.SetPrintLevel(printLevel=printLevel)
        
        # Parameter setting
        fit.mnrset(1)
        
        ierflg = ctypes.c_int(199)
        istat = ctypes.c_int(12)

        arglist = array( 'd', 10*[0.] )
        arglist[0] = 10000
        arglist[1] = 1.

        # Import a model
        fit.SetFCN(fcn_bkg)
        
        fit.mnparm(0,"N    ", 1,   0.01,   0.001,   1000,  ierflg)
        fit.mnparm(1,"index", index,     0.01,    -10,     -1,  ierflg)
        
        # This is for running multiple MLE.
        fit.mnparm(2,"package", self._pn,     0,    self._pn,     self._pn,  ierflg)
        fit.FixParameter(2)

        if model == "powerlaw":
            fit.mnparm(3,"model", 1,     0,    1,     1,  ierflg)
        elif model == "bknpl":
            fit.mnparm(3,"model", 2,     0,    2,     2,  ierflg)
            fit.mnparm(4,"Eb    ", 5000,  1,   1e3,   1e6,  ierflg)
            fit.mnparm(5,"index2", -2.2,     0.01,    -5,     -1,  ierflg)
        else:
            print("[Error] Model is either 'powerlaw' (default) or 'bknpl'.")
            return
        
        fit.FixParameter(3)

        if fix_idx:
            if model == "powerlaw":
                fit.FixParameter(1)
            else:
                fit.FixParameter(1)
                fit.FixParameter(5)


        # Initial minimization
        fit.SetErrorDef(1)

        fit.mnexcm("MIGRAD", arglist, 2, ierflg);
        fit.mnmnos()
        
        if ierflg.value == 0:
            if verbose==2:
                print("[Log] MINUIT finds a minimum successfully (MIGRAD is converged).")

        elif ierflg.value == 4:
            print("[Error] MIGRAD is NOT converged. Check initial parameters (minuit in mle.py).")
                
        else:
            print("[Error] An error occurs (type={}, https://root.cern.ch/doc/master/classTMinuit.html#ab48dd4b48800edc090396e35cb465fb9)".format(int(ierflg.value )))
        
        fit.mnimpr()
        
        logl, edm, errdef = map(ctypes.c_double, (0.18, 0.19, 0.20))
        nvpar, nparx, icstat = map(ctypes.c_int, (1983, 1984, 1985))
        N_min, N_err = map(ctypes.c_double, (0.50, 0.51))
        index_min, index_err = map(ctypes.c_double, (0.40, 0.41))
        
        fit.mnstat( logl, edm, errdef, nvpar, nparx, icstat )        
        fit.GetParameter(0, N_min, N_err)
        fit.GetParameter(1, index_min, index_err)
        
        self.N_bkg_fit = (float(N_min.value), float(N_err.value))
        self.index_bkg_fit = (float(index_min.value), float(index_err.value))
        
        if model == "powerlaw":
            self.bkg_pars = (self.N_bkg_fit[0], self.index_bkg_fit[0])
        else:
            Eb_min, Eb_err = map(ctypes.c_double, (0.30, 0.31))
            index2_min, index2_err = map(ctypes.c_double, (0.60, 0.61))
            fit.GetParameter(4, Eb_min, Eb_err)
            fit.GetParameter(5, index2_min, index2_err)
            self.Eb_bkg_fit = (float(Eb_min.value), float(Eb_err.value))
            self.index2_bkg_fit = (float(index2_min.value), float(index2_err.value))
            self.bkg_pars = (self.N_bkg_fit[0], self.index_bkg_fit[0], self.Eb_bkg_fit[0], self.index2_bkg_fit[0])
        
        
        self.bkg_flag = ierflg.value
        self.bkg_fit = fit
        self.bkg_logl = logl.value

        if verbose:
            print(u"[Log, Result] N = {:.3f} +/- {:.3f}".format(*self.N_bkg_fit))
            print(u"[Log, Result] Index = {:.3f} +/- {:.3f}".format(*self.index_bkg_fit))
            if model == "bknpl":
                print(u"[Log, Result] Eb = {:.3f} +/- {:.3f}".format(*self.Eb_bkg_fit))
                print(u"[Log, Result] Index2 = {:.3f} +/- {:.3f}".format(*self.index2_bkg_fit))

    def plotSED(self):
        hOff_fit = self.hOff.Clone()
        hOff_fit.Scale(self.b[0]/self.N_off)

        hSignal_fit = self.hSignal.Clone()
        hSignal_fit.Scale(10**(self.signu[0]-self.signu0))
        
        if self.addTheta:
            f, ax = plt.subplots(2,2, figsize=(15, 10))

            val_on, x_on, y_on = getArray(self.hOn)
            val_off, x_off, y_off = getArray(hOff_fit)
            val_s = []

            x_s, y_s = x_off, y_off
            for x in x_off:
                temp = []
                for y in y_off:
                    temp.append(hSignal_fit.Interpolate(x, y))
                val_s.append(temp)
            val_s = np.asarray(val_s).T

            ax[0][0] = plot2D(x_s, y_s, val_s+val_off, ax[0][0])
            ax[0][0].set_title("Model + Background (folded)", fontsize=15)
            ax[0][0].set_xlabel(r"log$_{10}$(Energy) [GeV]", fontsize=15)
            ax[0][0].set_ylabel(r"Theta$^2$ [deg$^2$]", fontsize=15)

            ax[0][1] = plot2D(x_on, y_on, val_on, ax[0][1])
            ax[0][1].set_title("Observed events", fontsize=15)
            ax[0][1].set_xlabel(r"log$_{10}$(Energy) [GeV]", fontsize=15)
            ax[0][1].set_ylabel(r"Theta$^2$ [deg$^2$]", fontsize=15)

            ax[1][0] = plot2D(x_s, y_s, val_s, ax[1][0])
            ax[1][0].set_title("DM folded spectrum", fontsize=15)
            ax[1][0].set_xlabel(r"log$_{10}$(Energy) [GeV]", fontsize=15)
            ax[1][0].set_ylabel(r"Theta$^2$ [deg$^2$]", fontsize=15)

            diff = []
            for i in range(len(y_s)):
                temp = []
                for j in range(len(x_s)):
                    if val_on[i][j]!=0:
                        temp.append((val_s[i][j]+val_off[i][j]-val_on[i][j])**2/val_on[i][j])
                    else:
                        temp.append(0)
                diff.append(temp)
            diff=np.asarray(diff)
            
            ax[1][1] = plot2D(x_s, y_s, diff, ax = ax[1][1], vmax=5)
            ax[1][1].set_title(r"$\chi^2$ for each bin", fontsize=15)
            ax[1][1].set_xlabel(r"log$_{10}$(Energy) [GeV]", fontsize=15)
            ax[1][1].set_ylabel(r"Theta$^2$ [deg$^2$]", fontsize=15)

            plt.tight_layout()
            
        else:
            f, ax = plt.subplots(2,1, figsize=(7, 7), gridspec_kw={'height_ratios':[5,1]})
            
            x_on, y_on = getArray(self.hOn)
            x_off, y_off = getArray(hOff_fit)
            
            x_s = x_off
            y_s = []
            for x in x_off:
                y_s.append(hSignal_fit.Interpolate(x))
            y_s = np.asarray(y_s)

            x_fit = x_off
            y_fit = y_off + y_s
            N_fit = int(sum(y_fit))

            ax[0].step(x_fit, y_fit, zorder=2, label="Total (signal+bg)", where="mid")
            ax[0].step(x_s, y_s, ls=":", label="DM signal fit", where="mid")
            ax[0].step(x_off, y_off, ls=":", label="Background fit", where="mid")
            ax[0].errorbar(x_on, y_on, yerr=np.sqrt(y_on), ls = "", marker="x", color="k", label="Observed")

            ax[0].set_xscale("log")
            ax[0].set_yscale("log")
            ax[0].set_xlim(80, 2e5)
            ax[0].set_ylim(0.8,)
            ax[0].set_ylabel("Counts", fontsize=15)
            ax[0].legend(fontsize=12, loc=3)
            ax[0].grid()

            ax[1].errorbar(x_on[y_on!=0], np.sign(y_on[y_on!=0]-y_fit[y_on!=0])*(y_fit[y_on!=0]-y_on[y_on!=0])**2./y_on[y_on!=0], yerr= 1, ls="", marker="x", color="k", label=r"$\chi^2")
            ax[1].set_xscale("log")
            ax[1].set_xlabel("Energy [GeV]", fontsize=15)
            ax[1].set_ylabel(r"$\chi^2$", fontsize=15)
            ax[1].set_xlim(80, 2e5)
            ax[1].set_ylim(-5, 5)
            ax[1].axhline(0, color="k", ls="--")
            ax[1].grid()

            plt.show(block=False)  
        
    def plotProfileLikelihood(self, invert=False, error=False, manual=False, upperLimit=True, xlim=[None, None]):

        gLb, gLSignu = self.__scanProfile__(skip_sig=manual)
        x_b, y_b = getArray(gLb)

        if manual:
            x_signu, y_signu = self.__manualScan__()
            print(x_signu, y_signu)
        else:
            x_signu, y_signu = getArray(gLSignu)

        
        self._likelihood = np.asarray([10**(x_signu+self.signu0), y_signu-self.logl]).T

        f, ax = plt.subplots(1,2, figsize=(12, 4))
        
        if invert:
            ax[0].plot(10**(x_signu+self.signu0), y_signu-self.logl, label="Likelihood")
            ax[1].plot(x_b, y_b-self.logl, label="Likelihood")
            if error:
                ax[1].axhline(1, color="r", ls=":", label=r"1$\sigma$ cont. (68%)")
                ax[1].axvline(self.b[0]+self.b[1], color="r", ls=":")
                ax[1].axvline(self.b[0]-self.b[1], color="r", ls=":")
            if upperLimit:
                ax[0].axhline(2.71/2, color="r", ls=":", label="Upper limit (95%)")
                ax[0].axvline(10**self.signu_ul, color="r", ls=":")
        else:
            ax[0].plot(10**(x_signu+self.signu0), -y_signu, label="Likelihood")
            ax[1].plot(x_b, -y_b, label="Likelihood")
            if error:
                ax[1].axhline(1, color="r", ls=":", label=r"1$\sigma$ cont. (68%)")
                ax[1].axvline(self.b[0]+self.b[1], color="r", ls=":")
                ax[1].axvline(self.b[0]-self.b[1], color="r", ls=":")
            if upperLimit:
                ax[0].axhline(-self.logl - 2.71/2, color="r", ls=":", label="Upper limit (95%)")
                ax[0].axvline(10**self.signu_ul, color="r", ls=":")
            
        ax[0].axvline(10**self.signu[0],color="k")
        ax[0].set_xscale("log")
        ax[0].set_xlabel(r"<$\sigma\nu$> [cm$^3$ s$^{-1}$]", fontsize=15)
        ax[0].set_xlim(xlim[0], xlim[1])

        ax[1].axvline(self.b[0],color="k")
        ax[1].set_xlabel(r"Expected background events", fontsize=15)

        for i in range(2):
            
            if invert:
                ax[i].axhline(0, color="k", label="Best fit")
                ax[i].set_ylabel(r"log($\mathcal{L}_{max}$) - log($\mathcal{L}$)", fontsize=15)
                ax[i].set_ylim(-1, 10)
            else:
                ax[i].axhline(-self.logl, color="k", label="Best fit")
                ax[i].set_ylabel(r"log($\mathcal{L}$)", fontsize=15)
                ax[i].set_ylim(-self.logl-10, -self.logl+1)
            
            ax[i].legend(fontsize=12)
            ax[i].grid()

    def plotContour(self, upperLimit = False):

        if upperLimit:
            self.fit.SetErrorDef(2.71/2)
            ctr = self.fit.Contour(50, 0, 1)
            x_1, y_1 = getArray(ctr)
            plt.plot(x_1,y_1+self.signu0, label=r"95% confidence interval")
            plt.axhline(self.signu_ul, color="r", ls=":", label="Upper limit")
        else:
            self.fit.SetErrorDef(1)
            ctr = self.fit.Contour(50, 0, 1)
            x_1, y_1 = getArray(ctr)

            self.fit.SetErrorDef(4)
            ctr = self.fit.Contour(50, 0, 1)
            x_2, y_2 = getArray(ctr)

            self.fit.SetErrorDef(9)
            ctr = self.fit.Contour(50, 0, 1)
            x_3, y_3 = getArray(ctr)

            plt.plot(x_1,y_1+self.signu0, label=r"1$\sigma$")
            plt.plot(x_2,y_2+self.signu0, label=r"2$\sigma$")
            plt.plot(x_3,y_3+self.signu0, label=r"3$\sigma$")

            
        plt.xlabel(r"Expected background events", fontsize=15)
        plt.ylabel(r"log$_{10}$(<$\sigma\nu$>) [cm$^3$ s$^{-1}$]", fontsize=15)
        plt.scatter(self.b[0], self.signu[0], marker="x", color="r")
        plt.legend(fontsize=12)

