import numpy as np
import os

from .mle import MLE
from .. import ResponseFunction
from .. import JProfile
from ..const import OUTPUT_DIR, defaultNum
from .. import const
from ..utils import defineTheta2Cut, listOfVersions
from .signal import calcSignal, combinedCalcSignal
from .fcn import fcn

from . import eventdisplay, vegas

from tqdm.notebook import trange, tqdm

from ROOT import TMinuit, Math

import ctypes

from array import array

import multiprocess

import random

from ..ResponseFunction.eventdisplay import th2cut_ext

def calcUpperLimits(dwarf, channel, package="EventDisplay", 
    irf=None, jProfile = None, jArray=True, version="all", th2Cut = 0, 
    addTheta=False, averagedIRF=True, method = 1, fix_b=False, 
    filename = None, seed=0, jSeed = "median", overWrite=False,
    mass = np.logspace(2, 4.5, 20), 
    bkgModel=None, ext=False, statistic="unbinned",
    verbosity=True, 
    returnTS=False, returnProfile = False, returnUL=False, returnMLE = False,
    write=True,
    test=False,  **kwargs):

    if ext and (th2Cut == 0):
        th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
    else:
        th2Cut = defineTheta2Cut(package, th2Cut)

    useBias = kwargs.get("useBias", True)

    if channel == "delta" or channel == "gamma":
        if package == "EventDisplay":
            if not(useBias):
                mass4gamma = const.mass4gamma_disp
            else:
                mass4gamma = const.mass4gamma
        elif package == "VEGAS":
            mass4gamma = const.mass4gamma_vegas
            new_mass = []
            for i, m in enumerate(mass4gamma):
                if i%5 == 0:
                    new_mass.append(m)
            mass4gamma = np.array(new_mass)
            
        mass = mass4gamma[(mass4gamma>=mass[0])*(mass4gamma<=mass[-1])]

    DM_spectra = kwargs.get("DM_spectra", "PPPC")

    if DM_spectra == "HDM" and min(mass) < 1000:
        mass = np.asarray([1000]+mass[mass>1000].tolist())

    if verbosity:
        print("="*50)
        print("[Log] Package      :", package)
        print("[Log] Dwarf        :", dwarf)
        print("[Log] Channel      :", channel)
        if ext:
            print(f"[Log] Dataset      : Extended (theta2={th2Cut})")
        else:
            print("[Log] Dataset      : Point-like")

        print("[Log] Dimention    :", int(addTheta)+1)

        if bkgModel == "ex":
            print("[Log] Background   : Extrapolation (ex)")
        elif bkgModel == "sm":
            print("[Log] Background   : Smoothing (sm)")
        elif bkgModel == "alt":
            print("[Log] Background   : Alternative (alt)")
        elif bkgModel == "gaus":
            print("[Log] Background   : Gaussian (gaus)")
        else:
            print("[Log] Background   : None")
            
        if useBias:
            print(r"[Log] Dispersion   : Etr vs ratio")
        else:
            print(r"[Log] Dispersion   : Etr vs Erec")
        print("[Log] Minimum mass : {:.0f} GeV".format(mass[0]))
        print("[Log] Maximum mass : {:.0f} TeV".format(mass[-1]/1e3))
        print("="*50)

    if verbosity>1:
        print("[Log] Initialization", end="\r")

    ul = []

    if jArray:
        if jSeed == "median":
            jSeed = defaultNum[dwarf]
        elif jSeed == -1:
            jSeed = random.randrange(0, 100000)


        if verbosity>1:
            print("[Log] Importing the J profile (seed: {}).".format(jSeed))

    singleIRF = bool((averagedIRF) + (version != "all"))
    tau = [1] 
    if irf==None:
        if package=="EventDisplay":
            if singleIRF:
                try:                
                    importedIRF = ResponseFunction.EventDisplay.readIRFs(dwarf, version=version, ext=ext)
                    err = ResponseFunction.checkIRF(importedIRF)
                except:
                    err = True
                
                if err:
                    importedIRF = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, export=False, verbose=(verbosity>1), ext=ext, th2Cut=th2Cut)
                    
                tau = [1]
                irf = importedIRF
                if jArray:
                    jProfile = JProfile.generateConvolvedJ(dwarf, package, return_array=True, seed = jSeed, verbose=False, ext=ext, th2Cut=th2Cut, **kwargs)
            else:
                importedIRF = {}
                tau = []
                importedJProfile = {}
                for v in listOfVersions(dwarf):
                    try:
                        importedIRF[v] = ResponseFunction.EventDisplay.readIRFs(dwarf, version=v,  ext=ext)
                        err = ResponseFunction.checkIRF(importedIRF[v])
                    except:
                        err = True    
                    if err:
                        importedIRF[v] = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=v, export=False, verbose=(verbosity>1), ext=ext)
                    if jArray:
                        importedJProfile[v] = JProfile.generateConvolvedJ(dwarf, package, irf = importedIRF[v], return_array=True, seed = jSeed, verbose=False, version=v, ext=ext, th2Cut=th2Cut, **kwargs)
                    tau.append(importedIRF[v].exposure)
                tau = np.asarray(tau)/sum(tau)
                irf = importedIRF
                if jArray:
                    jProfile = importedJProfile
        elif package=="VEGAS":
            try:
                importedIRF = ResponseFunction.VEGAS.readIRFs(dwarf)
            except:
                importedIRF = ResponseFunction.VEGAS(dwarf, verbose=(verbosity>1))
            
            jProfile = JProfile.generateConvolvedJ(dwarf, package, irf = importedIRF, return_array=True, seed = jSeed, verbose=False,  **kwargs)
    else:
        importedIRF=irf
        jProfile = JProfile.generateConvolvedJ(dwarf, package, irf = importedIRF, version=version, return_array=True, seed = jSeed, verbose=False,  th2Cut=th2Cut, ext=ext)
    

    if verbosity>1:
        print("[Log] Initialization (Done)                      ")

    if verbosity>1:
        print("[Log] Upper-limit calculation", end="\r")
    gl = {}
    ts = []


    for i, M in tqdm(enumerate(mass), total=len(mass)) if verbosity else enumerate(mass):
        mle = MLE(dwarf, M, package, channel=channel, irf=importedIRF, jProfile=jProfile, jArray=jArray,
                th2Cut=th2Cut, addTheta=addTheta, statistic=statistic,
                averagedIRF=averagedIRF, version=version, tau=tau,
                seed=seed, jSeed=jSeed, test=test, 
                bkgModel=bkgModel, ext=ext,
                verbose=(True if verbosity>2 else False), **kwargs) 
        mle.minuit(method=method, upperLimit=True, fix_b=fix_b, **kwargs)

        ul.append([M, mle.signu_ul])
        ts.append([M, mle.ts])
        if mle.ts >=25:
            if verbosity>1:
                print("[Warning] TS value is higher than 25 (M={:.3f} TeV).".format(M/1000))
            continue
        
        gl[M] = mle.gL
        if verbosity>1:
            print(u"[Log, Result] M = {:5.2f} TeV, TS = {:5.2f}, b = {:5.0f}, <\u03C3\u03BD> = {:5.2e}, and  <\u03C3\u03BD> (95& upper limit) = {:5.2e}".format(M/1000., mle.ts, mle.b[0], 10**mle.signu[0], 10**mle.signu_ul))
    if verbosity>1:
        print("[Log] Upper-limit calculation (Done)                                           ")

    gl[0] = mle.logl0
    if addTheta:
        dim = "2D"
    else:
        dim = "1D"

    if write:
        if filename == None:
            if not(os.path.isdir(OUTPUT_DIR)):
                os.system("mkdir "+OUTPUT_DIR)
            if bkgModel == None:
                bkg = "null"
            else:
                bkg = bkgModel

            if version =="all":
                filename = "{}_{}_{}_{}_{}".format(package, dwarf, channel, dim, bkg)
            else:
                filename = "{}_{}_{}_{}_{}_{}".format(package, dwarf, channel, dim, bkg, version)
            if not(ext):
                filename += "_pt"

        if overWrite:
            np.save(OUTPUT_DIR+filename, ul)
        else:
            os.system("cp "+OUTPUT_DIR+filename+".npy"+" "+OUTPUT_DIR+filename[:-4]+"_prev.npy")
            np.save(OUTPUT_DIR+filename, ul)
        
        if verbosity:
            print("[Log] Upper limits are saved in '{}.npy'".format(OUTPUT_DIR+filename))

    output = {}
    if returnUL:
        output["UpperLimit"] = ul
    if returnProfile:
        output["Profile"] = gl
    if returnTS:
        output["TS"] = np.asarray(ts)
        np.save(OUTPUT_DIR+filename+"_ts", np.asarray(ts))
    if returnMLE:
        output["SampleMLE"] = mle

    if (returnUL or returnProfile or returnTS or returnMLE):
        return output


def calcExpectedLimits(dwarf, channel, package="EventDisplay", 
    irf=None, jProfile = None, jArray=False, version="all", th2Cut = 0, 
    addTheta=False, averagedIRF=False, method = 1, fix_b=False, 
    filename = None, seed=3, jSeed = -1,
    mass = np.logspace(2, 4.5, 12), 
    bkgModel=None, ext=False, test=False, 
    verbosity=True, runs = 1000, multi=True, **kwargs):
    
    if ext and (th2Cut == 0):
        th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
    else:
        th2Cut = defineTheta2Cut(package, th2Cut)

    useBias = kwargs.get("useBias", True)

    if package=="EventDisplay":
        if channel == "delta" or channel == "gamma":
            if not(useBias):
                mass4gamma = const.mass4gamma_disp
            else:
                mass4gamma = const.mass4gamma
                
            mass = mass4gamma[(mass4gamma>mass[0])*(mass4gamma<mass[-1])]

    if verbosity:
        print("="*50)
        print("[Log] Package      :", package)
        print("[Log] Dwarf        :", dwarf)
        print("[Log] Channel      :", channel)
        if useBias:
            print(r"[Log] Dispersion   : Etr vs ratio")
        else:
            print(r"[Log] Dispersion   : Etr vs Erec")
        print("[Log] Minimum mass : {:.0f} GeV".format(mass[0]))
        print("[Log] Maximum mass : {:.0f} TeV".format(mass[-1]/1e3))
        print("="*50)

    if verbosity>1:
        print("[Log] Initialization", end="\r")
    
    singleIRF = bool((averagedIRF) + (version != "all")) 

    DM_spectra = kwargs.get("DM_spectra", "PPPC")

    if DM_spectra == "HDM" and min(mass) < 1000:
        mass = np.asarray([1000]+mass[mass>1000].tolist())

    tau = [1]
    if irf==None:
        if package=="EventDisplay":
            if singleIRF:
                try:
                    importedIRF = ResponseFunction.EventDisplay.readIRFs(dwarf, version=version, ext=ext)
                    err = ResponseFunction.checkIRF(importedIRF)
                except:
                    err = True
                if err:
                    importedIRF = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, export=False, ext=ext)
                tau = [1]
            else:
                importedIRF = {}
                tau = []
                for v in listOfVersions(dwarf):
                    try:
                        importedIRF[v] = ResponseFunction.EventDisplay.readIRFs(dwarf, version=v, ext=ext)
                        err = ResponseFunction.checkIRF(importedIRF[v])
                    except:
                        err = True
                    if err:
                        importedIRF[v] = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=v, export=False, ext=ext)
                    tau.append(importedIRF[v].exposure)
                tau = np.asarray(tau)/sum(tau)

        elif package=="VEGAS":
            try:
                importedIRF = ResponseFunction.VEGAS.readIRFs(dwarf, verbose=False)
            except:
                importedIRF = ResponseFunction.VEGAS(dwarf, verbose=False)
    else:
        importedIRF = irf

    if package=="EventDisplay":
        raw_events = eventdisplay.readData(dwarf, rawdata=True, ext=ext, version=version)
    elif package=="VEGAS":
        raw_events = vegas.readData(dwarf, rawdata=True)
    else:
        return

    if test:
        bkg = kwargs.pop("events", raw_events[raw_events[:,2]==0.0])
        alpha = kwargs.get("alpha", np.average(raw_events[:,3]))
        N_on = kwargs.pop("N_on", len(bkg)*alpha)
    else:
        bkg = raw_events[raw_events[:,2]==0.0]
        alpha = raw_events[:,3][0]
        N_on = len(bkg)*alpha

    ul = {}

    
    if jProfile is None:
        if jArray and jSeed == -1:
            jSeed = random.randrange(0, JProfile.goodPropNum(dwarf)-1)

        jProfile = JProfile.generateConvolvedJ(dwarf, package, irf=importedIRF, version=version, return_array=True, seed = jSeed, verbose=False,  th2Cut=th2Cut, ext=ext)
    
    hSignal = {}
    for i, M in enumerate(mass):
        ul[M] = []
        if singleIRF:
            hSignal[M] = calcSignal(dwarf, M, importedIRF, package, eLowerCut=min(bkg[:,0]), jProfile=jProfile, channel=channel, addTheta=addTheta, th2Cut=th2Cut, version=version, jArray=jArray, jSeed=jSeed, verbose=False, ext=ext, DM_spectra=DM_spectra)
            hSignal[M].SetDirectory(0)
        else:
            hSignal[M] = {}
            for v in listOfVersions(dwarf):
                hSignal[M][v] = calcSignal(dwarf, M, importedIRF[v], package, eLowerCut=min(bkg[:,0]), jProfile=jProfile, channel=channel, addTheta=addTheta, th2Cut=th2Cut, version=version, jArray=jArray, jSeed=jSeed, verbose=False, ext=ext, DM_spectra=DM_spectra)
                hSignal[M][v].SetDirectory(0)

    if verbosity>1:
        print("[Log] Initialization (Done)                      ")

    if verbosity>1:
        print("[Log] Expected-upper-limits calculation", end="\r")
    
    if multi:
        pbar = tqdm(total=runs)
    else:
        pbar = tqdm(total=len(mass)*runs)
    
    
    ts_all = []
    
    manager = multiprocess.Manager()

    for j in range(runs):
        N_on_poi = np.random.poisson(N_on)
        if N_on_poi == 0:
            N_on_poi = 1
        selected = np.random.choice(range(len(bkg)), size=N_on_poi)

        events = bkg[selected]
        events[:,2] = 1
        processes = []
        
        if package=="EventDisplay":
            hOn, etc = eventdisplay.readData(dwarf, events=events, ext=ext, version=version)
        elif package=="VEGAS":
            hOn, etc = vegas.readData(dwarf, events=events)
        else:
            hOn = None

        output = manager.dict()
        
        for i, M in enumerate(mass):
            if multi:    
                mle_kwargs = {"channel":channel, "irf":importedIRF, "jProfile":jProfile, "jArray":jArray,
                    "th2Cut":th2Cut, "addTheta":addTheta, "expectedLimit":True,
                    "averagedIRF":averagedIRF, "tau":tau, "version":version, 
                    "seed":seed, "pN": i, "ext":ext, 
                    "bkgModel":bkgModel, "N_on": N_on_poi,
                    "verbosity":verbosity, "hOn": hOn,
                    "events": events,  "hSignal":hSignal[M], **kwargs}
                p = multiprocess.Process(target=multiprocessing_mle, args=(dwarf, M, package, output, ), kwargs=mle_kwargs)
                processes.append(p)
                p.start()
            else:
                mle = MLE(dwarf, M, package, channel=channel, irf=importedIRF, jProfile=jProfile, 
                        th2Cut=th2Cut, addTheta=addTheta, 
                        averagedIRF=averagedIRF, version=version, 
                        seed=seed, expectedLimit=True, 
                        bkgModel=bkgModel, ext=ext, N_on=N_on_poi,
                        verbose=(True if verbosity>1 else False), 
                        events = events, hSignal=hSignal[M], hOn = hOn, **kwargs) 
                mle.minuit(method=method, upperLimit=True, fix_b=fix_b, **kwargs)
                ts_all.append(mle.ts)
                if not(np.isnan(mle.signu_ul)):
                    ul[M].append(mle.signu_ul)

                if mle.ts >=25:
                    if verbosity>1:
                        print("[Warning] TS value is higher than 25 (M={:.3f} TeV).".format(M/1000))
                    continue
                pbar.update(1)

        if multi:
            for process in processes:
                process.join()

            for i, M in enumerate(mass):
                out = output.pop(M)
                signu_ul = out[0]
                ts = out[1]
                ts_all.append(ts)
                if not(np.isnan(signu_ul)):
                    ul[M].append(signu_ul)
            pbar.update(1)

    ts_all = np.asarray(ts_all)
    if verbosity>1:
        print("[Log] Expected-upper-limits calculation (Done)                                           ")
        print("[Log] There are {} out of {} runs ({:.2f})% which have TS>5.".format(sum(ts_all>5), runs*len(mass), sum(ts_all>5)/(runs*len(mass))*100.))

    if addTheta:
        dim = "2D"
    else:
        dim = "1D"

    if filename == None:
        if not(os.path.isdir(OUTPUT_DIR)):
            os.system("mkdir "+OUTPUT_DIR)
        if version == "all":
            filename = "{}_{}_{}_{}_exp".format(package, dwarf, channel, dim)
        else:
            filename = "{}_{}_{}_{}_{}_exp".format(package, dwarf, channel, dim, version)
        
    np.save(OUTPUT_DIR+filename, ul)
    if verbosity:
        print("[Log] Expected-upper-limits are saved in '{}.npy'".format(OUTPUT_DIR+filename))


def calcULSysError(dwarf, channel, package="EventDisplay", 
    irf=None, jArray=True, version="all", th2Cut = 0, 
    addTheta=False, averagedIRF=False, method = 1, 
    filename = None, seed=4, overWrite=False,
    mass = np.logspace(2, 4.5, 12), 
    bkgModel=None, runs = 1000, ext=False, 
    verbosity=True, **kwargs):
    
    if ext and (th2Cut == 0):
        th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
    else:
        th2Cut = defineTheta2Cut(package, th2Cut)
    useBias = kwargs.get("useBias", True)

    if verbosity:
        print("="*50)
        print("[Log] Package      :", package)
        print("[Log] Dwarf        :", dwarf)
        print("[Log] Channel      :", channel)
        print("[Log] Dimention    :", int(addTheta)+1)
        
        if bkgModel == "ex":
            print("[Log] Background   : Extrapolation (ex)")
        elif bkgModel == "sm":
            print("[Log] Background   : Smoothing (sm)")
        elif bkgModel == "alt":
            print("[Log] Background   : Alternative (alt)")
        elif bkgModel == "gaus":
            print("[Log] Background   : Gaussian (gaus)")
        else:
            print("[Log] Background   : None")
            
        if useBias:
            print(r"[Log] Dispersion   : Etr vs ratio")
        else:
            print(r"[Log] Dispersion   : Etr vs Erec")
        print("[Log] Minimum mass : {:.0f} GeV".format(mass[0]))
        print("[Log] Maximum mass : {:.0f} TeV".format(mass[-1]/1e3))
        print("="*50)

    if verbosity>1:
        print("[Log] Initialization", end="\r")

    ul = []

    DM_spectra = kwargs.get("DM_spectra", "PPPC")

    if DM_spectra == "HDM" and min(mass) < 1000:
        mass = np.asarray([1000]+mass[mass>1000].tolist())

    singleIRF = bool((averagedIRF) + (version != "all")) 
    if irf==None:
        if package=="EventDisplay":
            if singleIRF:
                try:
                    importedIRF = ResponseFunction.EventDisplay.readIRFs(dwarf, version=version, ext=ext)
                    err = ResponseFunction.checkIRF(importedIRF)
                except:
                    err = True
                if err:
                    importedIRF = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, export=False, verbose=(verbosity>1), ext=ext)
                tau = [1]
            else:
                importedIRF = {}
                tau = []
                for v in listOfVersions(dwarf):
                    try:
                        importedIRF[v] = ResponseFunction.EventDisplay.readIRFs(dwarf, version=v,  ext=ext)
                        err = ResponseFunction.checkIRF(importedIRF[v])
                    except:
                        err = True
                    if err:
                        importedIRF[v] = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=v, export=False, verbose=(verbosity>1), ext=ext)
                    tau.append(importedIRF[v].exposure)
                tau = np.asarray(tau)/sum(tau)
        elif package=="VEGAS":
            try:
                importedIRF = ResponseFunction.VEGAS.readIRFs(dwarf, verbose=(verbosity>1))
            except:
                importedIRF = ResponseFunction.VEGAS(dwarf, verbose=(verbosity>1))

            tau = [1]
    else:
        importedIRF=irf
        tau = [1]

    if verbosity>1:
        print("[Log] Initialization (Done)                      ")

    if verbosity>1:
        print("[Log] Upper-limit (including sys. error) calculation", end="\r")

    ul = {}
    for i, M in enumerate(mass):
        ul[M] = []

    manager = multiprocess.Manager()

    jSeeds = []

    for j in trange(runs):

        processes = []
        output = manager.dict()
        jSeed = random.randrange(0, 100000)
        jProfile = JProfile.generateConvolvedJ(dwarf, package, irf=importedIRF, version=version, return_array=True, seed = jSeed, verbose=False, ext=ext)
        jSeeds.append(jSeed)

        for i, M in enumerate(mass):
            mle_kwargs = {"channel":channel, "irf":importedIRF, "jProfile":jProfile, "jArray":jArray,
                    "th2Cut":th2Cut, "addTheta":addTheta, "expectedLimit":False, "jSeed": jSeed,
                    "averagedIRF":averagedIRF, "tau":tau, "version":version, 
                    "seed":seed, "pN": i, "ext":ext, 
                    "bkgModel":bkgModel, 
                    "verbosity":verbosity, **kwargs}
            p = multiprocess.Process(target=multiprocessing_mle, args=(dwarf, M, package, output, ), kwargs=mle_kwargs)
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

        for i, M in enumerate(mass):
            out = output.pop(M)
            signu_ul = out[0]
            if not(np.isnan(signu_ul)):
                ul[M].append(signu_ul)

            
    if verbosity>1:
        print("[Log] Upper-limit (including sys. error) calculation (Done)                                           ")

    if addTheta:
        dim = "2D"
    else:
        dim = "1D"


    if filename == None:
        if not(os.path.isdir(OUTPUT_DIR)):
            os.system("mkdir "+OUTPUT_DIR)
        if version == "all":
            filename = "{}_{}_{}_{}_sys".format(package, dwarf, channel, dim)
        else:
            filename = "{}_{}_{}_{}_{}_sys".format(package, dwarf, channel, dim, version)
        
    np.save(OUTPUT_DIR+filename, ul)
    np.save(OUTPUT_DIR+filename+"_seed", jSeeds)
    if verbosity:
        print("[Log] Upper-limits (including sys. error) are saved in '{}.npy'".format(OUTPUT_DIR+filename))

def multiprocessing_mle(dwarf, M, package, output, channel="tt", irf=None, jProfile=None, jArray=False,
                th2Cut=0, ext=False, addTheta=False, averagedIRF=False, tau = [1], version="all", seed=3, pN = 0, expectedLimit=False,
                bkgModel=None, verbosity= False, events = [], hSignal=None, **kwargs):
                
    mle = MLE(dwarf, M, package, channel=channel, irf=irf, jProfile=jProfile, jArray=jArray,
    th2Cut=th2Cut, addTheta=addTheta, 
    averagedIRF=averagedIRF, version=version, 
    seed=(seed*10+pN), expectedLimit=expectedLimit,
    bkgModel=bkgModel, tau=tau, ext=ext,
    verbose=False, 
    events = events,  hSignal=hSignal, **kwargs) 
    
    mle.minuit(method=2, upperLimit=True, fix_b=False, **kwargs)
    
    output[M] = [mle.signu_ul, mle.ts]
