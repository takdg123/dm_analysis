import numpy as np
import os

from .mle import MLE
from .. import ResponseFunction
from .. import JProfile
from .. import const
from ..const import OUTPUT_DIR, defaultNum
from ..utils import defineTheta2Cut, listOfVersions, getArray
from .signal import calcSignal, combinedCalcSignal
from .fcn import stackedfcn, stackedbinnedfcn

from tqdm.notebook import trange, tqdm

from ROOT import TMinuit, Math

import ctypes

from array import array

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from ..ResponseFunction.eventdisplay import th2cut_ext

from . import eventdisplay, vegas

def combinedUpperLimits(channel, package="EventDisplay", dwarfs = ["Segue_1", "UMi", "Draco", "Bootes_I"],
    addTheta=False, averagedIRF=True, method = 1, version="all", jSeed = "median", 
    filename = None, seed=0, overWrite=False, th2Cut=0, ext=False, sys=False,
    mass = np.logspace(2, 4.5, 15), bkgModel=None, verbosity=True, filter_he=False,
    returnTS = False, **kwargs):
    

    useBias = kwargs.get("useBias", True)

    if verbosity:
        print("="*50)
        print("[Log] Package      :", package)
        if len(dwarfs) == 1:
            print("[Log] Dwarf        :", dwarfs[0])
        else:
            print("[Log] # of Dwarfs  :", len(dwarfs))
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
            
        mass = mass4gamma[(mass4gamma>mass[0])*(mass4gamma<mass[-1])]

    DM_spectra = kwargs.get("DM_spectra", "PPPC")

    if DM_spectra == "HDM" and min(mass) < 1000:
        mass = np.asarray([1000]+mass[mass>1000].tolist())

    irf = {}
    tau = {}
    jProfile = {}
    singleIRF = averagedIRF

    
    for dwarf in dwarfs:
        if jSeed == "random" or sys:
            jS = -1
            allow_load=False
            save_array=False
        elif jSeed == "median":
            jS = defaultNum[dwarf]
            allow_load=True
            save_array=True

        else:
            jS = jSeed
            allow_load=False
            save_array=False

        if package == "EventDisplay":
            if singleIRF:
                try:
                    importedIRF = ResponseFunction.EventDisplay.readIRFs(dwarf, version=version, ext=ext)
                    err = ResponseFunction.checkIRF(importedIRF)
                    
                except:
                    err = True
                    
                if err:
                    importedIRF = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, export=True, ext=ext)

                tau[dwarf] = [1]
                irf[dwarf] = importedIRF
                jProfile[dwarf] = JProfile.generateConvolvedJ(dwarf, package, return_array=True, seed = jS, allow_load=allow_load, ext=ext, save_array=save_array, verbose=False, **kwargs)
            else:
                importedIRF = {}
                tau[dwarf] = []
                importedJProfile = {}
                for v in listOfVersions(dwarf):
                    try:
                        importedIRF[v] = ResponseFunction.EventDisplay.readIRFs(dwarf, version=v,  ext=ext)
                        err = ResponseFunction.checkIRF(importedIRF[v])
                    except:
                        err = True    
                    if err:
                        importedIRF[v] = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=v, export=False, ext=ext)
                    importedJProfile[v] = JProfile.generateConvolvedJ(dwarf, package, irf = importedIRF[v], return_array=True, seed = jS, verbose=False, version=v, ext=ext, **kwargs)
                    tau[dwarf].append(importedIRF[v].exposure)
                tau[dwarf] = np.asarray(tau[dwarf])/sum(tau[dwarf])
                irf[dwarf] = importedIRF
                jProfile[dwarf] = importedJProfile
        else:
            try:
                irf[dwarf] = ResponseFunction.VEGAS.readIRFs(dwarf)
            except:
                irf[dwarf] = ResponseFunction.VEGAS(dwarf, verbose=False)

            raw_events = vegas.readData(dwarf, rawdata=True)
            jProfile[dwarf] = JProfile.generateConvolvedJ(dwarf, package, irf = irf[dwarf], eLowerCut=min(raw_events[:,0]), return_array=True, seed = jS, verbose=False, **kwargs)
            tau[dwarf] = [1]

    if verbosity>1:
        print("[Log] Initialization (Done)                      ")

    if verbosity>1:
        print("[Log] Start upper-limit calculation")
    
    ts = []
    ul = []

    for i, M in tqdm(enumerate(mass), total=len(mass)) if verbosity else enumerate(mass):
        stackedMLE = {}
        for i, dwarf in enumerate(dwarfs):

            if ext and (th2Cut == 0):
                th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
            else:
                th2Cut = defineTheta2Cut(package, th2Cut)

            # if dwarf == "Segue_1":
            #     filter_he = False
            # else:
            #     filter_he = False

            mle = MLE(dwarf, M, package, channel=channel, irf=irf[dwarf], jProfile=jProfile[dwarf], jArray=True,
                    th2Cut=th2Cut, addTheta=addTheta, ext=ext,
                    averagedIRF=averagedIRF, version=version, tau=tau[dwarf],
                    seed=i+5,
                    bkgModel=bkgModel, filter_he=filter_he,
                    verbose=(True if verbosity>2 else False), **kwargs) 
            stackedMLE[dwarf] = mle
            
        ul_i, ts_i = combinedMinuit(stackedMLE, signu0 = mle.signu0, channel=channel, verbose=max(verbosity-1, 0), statistic = kwargs.get("statistic", "unbinned"))
        
        if ul_i != -1 and ts_i != -1:
            ul.append([M, ul_i])
            ts.append([M, ts_i])
            if ts_i >=25:
                if verbosity>1:
                    print("[Warning] TS value is higher than 25 (M={:.3f} TeV).".format(M/1000))
                continue
        
    if verbosity>1:
        print("[Log] Upper-limit calculation (Done)                                           ")

    if addTheta:
        dim = "2D"
    else:
        dim = "1D"

    if filename:
        if filename == None:
            if not(os.path.isdir(OUTPUT_DIR)):
                os.system("mkdir "+OUTPUT_DIR)
            if bkgModel == None:
                bkg = "null"
            else:
                bkg = bkgModel
            if version =="all":
                filename = "{}_{}_{}_{}_{}".format(package, "stacked", channel, dim, bkg)
            else:
                filename = "{}_{}_{}_{}_{}_{}".format(package, "stacked", channel, dim, bkg, version)
        
        if overWrite:
            np.save(OUTPUT_DIR+filename, ul)
        else:
            os.system("cp "+OUTPUT_DIR+filename+".npy"+" "+OUTPUT_DIR+filename[:-4]+"_prev.npy")
            np.save(OUTPUT_DIR+filename, ul)
        
        if returnTS:
            np.save(OUTPUT_DIR+filename+"_ts", ts)

        if verbosity:
            print("[Log] Upper limits are saved in '{}.npy'".format(OUTPUT_DIR+filename))

    return ul
    

def combinedExpectedUpperLimits(channel, runs=300, package="EventDisplay", 
    dwarfs = ["segue_1", "ursa_minor", "draco", "bootes"],
    addTheta=False, averagedIRF=True, method = 1, version="all", jSeed = "median", 
    filename = None, seed=0, overWrite=False, th2Cut=0, ext=False, sys=False,
    mass = np.logspace(2, 4.5, 15), bkgModel=None, verbosity=True, 
    returnTS = False, **kwargs):
    

    useBias = kwargs.get("useBias", True)

    if verbosity:
        print("="*50)
        print("[Log] Package      :", package)
        if len(dwarfs) == 1:
            print("[Log] Dwarf        :", dwarfs[0])
        else:
            print("[Log] # of Dwarfs  :", len(dwarfs))
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

        mass = mass4gamma[(mass4gamma>mass[0])*(mass4gamma<mass[-1])]

    DM_spectra = kwargs.get("DM_spectra", "PPPC")

    if DM_spectra == "HDM" and min(mass) < 1000:
        mass = np.asarray([1000]+mass[mass>1000].tolist())

    irf = {}
    tau = {}
    jProfile = {}
    singleIRF = averagedIRF

    if verbosity>1:
        print("[Log] Computing the J profile and reading IRFs ...       ")

    for dwarf in dwarfs:
        if jSeed == "median":
            jS = defaultNum[dwarf]
            allow_load=True
        else:
            jS = jSeed
            allow_load=False

        if package == "EventDisplay":
            if singleIRF:
                try:
                    importedIRF = ResponseFunction.EventDisplay.readIRFs(dwarf, version=version, ext=ext)
                    err = ResponseFunction.checkIRF(importedIRF)
                except:
                    err = True
                    
                if err:
                    importedIRF = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, export=False, ext=ext)

                tau[dwarf] = [1]
                irf[dwarf] = importedIRF
                jProfile[dwarf] = JProfile.generateConvolvedJ(dwarf, package, return_array=True, seed = jS, verbose=False, ext=ext, allow_load=allow_load, **kwargs)
            else:
                importedIRF = {}
                tau[dwarf] = []
                importedJProfile = {}
                for v in listOfVersions(dwarf):
                    try:
                        importedIRF[v] = ResponseFunction.EventDisplay.readIRFs(dwarf, version=v, ext=ext)
                        err = ResponseFunction.checkIRF(importedIRF[v])
                    except:
                        err = True    
                    if err:
                        importedIRF[v] = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=v, export=False, ext=ext)
                    importedJProfile[v] = JProfile.generateConvolvedJ(dwarf, package, irf = importedIRF[v], return_array=True, seed = jS, verbose=False, version=v,  ext=ext, **kwargs)
                    tau[dwarf].append(importedIRF[v].exposure)
                tau[dwarf] = np.asarray(tau[dwarf])/sum(tau[dwarf])
                irf[dwarf] = importedIRF
                jProfile[dwarf] = importedJProfile
        else:
            try:
                irf[dwarf] = ResponseFunction.VEGAS.readIRFs(dwarf)
            except:
                irf[dwarf] = ResponseFunction.VEGAS(dwarf, verbose=False)
            raw_events = vegas.readData(dwarf, rawdata=True)
            jProfile[dwarf] = JProfile.generateConvolvedJ(dwarf, package, irf = irf[dwarf], eLowerCut= min(raw_events[:,0]), return_array=True, seed = jS, verbose=False,  **kwargs)
            tau[dwarf] = [1]


    if verbosity>1:
        print("[Log] Initialization (Done)                      ")


    
    hSignal = {}
    for dwarf in dwarfs:
        if ext and (th2Cut == 0):
            th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
        else:
            th2Cut = defineTheta2Cut(package, th2Cut)
        hSignal[dwarf] = {}
        if package == "EventDisplay":
            raw_events = eventdisplay.readData(dwarf, rawdata=True, ext=ext)
        else:
            raw_events = vegas.readData(dwarf, rawdata=True)
        for i, M in enumerate(mass):
            if singleIRF:
                hSignal[dwarf][M] = calcSignal(dwarf, M, irf[dwarf], package, jProfile=jProfile[dwarf], channel=channel, addTheta=addTheta, th2Cut=th2Cut, version=version, verbose=False, ext=ext, eLowerCut=min(raw_events[:,0]), DM_spectra=DM_spectra)
                hSignal[dwarf][M].SetDirectory(0)
            else:
                hSignal[dwarf][M] = {}
                for v in listOfVersions(dwarf):
                    hSignal[dwarf][M][v] = calcSignal(dwarf, M, irf[dwarf][v], package, jProfile=jProfile[dwarf][v], channel=channel, addTheta=addTheta, th2Cut=th2Cut, version=version, verbose=False, ext=ext, eLowerCut=min(raw_events[:,0]), DM_spectra=DM_spectra)
                    hSignal[dwarf][M][v].SetDirectory(0)

    if verbosity>1:
        print("[Log] Start upper-limit calculation")

    ul = {}
    ts = {}
    pbar = tqdm(total=len(mass)*runs)

    for i, M in enumerate(mass):
        ul[M] = []
        ts[M] = []
        for r in range(runs):
            stackedMLE = {}
            for i, dwarf in enumerate(dwarfs):

                if ext and (th2Cut == 0):
                    th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
                else:
                    th2Cut = defineTheta2Cut(package, th2Cut)

                if package=="EventDisplay":
                    raw_events = eventdisplay.readData(dwarf, rawdata=True, ext=ext, version=version)
                elif package=="VEGAS":
                    raw_events = vegas.readData(dwarf, rawdata=True)
                else:
                    return
                
                bkg = raw_events[raw_events[:,2]==0.0]
                alpha = np.average(bkg[:,3])
                N_off = len(bkg)
                N_on = N_off*alpha
                
                N_on_poi = np.random.poisson(N_on)
                N_off_poi = np.random.poisson(N_off)

                if N_on_poi == 0:
                    N_on_poi = 1
                
                selected = np.random.choice(range(len(bkg)), size=N_on_poi)
                bkg_selected = np.random.randint(len(bkg), size=N_off_poi)

                events = bkg[selected]
                bkg_events = bkg[bkg_selected]

                events[:,2] = 1

                events = np.concatenate([events, bkg_events])

                if package=="EventDisplay":
                    hOn, hOff = eventdisplay.readData(dwarf, events=events, bkgModel=bkgModel, ext=ext)
                elif package=="VEGAS":
                    hOn, hOff = vegas.readData(dwarf, events=events, ext=ext)
                else:
                    hOn = None

                mle = MLE(dwarf, M, package, channel=channel, irf=irf[dwarf], jProfile=jProfile[dwarf], jArray=True,
                        th2Cut=th2Cut, addTheta=addTheta, ext=ext,
                        averagedIRF=averagedIRF, version=version, tau=tau[dwarf],
                        seed=i+20,  
                        bkgModel=bkgModel, 
                        N_on=N_on_poi, N_off = N_off_poi,
                        hOn = hOn, hOff=hOff,
                        expectedLimit=True, 
                        events = events, hSignal=hSignal[dwarf][M], 
                        verbose=(True if verbosity>2 else False), **kwargs) 
                stackedMLE[dwarf] = mle
                
            ul_i, ts_i = combinedMinuit(stackedMLE, signu0 = mle.signu0, channel=channel, verbose=max(verbosity-1, 0))
            
            if ul_i != -1 and ts_i != -1:
                ul[M].append(ul_i)
                ts[M].append(ts_i)
                if ts_i >=25:
                    if verbosity>1:
                        print("[Warning] TS value is higher than 25 (M={:.3f} TeV).".format(M/1000))
                    continue
            pbar.update(1)


        
    if verbosity>1:
        print("[Log] Expected upper-limit calculation (Done)                                           ")

    if addTheta:
        dim = "2D"
    else:
        dim = "1D"

    if filename == None:
        if not(os.path.isdir(OUTPUT_DIR)):
            os.system("mkdir "+OUTPUT_DIR)
        if bkgModel == None:
            bkg = "null"
        else:
            bkg = bkgModel
        if version =="all":
            filename = "{}_{}_{}_{}_{}_exp".format(package, "stacked", channel, dim, bkg)
        else:
            filename = "{}_{}_{}_{}_{}_{}_exp".format(package, "stacked", channel, dim, bkg, version)
    
    if overWrite:
        np.save(OUTPUT_DIR+filename, ul)
    else:
        os.system("cp "+OUTPUT_DIR+filename+".npy"+" "+OUTPUT_DIR+filename[:-4]+"_prev.npy")
        np.save(OUTPUT_DIR+filename, ul)
    
    if returnTS:
        np.save(OUTPUT_DIR+filename+"_ts", ts)

    if verbosity:
        print("[Log] Upper limits are saved in '{}.npy'".format(OUTPUT_DIR+filename))


def combinedStatisticUpperLimits(channel, runs=300, package="EventDisplay", dwarfs = ["segue_1", "ursa_minor", "draco", "bootes"],
    addTheta=False, averagedIRF=True, method = 1, version="all", jSeed = "median", 
    filename = None, seed=0, overWrite=False, th2Cut=0, ext=False, sys=False,
    mass = np.logspace(2, 4.5, 15), bkgModel=None, verbosity=True, 
    returnTS = False, **kwargs):
    
    addTheta = False

    if ext and (th2Cut == 0):
        th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
    else:
        th2Cut = defineTheta2Cut(package, th2Cut)
    useBias = kwargs.get("useBias", True)

    if verbosity:
        print("="*50)
        print("[Log] Package      :", package)
        if len(dwarfs) == 1:
            print("[Log] Dwarf        :", dwarfs[0])
        else:
            print("[Log] # of Dwarfs  :", len(dwarfs))
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

        mass = mass4gamma[(mass4gamma>mass[0])*(mass4gamma<mass[-1])]

    DM_spectra = kwargs.get("DM_spectra", "PPPC")

    if DM_spectra == "HDM" and min(mass) < 1000:
        mass = np.asarray([1000]+mass[mass>1000].tolist())

    irf = {}
    tau = {}
    jProfile = {}
    singleIRF = averagedIRF

    
    for dwarf in dwarfs:
        if jSeed == "median":
            jS = defaultNum[dwarf]
        else:
            jS = jSeed

        if package == "EventDisplay":
            if singleIRF:
                try:
                    importedIRF = ResponseFunction.EventDisplay.readIRFs(dwarf, version=version, ext=ext)
                    err = ResponseFunction.checkIRF(importedIRF)
                except:
                    err = True
                    
                if err:
                    importedIRF = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, export=False, ext=ext)

                tau[dwarf] = [1]
                irf[dwarf] = importedIRF
                jProfile[dwarf] = JProfile.generateConvolvedJ(dwarf, package, return_array=True, seed = jS, verbose=False, ext=ext, **kwargs)
            else:
                importedIRF = {}
                tau[dwarf] = []
                importedJProfile = {}
                for v in listOfVersions(dwarf):
                    try:
                        importedIRF[v] = ResponseFunction.EventDisplay.readIRFs(dwarf, version=v, ext=ext)
                        err = ResponseFunction.checkIRF(importedIRF[v])
                    except:
                        err = True    
                    if err:
                        importedIRF[v] = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=v, export=False, ext=ext)
                    importedJProfile[v] = JProfile.generateConvolvedJ(dwarf, package, irf = importedIRF[v], return_array=True, seed = jS, verbose=False, version=v, ext=ext, **kwargs)
                    tau[dwarf].append(importedIRF[v].exposure)
                tau[dwarf] = np.asarray(tau[dwarf])/sum(tau[dwarf])
                irf[dwarf] = importedIRF
                jProfile[dwarf] = importedJProfile
        else:
            try:
                irf[dwarf] = ResponseFunction.VEGAS.readIRFs(dwarf)
            except:
                irf[dwarf] = ResponseFunction.VEGAS(dwarf, verbose=False)
            raw_events = vegas.readData(dwarf, rawdata=True)
            jProfile[dwarf] = JProfile.generateConvolvedJ(dwarf, package, irf = irf[dwarf], eLowerCut= min(raw_events[:,0]), return_array=True, seed = jS, verbose=False, **kwargs)
            tau[dwarf] = [1]

    if verbosity>1:
        print("[Log] Initialization (Done)                      ")

    if verbosity>1:
        print("[Log] Start upper-limit calculation")
    
    
    hSignal = {}
    for dwarf in dwarfs:
        hSignal[dwarf] = {}
        for i, M in enumerate(mass):
            if singleIRF:
                hSignal[dwarf][M] = calcSignal(dwarf, M, irf[dwarf], package, jProfile=jProfile[dwarf], channel=channel, addTheta=addTheta, th2Cut=th2Cut, version=version, verbose=False, ext=ext, DM_spectra=DM_spectra)
                hSignal[dwarf][M].SetDirectory(0)
            else:
                hSignal[dwarf][M] = {}
                for v in listOfVersions(dwarf):
                    hSignal[dwarf][M][v] = calcSignal(dwarf, M, irf[dwarf][v], package, jProfile=jProfile[dwarf][v], channel=channel, addTheta=addTheta, th2Cut=th2Cut, version=version, verbose=False, ext=ext, DM_spectra=DM_spectra)
                    hSignal[dwarf][M][v].SetDirectory(0)


    ul = {}
    ts = {}
    for i, M in tqdm(enumerate(mass), total=len(mass)):
        ul[M] = []
        ts[M] = []
        for r in range(runs):
            stackedMLE = {}
            for i, dwarf in enumerate(dwarfs):

                if package=="EventDisplay":
                    raw_events = eventdisplay.readData(dwarf, rawdata=True, ext=ext, version=version)
                elif package=="VEGAS":
                    raw_events = vegas.readData(dwarf, rawdata=True)
                else:
                    return
                                
                N_on = sum(raw_events[:,2]==1.0)
                N_on_poi = np.random.poisson(N_on)

                hOn, etc = vegas.readData(dwarf)
                events = np.asarray([[hOn.GetRandom(), 0, 1, 1] for i in range(N_on_poi)])

                if package=="EventDisplay":
                    hOn, etc = eventdisplay.readData(dwarf, events=events)
                elif package=="VEGAS":
                    hOn, etc = vegas.readData(dwarf, events=events)
                else:
                    hOn = None

                mle = MLE(dwarf, M, package, channel=channel, irf=irf[dwarf], jProfile=jProfile[dwarf], jArray=True,
                        th2Cut=th2Cut, addTheta=addTheta, ext=ext,
                        averagedIRF=averagedIRF, version=version, tau=tau[dwarf],
                        seed=i+5,  
                        bkgModel=bkgModel, 
                        N_on=N_on_poi, expectedLimit=True,
                        events = events, hSignal=hSignal[dwarf][M], hOn = hOn,
                        verbose=(True if int(verbosity)>2 else False), **kwargs) 
                stackedMLE[dwarf] = mle
                

            ul_i, ts_i = combinedMinuit(stackedMLE, signu0 = mle.signu0, channel=channel, verbose=max(int(verbosity)-1, 0))
            
            if ul_i != -1 and ts_i != -1:
                ul[M].append(ul_i)
                ts[M].append(ts_i)
                if ts_i >=25:
                    if int(verbosity)>1:
                        print("[Warning] TS value is higher than 25 (M={:.3f} TeV).".format(M/1000))
                    continue
        
    if int(verbosity)>1:
        print("[Log] Expected upper-limit calculation (Done)                                           ")

    if addTheta:
        dim = "2D"
    else:
        dim = "1D"

    if filename == None:
        if not(os.path.isdir(OUTPUT_DIR)):
            os.system("mkdir "+OUTPUT_DIR)
        if bkgModel == None:
            bkg = "null"
        else:
            bkg = bkgModel
        if version =="all":
            filename = "{}_{}_{}_{}_{}_flc".format(package, "stacked", channel, dim, bkg)
        else:
            filename = "{}_{}_{}_{}_{}_{}_flc".format(package, "stacked", channel, dim, bkg, version)
    
    if overWrite:
        np.save(OUTPUT_DIR+filename, ul)
    else:
        os.system("cp "+OUTPUT_DIR+filename+".npy"+" "+OUTPUT_DIR+filename[:-4]+"_prev.npy")
        np.save(OUTPUT_DIR+filename, ul)
    
    if returnTS:
        np.save(OUTPUT_DIR+filename+"_ts", ts)

    if verbosity:
        print("[Log] Upper limits are saved in '{}.npy'".format(OUTPUT_DIR+filename))


def combinedMinuit(stackedMLE, channel = "tt", seed = 10, signu0 = -23, statistic="unbinned", verbose=True):


    if verbose == 2:
        printLevel = 1
    else:
        printLevel = -1
    
    Math.MinimizerOptions.SetDefaultMinimizer("Minuit");

    fit = TMinuit(2+len(stackedMLE))
    
    fit.mncler()

    fit.SetPrintLevel(printLevel=printLevel)
    
    # Parameter setting
    fit.mnrset(1)
    
    ierflg = ctypes.c_int(199)
    istat = ctypes.c_int(12)
    fit.mnparm(0,"signu", 1,     0.01,    -20,     15,  ierflg)
    fit.mnparm(1,"num", len(stackedMLE),   0,   len(stackedMLE),   len(stackedMLE),  ierflg)
    fit.FixParameter(1)

    stacked_logl0 = 0

    for i, dwarf in enumerate(stackedMLE):
        mle = stackedMLE[dwarf]
        b_null = (mle.N_on + mle.N_off)/(1.0 + mle.alpha)
        fit.mnparm(2+i,"b_{}".format(mle.dwarf), b_null,   0.1,   b_null*0.8,   b_null*1.2,  ierflg)
        stacked_logl0 += mle.nullHypothesis(b_null)

    for j, dwarf in enumerate(stackedMLE):
        mle = stackedMLE[dwarf]   
        fit.mnparm(len(stackedMLE)+2+j,"pn_{}".format(mle.dwarf), mle._pn,   0,   mle._pn,   mle._pn,  ierflg)
        fit.FixParameter(len(stackedMLE)+2+j)


    arglist = array( 'd', 10*[0.] )
    arglist[0] = 10000
    arglist[1] = 1.

    # Import a model
    if statistic == "unbinned":
        fit.SetFCN(stackedfcn)
    elif statistic == "binned":
        fit.SetFCN(stackedbinnedfcn)

    # Initial minimization
    fit.SetErrorDef(1)

    
    try:
        fit.mnexcm("MIGRAD", arglist, 2, ierflg);
    except:
        return -1, -1

    try:
        fit.mnmnos()
    except:
        return -1, -1
    
    if ierflg.value == 0:
        if verbose:
            print("[Log] MINUIT finds a minimum successfully (MIGRAD is converged).")

    elif ierflg.value == 4:
        for i in range(20):
            if verbose:
                print("[Warning] MIGRAD is NOT converged. Try again (trials: {}).".format(i+1), end="\r")
            fit.mnparm(0,"signu", (np.random.rand(1)[0]-0.5)*2,     0.1,    -20,     20,  ierflg)
            fit.mnexcm("MIGRAD", arglist, 2, ierflg);
            
            if ierflg.value == 0:
                break
        
        if ierflg.value == 4:
            print("[Error] MIGRAD is NOT converged. Check initial parameters (minuit in mle.py).")
            return np.nan, -1
        else:
            if verbose:
                print("[Log] MINUIT finds a minimum successfully (MIGRAD is converged).")

    else:
        print("[Error] An error occurs (type={}, https://root.cern.ch/doc/master/classTMinuit.html#ab48dd4b48800edc090396e35cb465fb9)".format(int(ierflg.value )))
    
    try:
        fit.mnimpr()
    except:
        return -1, -1
    
    logl, edm, errdef = map(ctypes.c_double, (0.18, 0.19, 0.20))
    nvpar, nparx, icstat = map(ctypes.c_int, (1983, 1984, 1985))
    signu_min, signu_err = map(ctypes.c_double, (0.40, 0.41))
    b_min, b_err = map(ctypes.c_double, (0.50, 0.51))

    fit.mnstat( logl, edm, errdef, nvpar, nparx, icstat )        
    fit.GetParameter(0, signu_min, signu_err)
    
    signu = (float(signu_min.value)+signu0, float(signu_err.value))
    flag = ierflg.value
    
    stacked_logl = logl.value
    
    stacked_ts = 2*(stacked_logl0-stacked_logl)
    
    signu_ul = np.nan
    
    # For calculating an upper limit
    fit.SetErrorDef(2.71/2)
    fit.mnexcm("MIGRAD", arglist, 2, ierflg);
    
    # Estimate an upper limit with profile likelihood
    istat = ctypes.c_int(12)

    sig_min = max((signu[0]-signu0), -1)
    
    fit.GetParameter(1, signu_min, signu_err)
    
    for i, dwarf in enumerate(stackedMLE):
        fit.GetParameter(2+i, b_min, b_err)
        fit.mnparm(2+i,"b_{}".format(dwarf), float(b_min.value),     0.01,  float(b_min.value)-3*float(b_err.value),    float(b_min.value)+3*float(b_err.value),  ierflg)

    if channel != "delta" or channel != "gamma":
        fit.mnparm(0,"signu", sig_min,     0.01,    sig_min-5,     sig_min+10,  ierflg)
    else:
        fit.mnparm(0,"signu", 1,     0.01,    sig_min-5,    sig_min+20,  ierflg)

    fit.mncomd("scan 1 100",istat)
    
    gLSignu = fit.GetPlot()
    
    x_signu, y_signu = getArray(gLSignu)
    
    gL = np.asarray([[x+signu0, y-stacked_logl] for x, y in zip(x_signu, y_signu)])
    aboveMax = (y_signu>min(y_signu))
    st_idx = list(aboveMax).index(False)

    valid = (y_signu[1:]<=y_signu[:-1])
    if sum(valid)>0:
        st_idx_2 = len(y_signu) - list(valid[::-1]).index(True)
        st_idx = max([st_idx, st_idx_2])

    try:
        logl = interp1d(gL[:,1][st_idx:], gL[:,0][st_idx:], kind='slinear')
        #print(gL[:,1][st_idx:], gL[:,0][st_idx:])
        signu_ul = logl(2.71/2)
        
        # plt.plot(gL[:,0], gL[:,1])
        # plt.axhline(2.71/2, color="r")
        # plt.axvline(signu_ul, color="r")
        # plt.ylim(-2, 30)
        # plt.grid()
        # plt.show(block=False)
        
    except:
        plt.plot(gL[:,0], gL[:,1])
        plt.axhline(2.71/2, color="r")
        plt.ylim(-2, 30)
        plt.grid()
        plt.show()
        print(gL[:,1], gL[:,0])
        return np.nan, -1
        #raise
    
    if verbose:
        print("[Log] An upper limit is estimated.")
    
    
    stacked_signu_ul = signu_ul
    return stacked_signu_ul, stacked_ts


