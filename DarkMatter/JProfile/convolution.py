import numpy as np
import os
import matplotlib.pyplot as plt

from ROOT import TFile, TMath
from ROOT import TGraph, TGraph2D

from .psf import *
from .profile import *

from .. import ResponseFunction

from .. import const
from ..utils import defineThetaCut, getArray

from tqdm.notebook import tqdm

from pyhank import qdht, iqdht
from scipy.interpolate import InterpolatedUnivariateSpline

from ..ResponseFunction.eventdisplay import th2cut_ext

from ..const import defaultNum

def generateConvolvedJ(dwarf, package="EventDisplay", filename = None, irf=None, gJProf=None, th2Cut=0, version="all", seed = -1, return_array=False, save_array=False, ext=False, step=0.004, verbose=True, **kwargs):
    
    if kwargs.pop("allow_load", False):
        if seed == defaultNum[dwarf]:
            if ext:
                filename = const.OUTPUT_DIR+"/JProfile_{}_{}_ext".format(package, dwarf)
            else:
                filename = const.OUTPUT_DIR+"/JProfile_{}_{}".format(package, dwarf)
            
            try:
                convJ1D_array = np.load(filename+"_1D.npy")
                convJ2D_array = np.load(filename+"_2D.npy", allow_pickle=True).item()

                return (convJ1D_array, convJ2D_array)
            except:
                save_array=True
                pass

    # Read IRFs
    if irf is not None:
        if hasattr(irf, "package"):
            if irf.package is not None:
                if irf.package != package:
                    if verbose: 
                        print("[Warning] IRFs and package are mismatched. The package name is changed from {} to {}.".format(package, irf.package))
                    package=irf.package
    else:
        if package=="VEGAS":
            energies = const.eVJbins
            try:
                irf = ResponseFunction.VEGAS.readIRFs(dwarf, verbose=False)
            except:
                irf = ResponseFunction.VEGAS(dwarf, mode = 3, verbose=False)

        elif package=="EventDisplay":
            try:
                irf = ResponseFunction.EventDisplay.readIRFs(dwarf, version=version, ext=ext)
                err = ResponseFunction.checkIRF(irf)
                if err:
                    irf = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, ext=ext)
            except:
                irf = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, ext=ext)    
        
    PSF = irf.PSF

    # Read J profile
    if gJProf is None:
        gJProf = calcJProfile(dwarf, seed=seed, step=step, **kwargs)

    # Define energies
    if package=="EventDisplay":
        energies = const.eEDJbins
    elif package=="VEGAS":
        energies = const.eVJbins
    
    minE, maxE = findMinMaxE(PSF)

    # Define theta_cut
    if ext and th2Cut == 0:
        thCut = defineThetaCut(package, th2cut_ext(dwarf, ext=ext))
    else:
        thCut = defineThetaCut(package, th2Cut)

    # # Make ROOT stuffs
    # if save_root:
    #     convJ1D = TGraph()
    #     convJ2D = TGraph2D()

    #     convJ1D.SetTitle("Convolved J profile (seed: {})".format(seed))
    #     convJ1D.GetHistogram().GetXaxis().SetTitle("Energy [GeV]");
    #     convJ1D.GetHistogram().GetYaxis().SetTitle(r"J (#theta^{{2}} < {:.3f}) [GeV^{{2}} cm^{{-5}}]".format(thCut**2.));

    #     convJ2D.SetTitle("Convolved 2D J profile (seed: {})".format(seed))
    #     convJ2D.GetHistogram().GetXaxis().SetTitle("Energy [GeV]")
    #     convJ2D.GetHistogram().GetYaxis().SetTitle("Theta [Deg]")

    
    convJ1D_array = []
    convJ2D_array = {}
    m=0
    n=0
    th_width = step
    
    for i, en in enumerate(tqdm(energies) if verbose else energies):
        if int((en/10**minE)*100)<99 or (en >= 10**maxE):
            continue
        
        J, J2D = calcHKConvJProf2D(gJProf, PSF, en=en, package=package, ext=ext, th2Cut=th2Cut, dwarf=dwarf)
        
        convJ1D_array.append([en, J])
        
        convJ2D_array[en] = J2D[J2D[:,0]<=(thCut+th_width)]

        # if save_root:
        #     convJ1D.SetPoint(m, en, J)
        #     m+=1

        #     for j2d in J2D:
        #         if j2d[0]<= thCut:
        #             convJ2D.SetPoint(n, round(np.log10(en), 1), j2d[0], j2d[1])
        #             n+=1
        #         else:
        #             convJ2D.SetPoint(n, round(np.log10(en), 1), j2d[0], j2d[1])
        #             n+=1
        #             break

    convJ1D_array = np.asarray(convJ1D_array)

    if save_array:
        if ext:
            filename = const.OUTPUT_DIR+"/JProfile_{}_{}_ext".format(package, dwarf)
        else:
            filename = const.OUTPUT_DIR+"/JProfile_{}_{}".format(package, dwarf)

        np.save(filename+"_1D", convJ1D_array)
        np.save(filename+"_2D", convJ2D_array)
        if verbose: 
            print("[Log] Finish. J profile is saved in {}_XD.npy.".format(filename))

  
    if return_array:
        return (convJ1D_array, convJ2D_array)

def calcHKConvJProf2D(gJProf, PSF, en=None, step=0.004, package="EventDisplay", th2Cut=0, ext=False, dwarf = None, verbose=False):
    
    if ext and (th2Cut == 0):
        thCut = defineThetaCut(package, th2cut_ext(dwarf=dwarf, ext=ext))
    else:
        thCut = defineThetaCut(package, th2Cut)

    th = gJProf[:,0]
    dth = np.diff(th)[0]
    th_rad = th*TMath.DegToRad()
    dth_rad = np.diff(th_rad)[0]

    if type(PSF) == TH2D:
        if en is None:
            raise
        else:
            gPSF1D = getPSF1D(PSF, en, package=package, return_array=True, step=step)
    elif type(PSF) == TGraph:
        gPSF1D = np.asarray(getArray(PSF)).T
        gPSF1D[:,1] *= TMath.RadToDeg()**2.


    h_dJ = gJProf[:,1]
    h_psf = gPSF1D[:,1]
    h_kr, ht_psf=qdht(th_rad, h_psf)      # forward Hankel Transform of PSF
    h_kr, ht_dJ=qdht(th_rad, h_dJ)      # forward Hankel Transform of J factor

    # convolve
    ht_conv=ht_psf*ht_dJ                      # Multiplication in Fourier space = Convolution 
    h_th_rad, h_conv=iqdht(h_kr, ht_conv);    # inverse Hankel Transform

    h_dJth_conv = h_conv*2*np.pi*np.sin(th_rad)*dth_rad
    h_dJth_conv = np.nan_to_num(h_dJth_conv)
    
    dJdOmega = h_dJth_conv/(2*np.pi*np.sin(th_rad)*dth_rad)
    dJdOmega = np.asarray([th, dJdOmega]).T

    if sum(h_dJth_conv)>0:
        spl = InterpolatedUnivariateSpline(th, h_dJth_conv/dth)
        J = spl.integral(0, thCut)
        J = np.nan_to_num(J)
    else:
        J = 0

    if verbose:
        print("[Log] Convolved J profile in {:.3f} TeV: {:.2e}".format(en/1000., J))
        
    return J, dJdOmega

def convert2Dto1D(J2D, package="EventDisplay", th2Cut=0, ext=False, th_ran=[0,0], dwarf=None):
    if ext and (th2Cut == 0):
        thCut = defineThetaCut(package, th2cut_ext(dwarf=dwarf, ext=ext))
    else:
        thCut = defineThetaCut(package, th2Cut)

    th = J2D[:,0]
    dth = np.diff(th)[0]
    th_rad = th*TMath.DegToRad()
    dth_rad = np.diff(th_rad)[0]
    h_dJth_conv = J2D[:,1]*(2*np.pi*np.sin(th_rad)*dth_rad)
    if sum(h_dJth_conv)>0:
        spl = InterpolatedUnivariateSpline(th, h_dJth_conv/dth)
        if (th_ran[0] != th_ran[1]):
            J = spl.integral(th_ran[0], th_ran[1])
        else:
            J = spl.integral(0, thCut)
        J = np.nan_to_num(J)
        return J
    else:
        return 0

def convert2Dto1D_multi(J2D, package="EventDisplay", th2Cut=0, ext=False, axis="theta", dwarf=None):
    if ext and (th2Cut == 0):
        thCut = defineThetaCut(package, th2cut_ext(dwarf=dwarf,ext=ext))
    else:
        thCut = defineThetaCut(package, th2Cut)

    J_tot = []
    if axis=="theta":
        if type(J2D) is dict:
            energies = list(J2D.keys())
            energies.sort()
            theta = J2D[energies[0]][:,0]
            for en in energies:
                selected_J2D = J2D[en]
                J = convert2Dto1D(selected_J2D, package=package, th2Cut = th2Cut, ext=ext, dwarf=dwarf)
                J_tot.append(J)
        elif J2D.Class_Name() == "TGraph2D":
            J2D, energies, theta = getArray(J2D)
            energies = list(set(energies))
            energies.sort()
            for en in energies:
                selected_J2D = J2D[J2D[:,1]==en]
                J = convert2Dto1D(selected_J2D[:,[2,0]], package=package, th2Cut = th2Cut, ext=ext, dwarf=dwarf)
                J_tot.append(J)
        elif J2D.Class_Name() == "TH2D":
            J2D, energies, theta = getArray(J2D)
            for i, en in enumerate(energies):
                selected_J2D = J2D[:,i]
                J = convert2Dto1D(np.asarray([theta, selected_J2D]).T, package=package, th2Cut = th2Cut, ext=ext, dwarf=dwarf)
                J_tot.append(J)

        return np.asarray(J_tot), energies
        
    elif axis == "energy":
        J2D, energies, theta = getArray(J2D)

        for th in theta:
            J2D = J2D[J2D[:,2]==th]
            #J = convert2Dto1D(J2D[:,[0,2]], package=package, thCut = thCut)
            #J_tot.append(J)
        #return np.asarray(J_tot), energies


def checkSanity(file, name1="gConvJ1D", name2="gConvJ2D", package="EventDisplay", ext=False, dwarf=None):
    File = TFile(file, "READ")

    J1D_x, J1D_y = getArray(File.Get(name1))
    raw_J2D = File.Get(name2)
    z, x, y = getArray(raw_J2D)
    if y[-1] == np.sqrt(th2cut_ext(dwarf,ext=ext)):
        ext = True
    if "ext" in file:
        ext=True

    J2D_y, J2D_x = convert2Dto1D_multi(raw_J2D, package=package, ext=ext, dwarf=dwarf)

    f, ax = plt.subplots(2,1, figsize=(7, 7), gridspec_kw={'height_ratios':[5,1]})
    ax[0].step(J1D_x, J1D_y, where="mid", label="1D (ED, soft)")
    ax[0].step(J1D_x, J2D_y, where="mid", label="2D (ED, soft)")
    ax[0].axvline(J1D_x[7])
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"Convolved J profile", fontsize=15)
    ax[0].legend(fontsize=12, loc=4)
    ax[0].grid()

    ax[1].plot(J1D_x[:-1], np.nan_to_num(abs(J1D_y[:-1]-J2D_y[:-1])/J2D_y[:-1]*100), label="2D/1D (soft)")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Energy [GeV]", fontsize=15)
    ax[1].set_ylabel("Ratio [%]", fontsize=15)
    ax[1].set_ylim(0, 1)
    ax[1].axhline(0, color="k", ls="--")
    ax[1].grid()


