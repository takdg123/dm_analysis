import numpy as np

import os

from ROOT import TFile, TH1D, TH2D, TMath

from .spectra import readSpectrum, PPPCspectra, HDMspectra, WINOspectra, gridInterpolation

from .. import ResponseFunction
from .. import const, utils
from ..const import HDM_Channel2Num
from .. import JProfile
from ..utils import getArray, defineTheta2Cut, printRunList, thetaEdges

from scipy.interpolate import interp1d, interp2d
import random

from tqdm.notebook import tqdm

from ..ResponseFunction.eventdisplay import th2cut_ext

def calcSignal(dwarf, M, irf, package="EventDisplay", DM_spectra="PPPC", 
    jProfile = None, jArray = True, jSeed = -1, channel="tt", ext=False,
    th2Cut=0, eLowerCut=None, eUpperCut = None, addTheta=False, 
    sigma=-23, version="all", ideal=False, normDisp=False, useBias=True, 
    verbose=False, useScipy = True,  **kwargs):
    
    if ext and (th2Cut == 0):
        th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
    else:
        th2Cut = defineTheta2Cut(package, th2Cut)

    if verbose:
        print("[Log] Importing the IRFs ({}, {}, {}).".format(dwarf, package, version))
    
    gEA = irf.EA              # Effective area
    
    if hasattr(irf, "package"):
        if irf.package is not None:
            if irf.package != package:
                if verbose: 
                    print("[Warning] IRFs and package are mismatched. The package name is changed from {} to {}.".format(package, irf.package))
                package=irf.package


    if useBias and hasattr(irf, "Bias"):
        hDisp = irf.Bias         # Energy bias
        z, x, y = getArray(hDisp, return_edges=True)
        if eLowerCut == None:
            eLowerCut = min(10**(x+3))
        if eUpperCut == None:
            eUpperCut = max(10**(x+3))
    else:
        useBias = False
        hDisp = irf.Edisp         # Energy dispersion
        z, x, y = getArray(hDisp, return_edges=True)
        if eLowerCut == None:
            eLowerCut = min(x)
        if eUpperCut == None:
            eUpperCut = max(x)
    eUpperCut = min(1e5, eUpperCut)

    t_exp = irf.exposure      # Exposure time

    # Signal spectrum
    if verbose:
        print("[Log] Importing the DM spectrum (channel: {}).".format(channel))
    
    if channel == "delta":
        PPPC_spec = None
    elif DM_spectra == "PPPC" and useScipy:
        PPPC_spec = gridInterpolation(channel=channel)
    elif DM_spectra=="PPPC":
        PPPC_spec = readSpectrum(channel)

    apply_factor = kwargs.pop("apply_factor", False)

    if apply_factor:
        factor = const.ratio_factor[dwarf]
    else:
        factor = 1

    nfactor = t_exp/(8*np.pi*pow(M, 2))*10**sigma*factor

    # J profile
    if jArray:
        if type(jProfile) == interp1d:
            jProfile_1d = jProfile
        elif type(jProfile) == interp2d:
            jProfile_2d = jProfile
        else:
            if jProfile is None:
                if jSeed == -1:
                    if kwargs.pop("general", False):
                        jSeed = random.randrange(0, JProfile.goodPropNum(dwarf)-1)
                    else:
                        jSeed = random.randrange(0, 100000)
                if verbose:
                    print("[Log] Importing the J profile (seed: {}).".format(jSeed))
                (J1, J2) = JProfile.generateConvolvedJ(dwarf, package, irf=irf, version=version, return_array=True, seed = jSeed, verbose=False, th2Cut=th2Cut, ext=ext, **kwargs)
            else:
                (J1, J2) = jProfile

            eLowerCut = max(min(J1[:,0]), eLowerCut)
            eUpperCut = min(max(J1[:,0]), eUpperCut)

            jProfile_1d = interp1d(J1[:,0], J1[:,1])

            en = list(J2.keys())
            th = J2[en[0]][:,0]
            z = []
            for e in en:
                z+=[list(J2[e][:,1])]
            z = np.asarray(z).T
            jProfile_2d = interp2d(np.log10(en), th, z, kind="quintic")
    else:
        loadFile = False
        if jProfile is None:
            filename = const.OUTPUT_DIR+"/JProfile_{}_{}.root".format(package, dwarf)
            loadFile = True
        elif type(jProfile) == str:
            filename = jProfile
            loadFile = True
        else:
            loadFile = False

        if loadFile:
            if verbose:
                print("[Log] Importing the J profile (file: {}).".format(filename))

            JFile = TFile(filename, "READ")

            if version == "all":
                jProfile_1d = JFile.Get("gConvJ1D")
                jProfile_2d = JFile.Get("gConvJ2D")

                if verbose:
                    print("[Log]", jProfile_1d.GetTitle(), "is imported.")
                    print("[Log]", jProfile_2d.GetTitle(), "is imported.")
            else:
                jProfile_1d = JFile.Get("gConvJ1D_{}".format(version))
                jProfile_2d = JFile.Get("gConvJ2D_{}".format(version))
        else:
            if addTheta:
                jProfile_2d = jProfile
                
            else:
                jProfile_1d = jProfile
                
        if addTheta:
            jz, jx, jy = getArray(jProfile_2d)
            z = []
            en = list(set(jx))
            en.sort()
            th = list(set(jy))
            th.sort()
            for e in en:
                z.append(jz[jz[:,1] == e][:,0])
            z = np.asarray(z).T
            
            jProfile_2d = interp2d(en, th, z, kind="quintic")
        else:
            jx, jy = getArray(jProfile_1d)

            jProfile_1d = interp1d(jx, jy)

        
    if package=="EventDisplay":
        if ext:
            eBinEdges = kwargs.get("energyEdges", np.logspace(1, 7, 101))
        else:
            eBinEdges = kwargs.get("energyEdges", const.energyEdges)
        
    elif package=="VEGAS":
        eBinEdges = kwargs.pop("energyEdges", const.eVJbins)
    
    if not(useBias) and not(ideal):
        z, etr, erec = getArray(hDisp, return_edges=True)
        if len(erec) != len(eBinEdges):
            print("[Warning] The energy bin edges may be wrong [Error 1]. Define energyEdges.")
        elif sum(erec == eBinEdges) != len(erec):
            print("[Warning] The energy bin edges may be wrong [Error 2]. Define energyEdges.")

    tBinEdges = thetaEdges(th2Cut)
    width = np.diff(tBinEdges)

    hg_1d = TH1D("hg_1D", "hg_1D", len(eBinEdges)-1, eBinEdges)
    hg_1d.SetTitle("1D count spectrum (N_{"+str(sigma)+"})")
    hg_1d.GetXaxis().SetTitle("Energy [GeV]")
    hg_1d.GetYaxis().SetTitle("Counts")
    
    if addTheta: 
        hg_2d = TH2D("hg_2D","hg_2D", len(eBinEdges)-1, eBinEdges, len(tBinEdges)-1, tBinEdges)
        hg_2d.SetTitle("2D count spectrum (N_{"+str(sigma)+"})")
        hg_2d.GetXaxis().SetTitle("Energy [GeV]")
        hg_2d.GetYaxis().SetTitle("Theta2 [deg^2]")
        hg_2d.GetZaxis().SetTitle("Counts")


    if verbose:
        print("[Log] Generating the signal spectrum.")

    
    for i in range(1, hDisp.GetNbinsX()+1) if not(ideal) else range(1, hg_1d.GetNbinsX()+1):
        if ideal:
            Etr = hg_1d.GetXaxis().GetBinCenter(i)
            Etr_u = hg_1d.GetXaxis().GetBinUpEdge(i)
            Etr_l = hg_1d.GetXaxis().GetBinLowEdge(i)
            dEtr = Etr_u-Etr_l
            if Etr < 70:
                continue
        else:
            if useBias:
                Etr = 10**(hDisp.GetXaxis().GetBinCenter(i)+3)
                Etr_u = 10**(hDisp.GetXaxis().GetBinUpEdge(i)+3)
                Etr_l = 10**(hDisp.GetXaxis().GetBinLowEdge(i)+3)
                dEtr = Etr_u-Etr_l

            else:
                Etr = hDisp.GetXaxis().GetBinCenter(i)
                Etr_u = hDisp.GetXaxis().GetBinUpEdge(i)
                Etr_l = hDisp.GetXaxis().GetBinLowEdge(i)
                dEtr = Etr_u-Etr_l

        if Etr < 70:
            continue
        # Effective Area
        Elog10TeV = np.log10(Etr/1000.0)
        
        if gEA.Class_Name() == "TH1D":
            A = gEA.Interpolate(Elog10TeV)  
        else:
            A = gEA.Eval(Elog10TeV)         
        
        A *= 1e4        
        if A <= 0:
            continue

        # Signal spectrum
        x = Etr/M

        if channel == "delta":
            if abs(x -1.0) < 0.01:
                dNdE = 2./dEtr
            else:
                continue
        else:
            if (Etr>eUpperCut):
                dNdE = 0
                continue
            elif x < 1.001 and x>=1e-6:
                x_u = Etr_u/M
                x_l = Etr_l/M
                if x_u > 1:
                    x_u = 1
                x_list = np.linspace(x_l, x_u, 100)
                dx = np.diff(x_list)

                if channel == "wino" or DM_spectra == "WINO":
                    dNdE = WINOspectra(x=x, M=M)

                elif DM_spectra == "PPPC":
                    dNdE = PPPCspectra(channel, x_list, M,  PPPC=PPPC_spec, useScipy=useScipy)
                    dNdE = (sum(utils.center_pt(dNdE)*dx)/(x_u-x_l))

                    #dNdE = PPPCspectra(channel, x, M,  PPPC=PPPC_spec, useScipy=useScipy)[0]
                    
                elif DM_spectra == "HDM":

                    dNdE, delta = HDMspectra(channel, x_list, M)
                    dNdE = (sum(utils.center_pt(dNdE)*dx)/(x_u-x_l))
                    
                    #dNdE, delta = HDMspectra(channel, x, M)
                    dNdE += delta/dEtr

                if dNdE < 0:
                    dNdE = 0

            else:
                dNdE = 0
                continue
        
        if ideal:
            norm = 1
        else:
            norm = 0
            
            for j in range(1, hg_1d.GetNbinsX()+1):
                E = hg_1d.GetXaxis().GetBinCenter(j)
                dE = hg_1d.GetXaxis().GetBinWidth(j)
                E_u = hg_1d.GetXaxis().GetBinUpEdge(j)
                E_l = hg_1d.GetXaxis().GetBinLowEdge(j)

                if E_u < eLowerCut:
                    continue
                elif E_l > eUpperCut:
                    continue

                if useBias:
                    ratio = E/Etr
                    if ratio > 3:
                        continue
                    
                    D = hDisp.Interpolate(np.log10(Etr)-3, ratio)
                else:
                    
                    D = hDisp.Interpolate(Etr, E)
                

                if D>0:
                    norm +=D*dE
            
            if norm == 0:
                continue
            elif (abs(norm-1) > 0.05) and (normDisp):
                print("[Error] Normalization error at {:.2f} TeV. norm = {:.2f}. Recommend to set normDisp = False.".format(Etr, norm))
        
        for j in range(1, hg_1d.GetNbinsX()+1):
            E = hg_1d.GetXaxis().GetBinCenter(j)
            dE = hg_1d.GetXaxis().GetBinWidth(j)
            E_u = hg_1d.GetXaxis().GetBinUpEdge(j)
            E_l = hg_1d.GetXaxis().GetBinLowEdge(j)


            if E_u < eLowerCut:
                continue
            elif E_l > eUpperCut:
                continue

            if ideal:
                if Etr == E:
                    D = 1/dE
                else:
                    D = 0
                    continue
            else:
                if useBias:
                    ratio = E/Etr
                    if ratio > 3:
                        continue
                    D = hDisp.Interpolate(np.log10(Etr)-3, ratio)
                else:
                    D = hDisp.Interpolate(Etr, E)
                
                if D<0:
                    continue

            # J profile

            th = np.linspace(0.002, np.sqrt(th2Cut), 1000)

            for k in range(1, len(tBinEdges)):

                th2 = (tBinEdges[k]+tBinEdges[k-1])/2.
                if addTheta:
                    j_temp = jProfile_2d(np.log10(Etr), th)[::,0]
                    if th2 < th2Cut:
                        j_E = JProfile.convert2Dto1D(np.asarray([th, j_temp]).T, package=package, ext=ext, th_ran=[np.sqrt(tBinEdges[k-1]), np.sqrt(tBinEdges[k])])
                    else:
                        j_E = 0
                        continue
                else:

                    j_E = jProfile_1d(Etr)
                    
                
                # Multiply all (dN/dE' J(E') A(E') D(E|E') dE')
                if j_E > 0 and not(np.isnan(j_E)):  
                    Sum = nfactor*A*j_E*dNdE*D*dEtr*dE/norm
                    if addTheta:
                        hg_2d.Fill(E, th2, Sum)
                    else:
                        hg_1d.Fill(E, Sum)
                        break
        
        
    if verbose:
        print("[Log] Done.")
                
    if addTheta:
        hg_2d.SetDirectory(0)
        return hg_2d
    else:
        hg_1d.SetDirectory(0)
        return hg_1d

def combinedCalcSignal(dwarf, M, package="EventDisplay", DM_spectra="PPPC", irf=None, jProfile=None, jArray=True, jSeed = -1, 
    th2Cut = 0, sigma=-23, channel="tt", runbyrun = False, addTheta=False, 
    verbose=False, ideal=False, eLowerCut=None, version="all", normDisp=False, 
    useBias=True, ext=False, **kwargs):
    if ext and (th2Cut == 0):
        th2Cut = defineTheta2Cut(package, th2cut_ext(dwarf=dwarf, ext=ext))
    else:
        th2Cut = defineTheta2Cut(package, th2Cut)

    if runbyrun:
        hg = {}
        print("[Error] This function is not ready.")
    else:
        if irf ==None:
            try:
                irf = ResponseFunction.EventDisplay.readIRFs(dwarf, version=version, norm=normDisp, ext=ext)
            except:
                irf = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, norm=normDisp, ext=ext)
        
        hg = calcSignal(dwarf, M, irf, package, jProfile = jProfile, DM_spectra=DM_spectra, th2Cut = th2Cut, channel=channel, sigma=sigma, addTheta=addTheta, ideal=ideal, eLowerCut=eLowerCut, version=version, normDisp=normDisp, useBias=useBias, jArray=jArray, jSeed = jSeed, verbose=verbose, ext=ext, **kwargs)
        hg.SetDirectory(0)

        return hg

def th2toth(th1, th2):
    th2_l = np.sqrt(th1)
    th2_u = np.sqrt(th2)
    th_edge = np.arange(th2_l, th2_u, step=0.0001)
    th = (th_edge[1:]+th_edge[:-1])/2.
    dth = th_edge[1:]-th_edge[:-1]
    return th, dth

