import numpy as np

import os

from ROOT import TFile, TH1D, TH2D, TMath

from .spectra import readSpectrum, PPPCspectra, HDMspectra, gridInterpolation

from .. import ResponseFunction
from .. import const
from ..const import HDM_Channel2Num
from .. import JProfile
from ..utils import getArray, defineTheta2Cut, printRunList, thetaEdges

from scipy.interpolate import interp1d, interp2d
import random

from tqdm.notebook import tqdm

def forwardFolding(model, pars, dwarf, package="EventDisplay", irf=None, useBias=True, verbose=False, alpha=1, **kwargs):

    if verbose:
        print("[Log] Importing the IRFs ({}).".format(dwarf))

    if irf is None:
        if package=="EventDisplay":
            irf = ResponseFunction.EventDisplay.readIRFs(dwarf)
        elif package == "VEGAS":
            irf = ResponseFunction.VEGAS.readIRFs(dwarf)

    else:
        if hasattr(irf, "package"):
            if irf.package != package:
                if verbose: 
                    print("[Warning] IRFs and package are mismatched. The package name is changed from {} to {}.".format(package, irf.package))
                package=irf.package

    gEA = irf.EA              # Effective area
    
    if useBias and hasattr(irf, "Bias"):
        hDisp = irf.Bias         # Energy bias
    else:
        useBias = False
        hDisp = irf.Edisp         # Energy dispersion
    
    nfactor = irf.exposure*alpha     # Exposure time

    if package=="EventDisplay":
        eBinEdges = kwargs.pop("energyEdges", const.energyEdges)
        
    elif package=="VEGAS":
        eBinEdges = kwargs.pop("energyEdges", const.eVJbins)
    
    eBinEdges = kwargs.pop("eBinEdges", eBinEdges)

    hg_1d = TH1D("hg_1D", "hg_1D", len(eBinEdges)-1, eBinEdges)
    hg_1d.SetTitle("1D count spectrum")
    hg_1d.GetXaxis().SetTitle("Energy [GeV]")
    hg_1d.GetYaxis().SetTitle("Counts")

    if verbose:
        print("[Log] Generating the signal spectrum.")

    for i in range(1, hDisp.GetNbinsX()+1):
    
        if useBias:
            Etr = 10**(hDisp.GetXaxis().GetBinCenter(i)+3)
            dEtr = 10**(hDisp.GetXaxis().GetBinUpEdge(i)+3)-10**(hDisp.GetXaxis().GetBinLowEdge(i)+3)
        else:
            Etr = hDisp.GetXaxis().GetBinCenter(i)
            dEtr = hDisp.GetXaxis().GetBinWidth(i)

        # Effective Area
        Elog10TeV = np.log10(Etr/1000.0)
        
        if gEA.Class_Name() == "TH1D":
            A = gEA.Interpolate(Elog10TeV)  
        else:
            A = gEA.Eval(Elog10TeV)         
        
        A *= 1e4
        if A <= 0:
            continue
        
        dNdE = model(Etr, *pars)
        
        norm = 0

        for j in range(1, hg_1d.GetNbinsX()+1):
            E = hg_1d.GetXaxis().GetBinCenter(j)
            dE = hg_1d.GetXaxis().GetBinWidth(j)

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


        for j in range(1, hg_1d.GetNbinsX()+1):
            E = hg_1d.GetXaxis().GetBinCenter(j)
            dE = hg_1d.GetXaxis().GetBinWidth(j)

            if useBias:
                ratio = E/Etr
                if ratio > 3:
                    continue
                D = hDisp.Interpolate(np.log10(Etr)-3, ratio)
            else:
                D = hDisp.Interpolate(Etr, E)
            
            if D<0:
                continue

        
            Sum = nfactor*A*dNdE*D*dEtr*dE/norm
            
            hg_1d.Fill(E, Sum)
        
        
    if verbose:
        print("[Log] Done.")
                
    hg_1d.SetDirectory(0)
    return hg_1d
