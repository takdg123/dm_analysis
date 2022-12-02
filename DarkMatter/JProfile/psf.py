import numpy as np

import os

from ROOT import TFile, TMath
from ROOT import TCanvas, gPad
from ROOT import TGraph, TGraph2D, TH1D, TH2D

from ..utils import center_pt
from .. import const
from ..utils import getArray

# Get 1D PSF
def getPSF1D(h, en, step=0.004, package="EventDisplay", verbose = False, check=False, return_array = False):
# INPUT
#    h: PSF loaded by readPSFFile, which is not normalized (dP/dr = 2*pi*r dP/dOmega)
#    en: energy in GeV
# OUTPUT
#    gTh1D_nm: normalized PSF, (x, y) = (theta, dP/dOmega)
    gTh1D_nm = TGraph()

    norm = 0
    Ntot = 0
    gTh1DPoints = []

    thknots = np.asarray([[th, step] for th in np.arange(step/2, 2, step=step)])
    
    psf_array = []

    for i, (th, dth) in enumerate(thknots):
        
        if package=="VEGAS":
            N = h.Interpolate(np.log10(en/1000.0), th)
            gTh1DPoints.append([i, th, N/th])
            norm += N*dth
        elif package=="EventDisplay":
            N = h.Interpolate(np.log10(en/1000.0), np.log10(th))
            gTh1DPoints.append([i, th, N/th**2.])
            norm += N*dth/th

        Ntot += N

    norm *= 2*np.pi

    for point in gTh1DPoints:
        if return_array:
            if norm == 0:
                psf_array.append([point[1], 0])
            else:
                psf_array.append([point[1], point[2]/norm*TMath.RadToDeg()**2.])
        else:
            gTh1D_nm.SetPoint(point[0], point[1], point[2]/norm)

    if Ntot < 1000.0 and verbose:
        print("[Warning] Low number of events! ")

    if check:
        hCheckNorm = TH2D("hCheckNorm","hCheckNorm", 1000, -2, 2, 1000, -2, 2)
        hCheckNorm.SetStats(0)
        Sum = []
        for i in range(hCheckNorm.GetNbinsX()):
            for j in range(hCheckNorm.GetNbinsY()):
                x = hCheckNorm.GetXaxis().GetBinCenter(i)
                y = hCheckNorm.GetYaxis().GetBinCenter(j)
                dx = hCheckNorm.GetXaxis().GetBinWidth(i)
                dy = hCheckNorm.GetYaxis().GetBinWidth(j)
                r = np.sqrt(x*x+y*y)
                psf = gTh1D_nm.Eval(r)
                hCheckNorm.SetBinContent(i, j, psf)
                Sum.append(psf*dx*dy)

        print("[Log] Getting PSF with energy of: ", en, "GeV")
        print("[Log] PSF Integral from +/- 2 deg: ", sum(Sum))

    gTh1D_nm.GetXaxis().SetTitle("Theta [deg]")
    gTh1D_nm.GetYaxis().SetTitle(r"dP/d$\Omega$")

    if return_array:
        return np.asarray(psf_array)
    else:
        return gTh1D_nm

def getPSFcont(h, step = 0.004, en=None, package="EventDisplay"):
    if type(h) == TH2D:
        if en is None:
            raise
        else:
            gTh1D = getPSF1D(h, en, package=package, step=step)
    elif type(h) == TGraph:
        gTh1D = h

    PSF_cont = [[0, 0]]
    ths = np.linspace(0, 2, 501)
    ths_ctr = center_pt(ths)
    ths_w = np.diff(ths)
    for i, th in enumerate(ths_ctr):
        psf = gTh1D.Eval(th)*2*np.pi*np.sin(th*TMath.DegToRad())*ths_w[i]/TMath.DegToRad()
        PSF_cont.append([th, psf+PSF_cont[-1][1]])
        
    PSF_cont=np.asarray(PSF_cont).T
    return tuple(PSF_cont)
    
def findMinMaxE(PSF):
    z, x, y = getArray(PSF, return_edges=True)
    minE = min(x)
    maxE = max(x)
    minF = False
    for i in range(PSF.GetNbinsX()):
        tot = PSF.Integral(i+1, i+1, 0, -1)
        if tot == 0:
            if minF:
                maxE = PSF.GetXaxis().GetBinCenter(i+1)
                break
            else:
                minE = PSF.GetXaxis().GetBinCenter(i+1)
        else:
            minF = True
    

    return round(minE, 1)+3, round(maxE, 1)+3

