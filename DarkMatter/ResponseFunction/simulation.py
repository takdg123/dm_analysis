import numpy as np
from ROOT import TGraph, TH2D
from ..utils import gaus
from .. import const
from ..utils import newirf as fakeirf

def fakePSF(sigma, step=0.004):
    d = np.arange(step/2., 2, step=step)

    psf =  gaus(d, 0, sigma)* 4 / np.size(d) / (2*np.pi*d*step)

    gTest = TGraph()
    gTest.SetTitle("PSF (fake)")
    for i, (th, p) in enumerate(zip(d, psf)):
        gTest.SetPoint(i, th, p)

    return gTest

def fakeEA(vals, benchmark=None, energies = np.arange(-2, 6, step=0.2), low_cutoff = 0, high_cutoff = 0, s=1):

    gTest = TGraph()
    gTest.SetTitle("EA (fake)")
    gTest.SetName("EffectiveArea")
    gTest.GetXaxis().SetTitle("log10 Energy [TeV]")
    gTest.GetYaxis().SetTitle("Effective area [m^2]")
    
    if benchmark == "EventDisplay":
        energies = np.arange(-2, 6, step=0.2)
    elif benchmark == "VEGAS":
        energies = np.log10(const.eVJbins)-3

    if np.size(vals) == 1:
        if high_cutoff == 0:
            EA = np.exp(-(low_cutoff/10**(energies))**s)*vals
        else:
            EA = np.exp(-(low_cutoff/10**(energies))**s)*vals*np.exp(-(10**(energies)/high_cutoff)**s)
    else:
        energies = vals[:,0]
        EA = vals[:,1]
    for i, e in enumerate(energies):
        gTest.SetPoint(i, e, EA[i])
        
    return gTest

def fakeBias(sigma, benchmark=None, energies = np.arange(-2, 6, step=0.2)):

    if benchmark == "EventDisplay":
        energies = np.arange(-2, 6, step=0.2)
    elif benchmark == "VEGAS":
        energies = np.log10(const.eVJbins)-3

    ratio = np.arange(0, 3.01, step=0.04)
    bias_arr = gaus(ratio, 1, sigma)
    hBias = TH2D("hBias", "Energy Migration (fake)", len(energies)-1, energies, len(ratio)-1, ratio)
    hBias.GetXaxis().SetTitle("log10 True Energy [GeV]");
    hBias.GetYaxis().SetTitle("Ratio (Erec/Etr)");

    for i in range(len(ratio)-1):
        for j in range(len(energies)-1):
            hBias.SetBinContent(j+1, i+1, bias_arr[i])
    return hBias

def fakeIRFs(PSF, EA, Bias, exposure, benchmark=None, energies=np.arange(-2, 6, step=0.2), cutoff=None):
    fake = fakeirf()
    fake.PSF = fakePSF(PSF)
    fake.EA = fakeEA(EA, energies=energies, cutoff=cutoff, benchmark=benchmark)
    fake.Bias = fakeBias(Bias, energies=energies, benchmark=benchmark)
    fake.exposure = exposure
    return fake

def fakeDisp(etr, erec, matrix):

    hBias = TH2D("DispersionMatrix", "DispersionMatrix (fake)", len(etr)-1, etr, len(erec)-1, erec)
    hBias.GetXaxis().SetTitle("log10 True Energy [GeV]");
    hBias.GetYaxis().SetTitle("Ratio (Erec/Etr)");

    for i in range(len(etr)-1):
        for j in range(len(erec)-1):
            hBias.SetBinContent(i+1, j+1, matrix[i][j])
    return hBias


