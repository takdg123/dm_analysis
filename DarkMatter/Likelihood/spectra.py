import numpy as np

import os

from ROOT import TGraph2D, TH2D

from pathlib import Path

from ..const import SCRIPT_DIR, PPPC_Channel2Num, HDM_Channel2Num

from scipy.interpolate import RectBivariateSpline

from ..utils import getArray

from HDMSpectra import HDMSpectra

# Read PPPC DM file (AtProduction_gammas1.dat)
def readSpectrum(channel="tt", data = SCRIPT_DIR+"/external/PPPCSpectra/AtProduction_gammas.dat", plotting=False):
    channel_list = PPPC_Channel2Num.keys()
    
    if channel not in channel_list:
        print("[Error] Channel type is a wrong")
        raise
    else:
        index = PPPC_Channel2Num[channel]

    gSpec = TGraph2D()
    gSpec.SetTitle("PPPC DM spectra ({})".format(channel))

    with open(data) as f:
        j = 0
        for line in f.readlines()[1:]:
            m = float(line.split()[0])
            x = float(line.split()[1])
            val = float(line.split()[index])

            gSpec.SetPoint(int(j), x, m, val)
            j+=1
            
    gSpec.GetHistogram().GetXaxis().SetTitle("log_{10} x")
    gSpec.GetHistogram().GetYaxis().SetTitle("M_{#chi} [GeV]")
    return gSpec

def PPPCspectra(channel, x_list, M, PPPC=None, data = SCRIPT_DIR+"/external/PPPCSpectra/AtProduction_gammas.dat", return_dNdx=False, useScipy = True):
    
    if PPPC is None:
        PPPC = readSpectrum(channel, data=data)
    elif type(PPPC) == RectBivariateSpline:
        useScipy = True
    
    if M > 100000: # 100 TeV
        return np.zeros(np.size(x_list))

    if np.size(x_list) == 1:
        if abs(x_list-1.0) < 1e-8:
            x_list = np.asarray([1.0])
        else:
            x_list = np.asarray([x_list])
    
    if useScipy:
        if (type(PPPC) != RectBivariateSpline):
            PPPC = gridInterpolation(PPPC=PPPC, channel=channel, data=data)

        dNdlog10x = PPPC(np.log10(x_list), M)[::,0]
        dNdx = dNdlog10x*np.log10(np.exp(1))/x_list
        dNdx[x_list>1] = 0
        dNdx[dNdx<=0] = 0

    else:
        dNdx = []
        for x in x_list:
            if abs(x-1.0) < 1e-8:
                dNdlog10x = PPPC.Interpolate(0, M)
            elif x > 1.0:
                dNdlog10x = 0
            else:
                dNdlog10x = PPPC.Interpolate(np.log10(x), M)
                
        
            if dNdlog10x <= 0:
                dNdlog10x = 0

            dNdx.append(dNdlog10x*np.log10(np.exp(1))/x)
        
    dNdx = np.asarray(dNdx)

    dNdE = dNdx/M

    if return_dNdx:
        return dNdx
    else:
        return dNdE

def HDMspectra(channel, x_list, M, data = SCRIPT_DIR+"/external/HDMSpectra/data/HDMSpectra.hdf5", return_dNdx=False, neutrino=False):
    if neutrino:
        finalstate = 12   # photon
    else:
        finalstate = 22   # photon
    initialstate = HDM_Channel2Num[channel]

    if M/2 < 500:
        return np.zeros(len(x_list)), 0
    
    if np.size(x_list) == 1:
        if abs(x_list-1.0) < 1e-5:
            x_list = np.asarray([1.0])
        else:
            x_list = np.asarray([x_list])

    else:
        valid = (x_list <=1)*(x_list >=1e-6)

    dNdx = np.zeros(len(x_list))
    if channel == "gamma" or channel == "ZZ":
        
        temp = HDMSpectra.spec(finalstate, initialstate, x_list[valid], M/2., data = data, annihilation=True, delta=True)
        
        cont = temp[:-1]
        dNdx[valid] = cont

        if 1.0 in x_list:
            delta = temp[-1]
        else:
            delta = 0

    else:
        cont = HDMSpectra.spec(finalstate, initialstate, x_list[valid], M/2., data = data, annihilation=True)
        dNdx[valid] = cont
        delta = 0

    dNdx = np.asarray(dNdx)
    dNdE = dNdx/M

    if return_dNdx:
        return dNdx, delta
    else:
        return dNdE, delta


def gridInterpolation(PPPC = None, channel=None, data=SCRIPT_DIR+"/external/PPPCSpectra/AtProduction_gammas.dat"):
    if PPPC is None:
        PPPC = readSpectrum(channel, data=data)
        
    z, x, y = getArray(PPPC)
    Ms = list(set(y))
    Ms.sort()
    Ms = np.asarray(Ms)
    xs = list(set(x))
    xs.sort()
    xs = np.asarray(xs)

    output = []
    for x in xs:
        fz = z[z[:,1]==x]
        if (Ms == fz[:,2]).all():
            output.append(fz[:,0])
    output = np.asarray(output)    

    PPPC = RectBivariateSpline(xs, Ms, output, )
    return PPPC