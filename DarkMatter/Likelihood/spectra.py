import numpy as np

import os

from ROOT import TGraph2D, TH2D

from pathlib import Path

from ..const import REF_DIR, SCRIPT_DIR, PPPC_Channel2Num, HDM_Channel2Num

from scipy.interpolate import RectBivariateSpline

from ..utils import getArray

from HDMSpectra import HDMSpectra

from astropy.table import Table

from scipy.interpolate import interp1d, interp2d

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

def WINOspectra(x=None, M=None, return_table=False):
    tab = Table(np.load(REF_DIR+"wino_dnde.npy"))
    if return_table:
        return tab

    return get_spectra_from_table(x, M, tab)

def Qspectra(x=None, M=None, return_table=False):

    tab = Table(np.load(REF_DIR+"quintuplet_dnde.npy"))
    if return_table:
        return tab

    return get_spectra_from_table(x, M, tab)


def COSMIXspectra(channel, x_list, M, return_dNdx=False):
    from ..external.COSMIXs.Interpolate import Interpolate

    if channel == "ee":
        channel = "e"
    elif channel == "tt":
        channel = "tau"
    
    spec = Interpolate(M, channel, "Gamma")

    s = spec.make_spectrum()
    s["dNdx"] = s["dNdLog10x"]*np.log10(np.exp(1))/10**s["Log10[x]"]
    s["dNdE"] = s["dNdx"]/M

    if return_dNdx:
        interp = interp1d(s["Log10[x]"], s["dNdx"], fill_value="extrapolate")
        return interp(np.log10(x_list))
    else:
        interp = interp1d(s["Log10[x]"], s["dNdE"], fill_value="extrapolate")
        return interp(np.log10(x_list))


def PPPCspectra(channel, x_list, M, PPPC=None, data = SCRIPT_DIR+"/external/PPPCSpectra/AtProduction_gammas.dat", return_dNdx=False, useScipy = True):
    
    if PPPC is None:
        PPPC = readSpectrum(channel, data=data)
    elif type(PPPC) == RectBivariateSpline:
        useScipy = True
    
    if M > 100000: # 100 TeV
        return np.zeros(np.size(x_list))

    x_list = np.atleast_1d(x_list)

    if np.size(x_list) == 1:
        if abs(x_list[0]-1.0) < 1e-8:
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
        
        temp = HDMSpectra.spec(finalstate, initialstate, x_list[valid], M, data = data, annihilation=True, delta=True)
        
        cont = temp[:-1]
        dNdx[valid] = cont

        if 1.0 in x_list:
            delta = temp[-1]
        else:
            delta = 0

    else:
        cont = HDMSpectra.spec(finalstate, initialstate, x_list[valid], M, data = data, annihilation=True)
        dNdx[valid] = cont
        delta = 0

    dNdx = np.asarray(dNdx)
    dNdE = dNdx/M

    if return_dNdx:
        return dNdx, delta
    else:
        return dNdE, delta

def get_spectra_from_table(x, M, tab):
    x = np.atleast_1d(x)
    if (np.size(x) == 1) and (abs(x[0]-1.0) < 1e-8):
        log10x = np.asarray([0])
    else:
        log10x = np.atleast_1d(np.log10(x))

    include_delta = (0 in log10x)

    if M in list(tab["mass"]):
        tab = tab[tab["mass"]==M]
        spectra = interp1d(np.log10(tab["x"]), tab["dNdE"])
        
        # spectra = interp1d(np.log10(tab["x"]), tab["dNdE_endpoint"])
        # dnde = spectra(log10x)
        #dnde = dnde - dnde2
        #dnde = np.zeros(len(log10x))
        dnde = spectra(log10x)

        if include_delta:
            dnde[-1] = 0
            delta = tab["dNdE"][-1]
        else:
            
            delta = 0

        if len(log10x) == 1:
            return dnde[0], delta
        else:
            return dnde, delta
    else:
        # Unique sorted axes
        spectra, delta_spectra = regularGridInterpolation(tab)

        if np.size(x)==np.size(M):
            dNdE = np.nan_to_num(spectra((log10x, M)))
            if include_delta:
                dNdE[-1] = 0
                delta = delta_spectra(M)
            else:
                delta = 0
        else:
            M = np.ones(len(log10x))*M
            dNdE = np.nan_to_num(spectra(np.asarray([log10x, M]).T))
            if include_delta:
                dNdE[-1] = 0
                delta = delta_spectra(M[0])
            else:
                delta = 0
        return dNdE, delta

def regularGridInterpolation(tab, remove_delta = True):
    from scipy.interpolate import RegularGridInterpolator
    x = np.asarray(tab['x'])
    mass = np.asarray(tab['mass'])
    dNdE = np.asarray(tab['dNdE'])

    # Compute log10(x)
    logx = np.log10(x)

    # Get sorted unique values (grid axes)
    x_vals = np.unique(logx)
    
    mass_vals = np.unique(mass)

    # Create mapping from (mass, logx) to indices in the grid
    mass_idx = {val: i for i, val in enumerate(mass_vals)}
    x_idx = {val: i for i, val in enumerate(x_vals)}

    # Initialize z-grid
    z = np.full((len(x_vals), len(mass_vals)), np.nan)

    # Vectorized index calculation
    ix = np.array([mass_idx[m] for m in mass])
    iy = np.array([x_idx[lx] for lx in logx])

    # Fill the grid
    z[iy, ix] = dNdE

    spectra = RegularGridInterpolator((x_vals, mass_vals), z)
    
    new_tab = tab[tab['x'] == 1]
    delta_spectra = interp1d(new_tab["mass"], new_tab["dNdE"])

    return spectra, delta_spectra

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