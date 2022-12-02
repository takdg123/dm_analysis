import numpy as np
import ctypes
from scipy import stats
from ..const import defaultNum
from .eventdisplay import readData
from .signal import combinedCalcSignal
from ..utils import getArray

from scipy.interpolate import interp1d

from ROOT import TH1D

def generateFakeEvents(dwarf, M, sigma, irf=None, addTheta=True, decay="tt", ext=False, jSeed=None, poisson=False, **kwargs):

    if jSeed is None:
        jSeed = defaultNum[dwarf]
    hOn, hOff, n1, n2, evts, alpha = readData(dwarf, addTheta=addTheta, 
                                            ext=ext, full_output=True,
                                            **kwargs)
    hSignal = combinedCalcSignal(dwarf, M, "EventDisplay", decay=decay, irf=irf, jSeed=jSeed, eLowerCut=min(evts[:,0]),
            sigma=sigma, addTheta=addTheta, ext=ext, averagedIRF=True, **kwargs)
    
    if addTheta:
        events = gen_evt_2d(hSignal, hOff, poisson=poisson)
    else:
        events = gen_evt_1d(hSignal, hOff, poisson=poisson)

    return events

def fake_evt_set(N, hSignal, events, minE = np.nan, **kwargs):

    sig = gen_evt_1d(hSignal, minE = minE, **kwargs).tolist()
    bkg = events[events[:,2]==0][:,0]
    N_off = int(N-len(sig))
    selected = np.random.choice(len(bkg), size=N_off)
    bkg_selected = bkg[selected]
    for e in bkg_selected:
        sig.append([e])
    return np.asarray(sig)

def gen_evt_1d(hSignal, hOff=None, poisson=False, alpha=1, ROOT=False, minE = np.nan):

    if type(hSignal) == np.ndarray:
        xs, ys = hSignal[:,0], hSignal[:,1]
    else:
        xs, ys = getArray(hSignal)
    if hOff is not None:
        xb, yb = getArray(hOff)
    else:
        xb = xs
        yb = ys
        ys = np.zeros(len(xs))
    
    x_on = xb[yb>0]
    y_on = (yb + ys)[yb>0]
    N_on = int(sum(y_on)/alpha)

    if ROOT:
        edges = getArray(hSignal, return_edges=True)[1]
        h_samp = TH1D("sampling", "sampling", len(edges)-1, edges)
        for i, (b, s) in enumerate(zip(yb,ys)):
            if b >0:
                h_samp.SetBinContent(i+1, b+s)
            else:
                h_samp.SetBinContent(i+1, 0)

        if poisson:
            N_on = np.random.poisson(N_on)
        
        events = np.asarray([[h_samp.GetRandom()] for i in range(N_on*1000)])
        events = events[events[:,0]>minE]
        selected = np.random.choice(len(events), size=N_on)
        events = events[selected]
    else:
        int_s = interp1d(np.log10(x_on[y_on>0]), np.log10(y_on[y_on>0]), kind=2, bounds_error=False, fill_value="extrapolate")
        fit_x = np.linspace(min(np.log10(x_on[y_on>0])), max(np.log10(x_on[y_on>0])), 1000)
        fit_y = 10**int_s(fit_x)

        if poisson:
            N_on = np.random.poisson(N_on)

        fit_y = np.nan_to_num(fit_y)
    
        onRegion = stats.rv_discrete(name='onRegion', values=(10**(fit_x+3), fit_y/sum(fit_y)))
        events = onRegion.rvs(size=N_on*100)
        events = events[events>minE*1000]
        selected = np.random.choice(len(events), size=N_on)
        events = events[selected]
        events = np.asarray([[evt/1000] for evt in events])
    return events

def gen_evt_2d(hSignal, hOff, poisson=False):

    zs, xs, ys = getArray(hSignal)
    zb, xb, yb = getArray(hOff, return_edges=True)
    N_bkg = round(np.sum(zb))
    N_sig = round(sum(sum(zs)[sum(zb)>0]))
    N_on = N_bkg + N_sig
    ratio = N_sig/N_on
    minE = xb[:-1][sum(zb)>0][0]

    if poisson:
        N_on = np.random.poisson(N_on)
        N_bkg = round(N_on * (1-ratio))
        N_sig = round(N_on * ratio)

    e, th = map(ctypes.c_double, (0.18, 0.20))
    event_s = []
    while True:
        hSignal.GetRandom2(e, th)
        if e > minE:
            event_s.append([float(e.value), float(th.value)])
        if len(event_s) == round(N_sig):
            break
    event_s = np.asarray(event_s)

    z, x, y = getArray(hOff)
    engergies = np.sum(z, axis=0)

    int_s = interp1d(np.log10(x[engergies>0]), np.log10(engergies[engergies>0]), kind=1)
    fit_x = np.linspace(min(np.log10(x[engergies>0])), max(np.log10(x[engergies>0])), 1000)
    fit_y = 10**int_s(np.log10(fit_x))

    e = stats.rv_discrete(name='e', values=(10**(fit_x+3), fit_y/sum(fit_y)))
    evt_energies = e.rvs(size=round(N_bkg)*100)
    selected = np.random.choice(len(evt_energies), size=round(N_bkg))
    evt_energies = evt_energies[selected]

    thetas = np.sum(z, axis=1)

    int_s = interp1d(y[thetas>0], thetas[thetas>0], kind=1)
    fit_x = np.linspace(min(y[thetas>0]), max(y[thetas>0]), 1000)
    fit_y = int_s(fit_x)

    th = stats.rv_discrete(name='th', values=(fit_x*1e8, fit_y/sum(fit_y)))
    evt_thetas = th.rvs(size=round(N_bkg))

    event_bkg = np.asarray([evt_energies/1e3, evt_thetas/1e8]).T
    events = np.concatenate((event_bkg, event_s), axis=0)

    return events
