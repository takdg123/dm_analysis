import numpy as np

from scipy import stats

from ROOT import TH1D, TH2D

from .. import const

from ..utils import getArray, defineTheta2Cut

from .. import Likelihood

def convert2Dsignal(h, countSpectrum=True):
    binEdges = []
    for i in range(1, h.GetNbinsX()+1):
        binEdges.append(h.GetXaxis().GetBinLowEdge(i))
    binEdges.append(h.GetXaxis().GetBinUpEdge(i))    
    binEdges=np.asarray(binEdges)
    
    hg = TH1D("hg", "hg_1D_from_2D", len(binEdges)-1, binEdges)
    for i in range(1, h.GetNbinsX()+1):
        if countSpectrum:
            val = 0
            for j in range(1, h.GetNbinsY()+1):
                val += h.GetBinContent(i, j)
        else:
            val = 0
            for j in range(1, h.GetNbinsY()+1):
                val += h.GetBinContent(i, j)*h.GetYaxis().GetBinWidth(j)
        hg.SetBinContent(i, val)
    return hg

def check2Dsignal(h1d, h2d, countSpectrum = True, chi=False):
    h2to1d = convert2Dsignal(h2d, countSpectrum=countSpectrum)
    nBins_1d = h1d.GetNbinsX()
    nBins_2d = h2to1d.GetNbinsX()
    if chi:
        x, y = getArray(h2to1d)
        x0, y0 = getArray(h1d)

        y0_v = y0[(y0!=0)]
        y_v = y[(y0!=0)]

        if max(y_v)<1 or max(y0_v)<1:
            print("[Error] Chi squre may not work.")
        else:
            chi = sum((y0_v-y_v)**2/y0_v)
            dof = len(y0_v)
            if chi/dof < 2:
                print(u"[Log] \u03C7\u00b2/dof is {:.2f}/{:.0f}. They are consistent".format(chi, dof))
            else:
                print("[Error] The 2D signal is needed to be checked.")
    else:
        ratio = abs(h1d.Integral(0, nBins_1d+1)-h2to1d.Integral(0, nBins_2d+1))/h1d.Integral(0, nBins_1d+1)
       
        if ratio < 0.05:
            print("[Log] Total signal ratio is {:.2f}%. They are consistent".format(ratio*100))
        else:
            print("[Error] The 2D signal is needed to be checked.")

def fakeit(dwarf, M, sigma, package, jProfile = None, bkgOnly=False, irf=None, addTheta=False, channel="tt", verbose=False):
    if package=="EventDisplay":
        events = __fakeit_ED__(dwarf, M, jProfile, sigma, bkgOnly=bkgOnly, irf=irf, addTheta=addTheta, channel=channel)
    elif package=="VEGAS":
        events = __fakeit_VEGAS__(dwarf, M, jProfile, sigma, irf=irf, channel=channel)
    else:
        print("[Error] The package name is not specified.")
        raise
    return events

def __fakeit_VEGAS__(dwarf, M, sigma, jProfile = None, bkgOnly=False, irf=None, channel="tt", verbose=False):
    thCut = defineTheta2Cut("VEGAS", 0)
    if irf==None:
        hg_1d = Likelihood.combinedCalcSignal(dwarf, M, "VEGAS", jProfile=jProfile, channel=channel, sigma=sigma)
    else:
        hg_1d = Likelihood.calcSignal(dwarf, M, irf, "VEGAS", jProfile=jProfile, channel=channel, sigma=sigma)
    hOn, hOff = Likelihood.vegas.readData(dwarf)
    x_off, y_off = getArray(hOff)
    x_s, y_s = getArray(hg_1d)
    x_on = x_off
    y_on = y_off + y_s
    N_on = int(sum(y_on))
    onRegion = stats.rv_discrete(name='onRegion', values=(x_on, y_on/sum(y_on)))
    events = onRegion.rvs(size=N_on)
    return events

def __fakeit_ED__(dwarf, M, sigma, jProfile=None, bkgOnly=False, irf=None, addTheta=False, channel="tt", verbose=False):
    thCut = defineTheta2Cut("EventDisplay", 0)

    if not(bkgOnly):
        if irf==None:
            hg_2d = Likelihood.combinedCalcSignal(dwarf, M, "EventDisplay", jProfile=jProfile, thCut = thCut, channel=channel, sigma=sigma, addTheta=addTheta, averagedIRF=True)
        else:
            hg_2d = Likelihood.calcSignal(dwarf, M, irf, "EventDisplay", jProfile=jProfile, thCut = thCut, channel=channel, sigma=sigma, addTheta=addTheta)

    hOn_2d, hOff_2d, n1, n2, evts, alpha = Likelihood.eventdisplay.readData(dwarf, thCut=thCut, addTheta=addTheta, full_output=True)
    N_tot = 0
    N_bkg = 0
    N_sig = 0
    hOn_2d.SetTitle("Signal+Background")
    for i in range(1, hOn_2d.GetNbinsX()+1):
        for j in range(1, hOn_2d.GetNbinsY()+1):
            N_bkg += hOff_2d.GetBinContent(i, j)
            if bkgOnly:
                total = hOff_2d.GetBinContent(i, j)
                N_sig = 0
            else:
                if hOff_2d.GetXaxis().GetBinCenter(i) != hg_2d.GetXaxis().GetBinCenter(i):
                    print("[Warning] Check the energy-bin definition.")
                    continue
                if hOff_2d.GetYaxis().GetBinCenter(j) != hg_2d.GetYaxis().GetBinCenter(j):
                    print("[Warning] Check the theta-bin definition.")
                    continue
                N_bkg += hOff_2d.GetBinContent(i, j)
                N_sig += hg_2d.GetBinContent(i, j)
                total = hOff_2d.GetBinContent(i, j)+hg_2d.GetBinContent(i, j)
            hOn_2d.SetBinContent(i, j, total)
            N_tot+=total
    
    events_1D = []
    events_2D = []
    for j in range(1, hOn_2d.GetNbinsY()+1):
        x_on = []
        y_on = []
        
        for i in range(1, hOn_2d.GetNbinsX()+1):
            x_on.append(hOn_2d.GetXaxis().GetBinCenter(i))
            y_on.append(hOn_2d.GetBinContent(i, j))
        x_on = np.asarray(x_on)  
        y_on = np.asarray(y_on)
        
        if int(sum(y_on)) >=1:
            onRegion = stats.rv_discrete(name='onRegion', values=(x_on, y_on/sum(y_on)))
            events_1D += (onRegion.rvs(size=round(sum(y_on)))).tolist()
    events_1D = np.asarray(events_1D)

    if addTheta:
        for E in events_1D:
            x_on = []
            y_on = []
            for i in range(1, hOn_2d.GetNbinsY()+1):
                th = hOn_2d.GetYaxis().GetBinCenter(i)
                x_on.append(th)
                y_on.append(hOn_2d.Interpolate(E, th))
            
            x_on = np.asarray(x_on)  
            y_on = np.asarray(y_on)
            onRegion = stats.rv_discrete(name='onRegion', values=(x_on*1000, y_on/sum(y_on)))
            th = (onRegion.rvs(size=1)/1000)[0]
            events_2D.append([E, th])

    events_2D = np.asarray(events_2D)

    if verbose: 
        print("{} events are generated.".format(int(N_tot)))

    if addTheta:
        return events_2D
    else:
        return events_1D