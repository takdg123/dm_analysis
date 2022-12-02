import numpy as np

import matplotlib.pyplot as plt

import os

import ctypes

from astropy import units as u
from astropy.coordinates import SkyCoord

from ROOT import TFile, TTree, TH1D, TH2D, TMath

from .spectra import readSpectrum

from .fcn import fcn

from ..utils import getArray, defineThetaCut

from .. import const
from .. import ResponseFunction

from array import array


DM_DIR = os.environ.get('DM')
REF_DIR = DM_DIR+"/RefData/"
DATA_DIR = DM_DIR+"/Data/"

# Read Event File
def initialize():
    for dwarf in const.ListOfDwarf:
        createEventFile(dwarf)
        
def createEventFile(dwarf):
    
    f = TFile(DATA_DIR+'VEGAS_EventFile_{}.root'.format(dwarf), 'RECREATE')
    
    t = TTree("eventTree","eventTree")

    if dwarf == "Segue_1":
        ra_dw = 15.0*(10.0 + 7.0/60 + 4.0/3600)
        dec_dw = (16.0 + 4.0/60 + 55.0/3600)
        path = REF_DIR+"/Pass5f/segue1_eventList_pass5f_wZnCorr.txt"
    elif dwarf == "Draco":
        ra_dw = 15.0*(17.0 + 20.0/60 + 12.4/3600)
        dec_dw = (57.0 + 54.0/60 + 55.0/3600)
        path = REF_DIR+"/Pass5f/{}_eventList_pass5f_wZnCorr.txt".format(dwarf)
    elif dwarf == "UMi":
        ra_dw = 15.0*(15.0 + 9.0/60 + 8.5/3600)
        dec_dw = (67.0 + 13.0/60 + 21.0/3600)
        path = REF_DIR+"/Pass5f/umi_eventList_pass5f_wZnCorr.txt".format(dwarf)
    elif dwarf == "Bootes_I":
        ra_dw = 15.0*(14.0 + 0.0/60 + 6/3600)
        dec_dw = (14.0 + 30.0/60 + 0.0/3600)
        path = REF_DIR+"/Pass5f/{}_eventList_pass5f_wZnCorr.txt".format(dwarf)

    with open(path) as f_temp:
        InputData = []
        for line in f_temp.readlines()[2:]:
            InputData.append(line.split())
    InputData = np.asarray(InputData)

    ra = InputData[:,3].astype("float")
    dec = InputData[:,4].astype("float")
    c1 = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    c2 = SkyCoord(ra=ra_dw*u.degree, dec=dec_dw*u.degree, frame='icrs')
    thetaList = c1.separation(c2).deg

    runNum = array( 'i', [0] )
    runLT = array( 'd', [0.] )
    time = array( 'd', [0.] )
    ra = array( 'd', [0.] )
    dec = array( 'd', [0.] )
    isOn = array( 'i', [0] )
    w = array( 'd', [0.] )
    energy = array( 'd', [0.] )
    theta = array( 'd', [0.] )

    t.Branch("runNum",runNum,"runNum/I")
    t.Branch("runLT",runLT,"runLT/D")
    t.Branch("time",time,"time/D")
    t.Branch("ra",ra,"ra/D")
    t.Branch("dec",dec,"dec/D")
    t.Branch("isOn",isOn,"isOn/O")
    t.Branch("w",w,"w/D")
    t.Branch("energy",energy,"energy/D")
    t.Branch("theta",theta,"theta/D")

    for data, th in zip(InputData, thetaList):
        data = data.astype("float")
        runNum[0] = int(data[0])
        runLT[0] = float(data[1])
        time[0] = float(data[2])
        ra[0] = float(data[3])
        dec[0] = float(data[4])
        isOn[0] = int(data[8])
        w[0] = float(data[9])
        energy[0] = float(data[7])
        theta[0] = th
        t.Fill()
        
    t.Write()
    f.Close()

def readData(dwarf, events = [], thCut = 0.17, rawdata=False, addTheta=False, getRuns = False, eLowerCut=100, eUpperCut = 1e5, full_output=False, **kwargs):
    
    eBinEdges = const.eVJbins

    hOn = TH1D("hOn", "hOn_1D", len(eBinEdges)-1, eBinEdges)
    hOn.SetTitle("Count spectrum (on region)")
    hOn.GetXaxis().SetTitle("Energy [GeV]")
    hOn.GetYaxis().SetTitle("Counts")
    
    hOff = TH1D("hOff", "hOff_1D", len(eBinEdges)-1, eBinEdges)
    hOff.SetTitle("Count spectrum (off region)")
    hOff.GetXaxis().SetTitle("Energy [GeV]")
    hOff.GetYaxis().SetTitle("Counts")

    if addTheta:
        tBinEdges = np.linspace(0, thCut, 25)
        
        hOn_2d = TH2D("hOn","hOn_2D", len(eBinEdges)-1, eBinEdges, len(tBinEdges)-1, tBinEdges)
        hOn_2d.SetTitle("2D count spectrum (on region)")
        hOn_2d.GetXaxis().SetTitle("Energy [GeV]")
        hOn_2d.GetYaxis().SetTitle("Theta [deg]")
        hOn_2d.GetZaxis().SetTitle("Counts")

        print("[Warning] VEGAS data does not contain theta information for off-region events.")
        hOff_2d = TH2D("hOff","hOff_2D", len(eBinEdges)-1, eBinEdges, len(tBinEdges)-1, tBinEdges)
        hOff_2d.SetTitle("2D count spectrum (off region)")
        hOff_2d.GetXaxis().SetTitle("Energy [GeV]")
        hOff_2d.GetYaxis().SetTitle("Theta [deg]")
        hOff_2d.GetZaxis().SetTitle("Counts")
    

    if len(events)==0:
        if dwarf == "Segue_1":
            path = "VEGAS_Segue_1_events.npy"
        elif dwarf == "Draco":
            path = "VEGAS_Draco_events.npy"
        elif dwarf == "UMi":
            path = "VEGAS_UMi_events.npy"
        elif dwarf == "Bootes_I":
            path = "VEGAS_Bootes_I_events.npy"

        events = np.load(DATA_DIR+path)
        # File = TFile(DATA_DIR+"VEGAS_EventFile_{}.root".format(dwarf), 'READ')
        # tEv = File.eventTree
        # events = []
        # runs = []
        # w = []
        # Noff = 0
        # Non = 0
        # for i in range(tEv.GetEntries()):
        #     tEv.GetEntry(i)

        #     if getRuns:
        #         if np.size(runs) == 0:
        #             runs.append([tEv.runNum, tEv.runLT])
        #         elif tEv.runNum not in np.asarray(runs)[:,0]:
        #             runs.append([tEv.runNum, tEv.runLT])
        
        #     if tEv.isOn:
        #         events.append([tEv.energy, tEv.theta, 1, 1])
        #         if addTheta:
        #             hOn_2d.Fill(tEv.energy, tEv.theta)
        #         else:
        #             hOn.Fill(tEv.energy)
        #         Non+=1
        #     else:
        #         events.append([tEv.energy, tEv.theta, 0, tEv.w])
        #         w.append(tEv.w)
        #         Noff+=1
        #         if addTheta:
        #             hOff_2d.Fill(tEv.energy, tEv.theta, tEv.w)
        #         else:
        #             hOff.Fill(tEv.energy, tEv.w)

        # events = np.asarray(events)
        # w_avg = sum(w)/Noff
        # np.save(DATA_DIR+path, events)
        

    w = []
    Non = 0
    Noff = 0
    for evt in events:
        energy = evt[0]
        theta2 = evt[1]
        isOn = evt[2]
        alpha = evt[3]
    
        if isOn == 1.:
            Non += 1
            if addTheta:
                hOn_2d.Fill(energy, theta2) 
            else:
                hOn.Fill(energy)
        else:
            Noff += 1
            w.append(alpha)
            if addTheta:
                hOff_2d.Fill(energy, theta2, alpha)
            else:
                hOff.Fill(energy, alpha)

    w_avg = np.average(w)
    #events = events
    
    if rawdata:
        return events

    if getRuns:
        return runs
    else:

        if addTheta:
            hOn_2d.SetDirectory(0)
            hOff_2d.SetDirectory(0)
            if full_output:
                return hOn_2d, hOff_2d, Non, Noff, events, w_avg
            else:
                return hOn_2d, hOff_2d
        else:
            hOn.SetDirectory(0)
            hOff.SetDirectory(0)
            if full_output:
                return hOn, hOff, Non, Noff, events, w_avg
            else:
                return hOn, hOff


def plotData(dwarf, thCut = 0, addTheta=False, eLowerCut=100, eUpperCut = 1e5, full_output=False):
    thCut = defineThetaCut("VEGAS", thCut)
    hOn, hOff = readData(dwarf, addTheta=addTheta, thCut=thCut, full_output=full_output, eLowerCut=eLowerCut, eUpperCut = eUpperCut)
    
    if addTheta:
        hDiff = TH1D("chi", "chi", 20, 0, 5)
        hDiff.SetTitle("#chi ^{2} distribution")
        for i in range(1, hOn.GetNbinsX()+1):
            for j in range(1, hOn.GetNbinsY()+1):
                if hOff.GetBinContent(i, j)!=0 :
                    diff = (hOn.GetBinContent(i, j)-hOff.GetBinContent(i, j))**2./hOff.GetBinContent(i, j)
                    hDiff.Fill(diff)
        c = TCanvas("Observation", "Observation", 900,300)
        c.Divide(3,1)
        c.cd(1)
        hOn.Draw("colz")
        gPad.SetLogx()
        c.cd(2)
        hOff.Draw("colz")
        gPad.SetLogx()
        c.cd(3)
        hDiff.Draw()
        hOn.SetDirectory(0)
        hOff.SetDirectory(0)
        hDiff.SetDirectory(0)
        c.Draw()
        return c, hOn, hOff, hDiff
    else:
        xOn, yOn = getArray(hOn)
        xOff, yOff = getArray(hOff)
        chisq = sum((yOn[yOff!=0]-yOff[yOff!=0])**2/yOff[yOff!=0])
        dof = len(yOff[yOff!=0])

        f, ax = plt.subplots(2,1, figsize=(7, 7), gridspec_kw={'height_ratios':[5,1]})
        ax[0].step(xOn, yOn,  label=r"On region", where="mid")
        ax[0].step(xOff, yOff, label=r"Off region", where="mid")
        #ax[0].text(0.82, 0.77, r"$\chi^2$ / dof = {:.1f} / {} = {:.2f}".format(chisq, dof, chisq/dof), ha="right", fontsize=12, transform=ax[0].transAxes)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlim(50, 2e5)
        ax[0].set_ylabel("Counts", fontsize=15)
        ax[0].legend(fontsize=12, loc=1, frameon=False)
        ax[0].grid()

        ax[1].errorbar(xOn[yOn!=0], np.sign(yOn[yOn!=0]-yOff[yOn!=0])*(yOff[yOn!=0]-yOn[yOn!=0])**2./yOn[yOn!=0], yerr= 1, marker="+", ls="", c="k", label="on/off)")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Energy [GeV]", fontsize=15)
        ax[1].set_ylabel(r"$\chi^2$", fontsize=15)
        ax[1].set_xlim(50, 2e5)
        ax[1].set_ylim(-5, 5)
        ax[1].axhline(0, color="k", ls="--")
        ax[1].grid()

    return ax
