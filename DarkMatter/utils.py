import numpy as np

import os

import re

import math

import matplotlib.pyplot as plt

from . import const

from ROOT import TH1D, TH2D, TGraph
from ROOT import TCanvas, gPad, gStyle

DM_DIR = os.environ.get('DM')
REF_DIR = DM_DIR+"/RefData/"
DATA_DIR = DM_DIR+"/Data/"

gaus = lambda x, m, sigma: (1./(np.sqrt(2.*np.pi)*sigma))* np.exp(-( (m-x)**2 / ( 2 * sigma**2 ) ) )

def center_pt(x):
    return (x[1:]+x[:-1])/2.

def thetaEdges(thCut):
    return np.arange(0, math.ceil(thCut*1000)/1000.+0.0001, step=0.001)

def printRunList(dwarf, path=None, package ="EventDisplay", printOutput=False, log_only=False, ext=False):
    if package=="EventDisplay":
        if path==None:
            if ext:
                runlist = os.listdir(DATA_DIR+dwarf+"_ext")
            else:
                runlist = os.listdir(DATA_DIR+dwarf)
        else: 
            runlist = os.listdir(path)
        if log_only:
            runlist = [l[:5] for l in runlist if ".anasum.log" in l]
        else:
            runlist = [l[:5] for l in runlist if ".anasum.root" in l]
            
    elif package=="VEGAS":
        if dwarf == "segue_1":
            path = REF_DIR+"/Pass5f/segue1_bias/"
        elif dwarf == "draco":
            path = REF_DIR+"/Pass5f/{}_bias/".format(dwarf)
        elif dwarf == "ursa_minor":
            path = REF_DIR+"/Pass5f/umi_bias/"
        elif dwarf == "bootes":
            path = REF_DIR+"/Pass5f/{}_bias/".format(dwarf)
        runlist = os.listdir(path)
        runlist = [l[:5] for l in runlist if "mat" in l]
    runlist = list(set(runlist))
    runlist.sort()
    if printOutput:
        for run in runlist:
            print(run)
    else:
        return runlist

# Convert root class to numpy array
def getArray(h, return_errors=False, return_edges=False):
    cName = h.Class_Name() 
    if cName == "TH1D" or cName=="TH1F":
        output = []
        error = []
        nBinsX = h.GetNbinsX()
        edges = []
        for i in range(1, nBinsX+1):
            output.append([h.GetBinCenter(i), h.GetBinContent(i)])
            edges.append(h.GetBinLowEdge(i))
            error.append(h.GetBinError(i))

        edges.append(h.GetBinLowEdge(i+1))
        output = np.asarray(output)

        if return_errors:
            return output[:,0], output[:,1], np.asarray(error)
        elif return_edges:
            return output[:,1], np.asarray(edges)
        else:
            return output[:,0], output[:,1]

    elif cName == "TH2D" or cName=="TH2F":
        nBinsX = h.GetNbinsX()
        nBinsY = h.GetNbinsY()
        shape = (nBinsY + 2, nBinsX + 2)
        
        array = np.ndarray(shape=shape, dtype="f8",
                           buffer=h.GetArray())
            
        array = array[tuple([slice(1, -1) for idim in range(array.ndim)])]
        if return_edges:
            xEdges = [h.GetXaxis().GetBinLowEdge(1)]+[h.GetXaxis().GetBinUpEdge(i+1) for i in range(nBinsX)]
            yEdges = [h.GetYaxis().GetBinLowEdge(1)]+[h.GetYaxis().GetBinUpEdge(i+1) for i in range(nBinsY)]
            return array, np.asarray(xEdges), np.asarray(yEdges)
        else:
            xBins = [h.GetXaxis().GetBinCenter(i+1) for i in range(nBinsX)]
            yBins = [h.GetYaxis().GetBinCenter(i+1) for i in range(nBinsY)]
            return array, np.asarray(xBins), np.asarray(yBins)

    elif cName == "TGraph" or cName=="TGraphErrors" or cName=="TGraphAsymmErrors":
        nPoints = h.GetN()
        output = []
        for i in range(nPoints):
            output.append([h.GetPointX(i),h.GetPointY(i)])
        output = np.asarray(output)
        if return_edges:
            width = np.diff(output[:,0])
            width = np.asarray([round(w, 3) for w in width])
            if sum(width == width[0]) == len(width):
                edges = (output[:,0]-width[0]/2.).tolist()
                edges.append(output[:,0][-1]+width[0]/2.)
                return np.asarray(edges), output[:,1]
            else:
                print("[Error] Edges cannot be estimated.")
                return output[:,0], output[:,1]
        else:
            return output[:,0], output[:,1]

    elif cName == "TGraph2D":
        output = []
        for z, x, y in zip(np.asarray(h.GetZ()),np.asarray(h.GetX()),np.asarray(h.GetY())):
            output.append([z, x, y])
        output = np.asarray(output)
        return output, output[:,1], output[:,2]

def convertHist(arr_x, arr_y):
    if len(arr_x) == len(arr_y):
        h = TGraph()
        for i, (x, y) in enumerate(zip(arr_x, arr_y)):
            h.SetPoint(i, x, y)
    return h
    
def findIrfFile(filename, return_name=False):
    with open(filename) as file:
        for line in file.readlines():
            effFile = re.findall("effective areas from ([a-zA-Z0-9\-\.\_]+)", line)
            if len(effFile)==1:
                break
    effFile = REF_DIR+"/effArea/"+effFile[0]
    if return_name:
        return effFile
    elif os.path.exists(effFile):
        return effFile
    else:
        return False

def findAltFile(version="v6", cut="soft", runNum = None, irf_file=None, ext=False):
    if irf_file is None:
        if version == "v6":
            if ext:
                if int(runNum) >= 63373 and int(runNum) <= 67410:
                    InFile = REF_DIR+"/effArea/effArea-v483-auxv01-CARE_June1702-Cut-NTel3-ExtendedSource-Moderate-TMVA-BDT-GEO-V6_2012_2013a-ATM61-T1234.root"
                elif int(runNum) >= 67411 and int(runNum) <= 70170:
                    InFile = REF_DIR+"/effArea/effArea-v483-auxv01-CARE_June1702-Cut-NTel3-ExtendedSource-Moderate-TMVA-BDT-GEO-V6_2012_2013b-ATM61-T1234.root"
                else:
                    print("[Error] There is no IRF file (run number={})".format(int(runNum)))
            else:
                if cut=="soft":
                    if int(runNum) >= 63373 and int(runNum) <= 67410:
                        InFile = REF_DIR+"/effArea/effArea-v483-auxv01-CARE_June1702-Cut-NTel2-PointSource-Soft-TMVA-BDT-GEO-V6_2012_2013a-ATM61-T1234.root"
                    elif int(runNum) >= 67411 and int(runNum) <= 70170:
                        InFile = REF_DIR+"/effArea/effArea-v483-auxv01-CARE_June1702-Cut-NTel2-PointSource-Soft-TMVA-BDT-GEO-V6_2012_2013b-ATM61-T1234.root"
                    else:
                        print("[Error] There is no IRF file (run number={})".format(int(runNum)))
                elif cut=="moderate":
                    if int(runNum) >= 63373 and int(runNum) <= 67410:
                        InFile = REF_DIR+"/effArea/effArea-v483-auxv01-CARE_June1702-Cut-NTel2-PointSource-Moderate-TMVA-BDT-GEO-V6_2012_2013a-ATM61-T1234.root"
                    elif int(runNum) >= 67411 and int(runNum) <= 70170:
                        InFile = REF_DIR+"/effArea/effArea-v483-auxv01-CARE_June1702-Cut-NTel2-PointSource-Moderate-TMVA-BDT-GEO-V6_2012_2013b-ATM61-T1234.root"
                    else:
                        print("[Error] There is no IRF file (run number={})".format(int(runNum)))
        elif version == "v5":
            if ext:
                InFile = REF_DIR+"/effArea/effArea-v483-auxv01-GRISU-Cut-NTel3-ExtendedSource-Moderate-TMVA-BDT-GEO-V5-ATM21-T1234.root"
            else:
                if cut=="soft":
                    InFile = REF_DIR+"/effArea/effArea-v483-auxv01-GRISU-Cut-NTel2-PointSource-Soft-TMVA-BDT-GEO-V5-ATM21-T1234.root"
                elif cut=="moderate":
                    InFile = REF_DIR+"/effArea/effArea-v483-auxv01-GRISU-Cut-NTel2-PointSource-Moderate-TMVA-BDT-GEO-V5-ATM21-T1234.root"
                else:
                    print("[Error] There is no IRF file (run number={})".format(int(runNum)))
        else:
            if ext:
                InFile = REF_DIR+"/effArea/effArea-v483-auxv01-GRISU-Cut-NTel3-ExtendedSource-Moderate-TMVA-BDT-GEO-V4-ATM21-T1234.root"
            else:
                if cut=="soft":
                    InFile = REF_DIR+"/effArea/effArea-v483-auxv01-GRISU-Cut-NTel2-PointSource-Soft-TMVA-BDT-GEO-V4-ATM21-T1234.root"
                elif cut == "moderate":
                    InFile = REF_DIR+"/effArea/effArea-v483-auxv01-GRISU-Cut-NTel2-PointSource-Moderate-TMVA-BDT-GEO-V4-ATM21-T1234.root"
                else:
                    print("[Error] There is no IRF file (run number={})".format(int(runNum)))
    else:
        InFile = irf_file
    return InFile

# Simple read bin file    
def readBinFile(path):
    with open(path) as f:
        data = []
        for line in f.readlines():
            try:
                data.append(float(line))
            except:
                continue

    return np.asarray(data)

def plot2D(x, y, z, ax = None, vmax=None, logx=True, logy=False):
    xticks = []
    for i in range(len(x)):
        if i%10==0:
            if logx:
                xticks.append([i, "{:.1f}".format(np.log10(x)[i])])
            else:
                if max(x)>1e2:
                    xticks.append([i, "{:.1e}".format(x[i])])
                else:
                    xticks.append([i, "{:.1f}".format(x[i])])
    xticks=np.asarray(xticks)        

    yticks = []
    for i in range(len(y)):
        if i%10==0:
            if logy:
                yticks.append([i, "{:.2f}".format(np.log10(y)[i])])
            else:
                yticks.append([i, "{:.2f}".format(y[i])])
    yticks=np.asarray(yticks)        

    if ax!=None:
        f = ax
    else:
        f = plt
    
    cnt = f.imshow(z, aspect='auto', vmax=vmax)
    
    if ax!=None:
        plt.colorbar(cnt, ax = ax)
        f.set_xticks(xticks[:,0].astype("int"))
        f.set_xticklabels(xticks[:,1])
        f.set_yticks(yticks[:,0].astype("int"))
        f.set_yticklabels(yticks[:,1])
        f.set_xlim(0, len(x))
        f.set_ylim(0, len(y))
    else:
        plt.colorbar(cnt)
        f.xticks(xticks[:,0].astype("int"), xticks[:,1])
        f.yticks(yticks[:,0].astype("int"), yticks[:,1])
        f.xlim(0, len(x))
        f.ylim(0, len(y))
    return f

def convertEdisp(h):
    hDispProb = h.Clone()
    hDispProb.GetZaxis().SetTitle("Probability");
    for i in range(1, hDispProb.GetNbinsX()+1):
        dEtr = hDispProb.GetXaxis().GetBinWidth(i)
        for j in range(1, hDispProb.GetNbinsY()+1):
            P = h.GetBinContent(i, j)*dEtr
            hDispProb.SetBinContent(i, j, P)
    hDispProb.SetDirectory(0)
    return hDispProb     

def convertToPDF(hg, norm = True):
    h = hg.Clone()
    h.SetTitle("Probability density function")
    if h.Class_Name() == "TH1D":
        if norm:
            h.GetYaxis().SetTitle("Likelihood")
        else:
            h.GetYaxis().SetTitle("Differential counts")
        for i in range(1, h.GetNbinsX()+1):
            dh = h.GetXaxis().GetBinWidth(i)
            val = h.GetBinContent(i)
            h.SetBinContent(i, val/dh)
        n_factor = h.Integral(1, h.GetNbinsX(), "width")
        if n_factor!=0 and norm:
            h.Scale(1.0/n_factor)
    elif h.Class_Name() == "TH2D":
        if norm:
            h.GetZaxis().SetTitle("Likelihood")
        else:
            h.GetZaxis().SetTitle("Differential counts")
            
        for i in range(1, h.GetNbinsX()+1):
            for j in range(1, h.GetNbinsY()+1):
                dx = h.GetXaxis().GetBinWidth(i)
                dy = h.GetYaxis().GetBinWidth(j)
                val = h.GetBinContent(i, j)
                h.SetBinContent(i, j, val/(dx*dy))
                
        n_factor = h.Integral(1, h.GetNbinsX(), 1, h.GetNbinsY(), "width")
        if n_factor!=0 and norm:
            h.Scale(1.0/n_factor)
    h.SetDirectory(0)
    return h

def defineThetaCut(package="", th2Cut=0):
    if th2Cut==0:
        if package=="VEGAS":
            thCut = np.sqrt(0.03)
        elif package=="EventDisplay":
            thCut = np.sqrt(0.008)
        else:
            print("[Error] Either package or thCut should be specified.")
            raise ValueError
    else:
        thCut = np.sqrt(th2Cut)
    return thCut

def defineTheta2Cut(package="", thCut=0):
    if thCut==0:
        if package=="VEGAS":
            thCut = 0.03
        elif package=="EventDisplay":
            thCut = 0.008
        else:
            print("[Error] Either package or thCut should be specified.")
            raise ValueError
    else:
        thCut = thCut
    return thCut

def LiMaSiginficance(N_on, N_off, alpha, type=1):
    if type == 1:
        temp = N_on*np.log((1.+alpha)/alpha*(N_on/(N_on+N_off)))+N_off*np.log((1+alpha)*(N_off/(N_on+N_off)))
    
        if np.size(temp) != 1:
            for i, t in enumerate(temp):
                if t > 0:
                    temp[i] = np.sqrt(t)
                else:
                    temp[i] = np.nan
        else:
            if temp >0:
                temp = np.sqrt(temp)
            else:
                temp = np.nan

        significance = np.sign(N_on-alpha*N_off)*np.sqrt(2.)*temp
    else:
        significance = (N_on-alpha*N_off)/np.sqrt(alpha*(N_on+N_off))
    return significance


def convertEvt2Hist(events, dwarf="segue_1", thCut=0, package="EventDisplay", addTheta=False, isOn=True):
    thCut = defineTheta2Cut(package, thCut)
    
    eBinEdges = np.load(const.OUTPUT_DIR+"/npy/signalBins_{}.npy".format(dwarf))
    tBinEdges = thetaEdges(thCut)

    if addTheta:
        hOn_2d = TH2D("hEvt","hEvt_1D", len(eBinEdges)-1, eBinEdges, len(tBinEdges)-1, tBinEdges)
        hOn_2d.SetTitle("2D count spectrum (on region)")
        hOn_2d.GetXaxis().SetTitle("Energy [GeV]")
        hOn_2d.GetYaxis().SetTitle("Theta2 [deg^2]")
        hOn_2d.GetZaxis().SetTitle("Counts")
        hOn_2d.SetDirectory(0)
    else:
        hOn = TH1D("hEvt", "hEvt_1D", len(eBinEdges)-1, eBinEdges)
        hOn.SetTitle("Count spectrum (on region)")
        hOn.GetXaxis().SetTitle("Energy [GeV]")
        hOn.GetYaxis().SetTitle("Counts")
        hOn.SetDirectory(0)

    w = []
    Non = 0
    Noff = 0
    for evt in events:
        energy = evt[0]
        theta = evt[1]
        if addTheta:
            hOn_2d.Fill(energy, theta)  
        else:
            hOn.Fill(energy)


    if addTheta:
        return hOn_2d
    else:
        return hOn

    
def getVersion(runNum):
    if int(runNum) > 63407:
        version = "v6"
    elif int(runNum) > 46549:
        version = "v5"
    else:
        version = "v4"
    return version


def listOfVersions(dwarf):
    try:
        events = np.load(const.DATA_DIR+'EventDisplay_Events_{}.npy'.format(dwarf), allow_pickle=True)
    except:
        try:
            events = np.load(const.DATA_DIR+'EventDisplay_Events_{}_ext.npy'.format(dwarf), allow_pickle=True)
        except:
            print("[Error] An event file does not exist.")
    version = ["v"+str(int(v)) for v in list(set(events[:,4]))]
    return version

def plotRoot(h, h2=None, logx=False, logy=False, same=True, logx2=False, logy2=False, logz=False, logz2=False):
    cName = h.Class_Name()
    if h2==None:
        c = TCanvas("", "", 400, 300)
        if cName == "TH2D" or cName == "TGraph2D":
            h.Draw("colz")
        elif cName == "TH1D":
            h.Draw("hist")
        else:
            h.Draw()
        if logx: c.SetLogx()
        if logy: c.SetLogy()
        if logz: c.SetLogx()
    else:
        cName2 = h2.Class_Name()
        if cName == "TH2D" or cName == "TGraph2D" or cName2 == "TH2D" or cName2 == "TGraph2D" or not(same):
            c = TCanvas("", "", 900,400)
            c.Divide(2,1)
            c.cd(1)
            if cName == "TH2D" or cName == "TGraph2D":
                h.Draw("colz")
            elif cName == "TH1D":
                h.Draw("hist")
            else:
                h.Draw()
            if logx: gPad.SetLogx()
            if logy: gPad.SetLogy()
            if logz: gPad.SetLogz()
            c.cd(2)
            if cName2 == "TH2D" or cName2 == "TGraph2D":
                h2.Draw("colz")
            elif cName2 == "TH1D":
                h2.Draw("hist")
            else:
                h2.Draw()
            if logx2: gPad.SetLogx()
            if logy2: gPad.SetLogy()
            if logz2: gPad.SetLogz()
        else:
            c = TCanvas("", "", 400, 300)
            h.SetLineColor(1)
            if cName == "TH1D":
                h.DrawClone("hist")
            else:
                h.DrawClone()
            h2.SetLineColor(2)
            if cName2 == "TH1D":
                h2.DrawClone("hist same")
            else:
                h2.DrawClone("same")
            if logx: c.SetLogx()
            if logy: c.SetLogy()
            if logz: c.SetLogz()
    gStyle.SetOptStat(0)
    c.Draw()

    if h2 == None:
        return c, h
    else:
        return c, h, h2

POWERLAW = lambda E, N0, idx: N0/1e14*(E/1000.)**idx

def BKNPOWER(E, N0, alpha, Eb, beta):
    
    normF = (Eb/1000.)**(-beta+alpha)
    if np.size(E)==1:
        if E < Eb:
            val = POWERLAW(E, N0, alpha)
        else:
            val = POWERLAW(E, N0, beta)*normF
    else:
        engs = np.asarray(E)
        cutoff = -1
        for i in range(len(engs)):
            if engs[i] >= Eb: 
                cutoff = i
                break
        if min(E)>Eb:
            val1 = POWERLAW(engs, N0, beta)*normF
            val2 = np.asarray([])
        elif max(E)<Eb:
            val1 = np.asarray([])
            val2 = POWERLAW(engs, N0, alpha)
        else:
            val1 = POWERLAW(engs[:cutoff], N0, alpha)
            val2 = POWERLAW(engs[cutoff:], N0, beta)*normF
        val = val1.tolist() + val2.tolist()
        
    return np.asarray(val)



class newirf:
    def __init__(self, package=None):
        self.package = package
