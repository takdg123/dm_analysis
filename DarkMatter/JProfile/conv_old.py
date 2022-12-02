import numpy as np
from numpy import *
import os

from ROOT import TFile, TMath
from ROOT import TCanvas, gPad
from ROOT import TGraph, TGraph2D, TH1D, TH2D

from .psf import *
from .profile import *

from .. import ResponseFunction

from .. import const
from ..utils import defineThetaCut, getArray

from tqdm.notebook import tqdm, trange

from ..const import defaultNum


# For conversion
CmToPc   = 3.24e-19       # 1cm in pc
MsToKg = 1.989e30         # 1 solar mass to kg
GeVToKg  = 1.78e-27       # GeV to kg
KgToGeV  = 1.0/GeVToKg    # kg to GeV  

def generateConvolvedJ(dwarf, package, filename = None, thCut=0, version="all", verbose=True, ext=False):

    if ext:
        thCut = np.sqrt(0.02)
    
    thCut = defineThetaCut(package, thCut**2)
    
    if filename == None:
        if not(os.path.isdir(const.OUTPUT_DIR)):
            os.system("mkdir "+const.OUTPUT_DIR)
        
        filename = const.OUTPUT_DIR+"/JProfile_{}_{}.root".format(package, dwarf)
    
    convJ1D = TGraph()
    convJ2D = TGraph2D()

    convJ1D.SetTitle("Convolved J profile")
    convJ1D.GetXaxis().SetTitle("Energy [GeV]");
    convJ1D.GetYaxis().SetTitle(r"J (#theta < {:.3f}) [GeV$^{{-2}}$ cm$^{{-5}}$]".format(thCut));

    convJ2D.SetTitle("Convolved 2D J profile")
    convJ2D.GetXaxis().SetTitle("Energy [GeV]")
    convJ2D.GetYaxis().SetTitle("Theta [Deg]")

    if package=="VEGAS":
        energies = const.eVJbins
        irf = ResponseFunction.VEGAS(dwarf, mode = 3, verbose=False)
        PSF = irf.PSF
    elif package=="EventDisplay":
        energies = const.eEDJbins
        try:
            irf = ResponseFunction.EventDisplay.readIRFs(dwarf, version=version, ext=ext)
        except:
            irf = ResponseFunction.EventDisplay.averagedIRFs(dwarf, version=version, ext=ext)
    PSF = irf.PSF

    gJProf = calcJProfile_old(dwarf, seed = defaultNum[dwarf])

    m=0
    n=0
    minE, maxE = findMinMaxE(PSF)

    for i, en in enumerate(tqdm(energies)):
        if (en <= 10**minE) or (en >= 10**maxE):
            continue

        gPSF = getPSF1D(PSF, en, package=package)
        hJProf2D, J = calcConvJProf2D(gJProf, gPSF, thCut=thCut)
        gRadJProf, J2 = getRadialConvJProf(hJProf2D, energy=en, package=package, thCut=thCut)
        
        convJ1D.SetPoint(m, en, J)

        m+=1

        theta, J2D = getArray(gRadJProf)
        
        for j in range(gRadJProf.GetN()):
            convJ2D.SetPoint(n, round(np.log10(en), 1), theta[j], J2D[j])
            n+=1
    try:
        OutFile = TFile(filename,"UPDATE");
    except:
        OutFile = TFile(filename,"RECREATE");

    if version == "all":
        OutFile.cd()
        convJ1D.Write("gConvJ1D")
        convJ2D.Write("gConvJ2D") 
    else:
        OutFile.cd()
        convJ1D.Write("gConvJ1D_{}".format(version))
        convJ2D.Write("gConvJ2D_{}".format(version)) 
    OutFile.Close()

    if verbose:
        try:
            clear_output()
        except:
            pass
        print("[Log] Finish. J profile is saved in {}.".format(filename))
 

# This is for making 1D J profile to 2D convoloved J profile
def calcConvJProf2D(gJProf, gPSF, nBins = 200, r_smooth = 1, theta_max = 1.5, thCut = 0, package="", return_conv=True, ext=False, verbose=False):

    if ext:
        thCut = np.sqrt(0.02)

    thCut = defineThetaCut(package, thCut**2.)

    x_min, x_max = -2., 2.   # This is based on the PSF test (within +/-2 deg, the sum of PSF is about 1)
    y_min, y_max = -2., 2.   

    binEdge = np.arange(-2.006, 2.010, step=0.004)

    hProfConv  = TH2D("hProfConv","J*PSF Profile",len(binEdge)-1,binEdge,len(binEdge)-1,binEdge)
    hProf2D    = TH2D("hProf2D","J Profile (x 10^{-20})",len(binEdge)-1,binEdge,len(binEdge)-1,binEdge)
    hProfConv1D  = TH1D("hProfConv1D","J*PSF Profile",len(binEdge)-1,binEdge)

    hProfConv.SetStats(0)
    hProf2D.SetStats(0)

    NBinsX = hProf2D.GetXaxis().GetNbins()
    NBinsY = hProf2D.GetYaxis().GetNbins()

    for i in range(1, NBinsX+1):
        for j in range(1, NBinsY+1):
            x0 = hProf2D.GetXaxis().GetBinCenter(i) # degree
            y0 = hProf2D.GetYaxis().GetBinCenter(j)
            
            dx = hProf2D.GetXaxis().GetBinWidth(i)
            dy = hProf2D.GetYaxis().GetBinWidth(j)
            
            if abs(x0) < 1e-5:
                x0 = 0
            if abs(y0) < 1e-5:
                y0 = 0

            r = np.sqrt(x0*x0 + y0*y0)                 # r: degree not radius
            if r < 1e-5:
                continue
            else:
                L_r = gJProf.Eval(r)/(2*pi*sin(r*TMath.DegToRad()))  # Luminosity profile / (2 pi sin(r))

            if L_r > 0.0:
                hProf2D.SetBinContent(i,j,L_r/1e20)

    if return_conv:

        for i in trange(1, NBinsX+1):
            x0 = hProfConv.GetXaxis().GetBinCenter(i)
            y0 = 0
            r0 = np.sqrt(x0*x0 + y0*y0)

            if (r0 > thCut) or (r0 > theta_max):
                hProfConv.Fill(x0, y0, 0) 
                hProfConv1D.Fill(x0, 0)
                continue

            x_min = x0 - r_smooth
            y_min = y0 - r_smooth
            x_max = x0 + r_smooth
            y_max = y0 + r_smooth

            z, x_arr, y_arr = getArray(hProfConv)

            n_min = hProfConv.GetXaxis().FindBin(x_min)
            m_min = hProfConv.GetYaxis().FindBin(y_min)
            n_max = hProfConv.GetXaxis().FindBin(x_max)
            m_max = hProfConv.GetYaxis().FindBin(y_max)

            for n in range(n_min, n_max+1):
                x = hProfConv.GetXaxis().GetBinCenter(n)
                dx = hProfConv.GetXaxis().GetBinWidth(n)

                for m in range(m_min, m_max+1):
                    y = hProfConv.GetYaxis().GetBinCenter(m)
                    dy = hProfConv.GetYaxis().GetBinWidth(m)

                    rJ = np.sqrt(x*x + y*y)
                    rP = np.sqrt(pow(x-x0,2.0) + pow(y-y0,2.0))

                    if rJ < 1e-5:
                        continue
                    else:
                        L_r = gJProf.Eval(rJ)/(2*pi*sin(rJ*TMath.DegToRad()))

                    F_r = gPSF.Eval(rP)*L_r*dx*dy
                    if F_r >0.0:
                        hProfConv.Fill(x0,y0,F_r) # Kind of integration
                        hProfConv1D.Fill(x0, F_r)

        for i in range(1, NBinsX+1):
            for j in range(1, NBinsY+1):
                x0 = hProfConv.GetXaxis().GetBinCenter(i)
                y0 = hProfConv.GetYaxis().GetBinCenter(j)
                
                r0 = np.sqrt(x0*x0 + y0*y0)
                
                if r0 > theta_max:  
                    continue
                
                F_r = hProfConv1D.Interpolate(r0)
                
                hProfConv.SetBinContent(i, j, F_r)

    J, J_nConv = calcConvJ(hProfConv, hProf2D, thCut = thCut)

    hProfConv.GetXaxis().SetTitle("Theta [deg]")
    hProfConv.GetYaxis().SetTitle("Theta [deg]")
    
    if verbose:
        print("[Log] Convolved J profile: {:.2e}".format(en/1000., J))
    
    if return_conv:
        return hProfConv, J, 
    else:
        return hProf2D, J_nConv


def calcConvJ(hProfConv=[], hProf2D =[], thCut=0, package="", verbose=False):

    thCut = defineThetaCut(package, thCut**2.)
    J = 0.0;
    J_nConv = 0.0;
    PSF_norm = 0.0;

    if len(hProfConv) == 0:
        h = hProf2D
    else:
        h = hProfConv


    for i in range(1, h.GetNbinsX()+1):
        for j in range(1, h.GetNbinsY()+1):
            x0 = h.GetXaxis().GetBinCenter(i)
            y0 = h.GetYaxis().GetBinCenter(j)
            dx = h.GetXaxis().GetBinWidth(i)*TMath.DegToRad() # working in degrees, need final J factors in terms of sr
            dy = h.GetYaxis().GetBinWidth(j)*TMath.DegToRad()

            r = np.sqrt(x0*x0 + y0*y0) 

            if r<thCut:   
                if len(hProfConv)!=0:
                    J += hProfConv.GetBinContent(i,j)*dx*dy
                if len(hProf2D)!=0:
                    J_nConv += hProf2D.GetBinContent(i,j)*1e20*dx*dy

    if verbose:
        if len(hProfConv)!=0:
            print("[Log] J factor (w/ convolution): ", J)
        if len(hProf2D)!=0:
            print("[Log] J factor (no convolution): ", J_nConv)
        
    return J, J_nConv


def getRadialConvJProf(h, energy=None, nBins=400, thCut=0, package="", verbose=False):
# INPUT
#    hJProf2D
# OUTOUT
#    hRadJProf: 2D (hJProf2D) to 1D
    thCut = defineThetaCut(package, thCut**2.)
    thBins = np.arange(0, 2.001, step=0.004)

    hRadJProf = TH1D("hRadJProf","hRadJProf",len(thBins)-1,thBins)
    hRadN = TH1D("nPerBin","nPerBin",len(thBins)-1,thBins)

    gRadJProf = TGraph()
    gRadJProf.Set(0)

    for i in range(1, h.GetNbinsX()+1):
        for j in range(1, h.GetNbinsY()+1):
            x = h.GetXaxis().GetBinCenter(i);
            y = h.GetYaxis().GetBinCenter(j);
            r = np.sqrt(x*x + y*y);

            hRadJProf.Fill(r, h.GetBinContent(i,j));
            hRadN.Fill(r,1.0);
    
    k = 0
    for i in range(1, nBins+1):
        n = hRadN.GetBinContent(i)
        r = hRadJProf.GetBinCenter(i)
        J = hRadJProf.GetBinContent(i);

        if n == 0:
            continue

        gRadJProf.SetPoint(k, r, J/n);
        k+=1

    J = calcConvJ_rad(gRadJProf, energy=energy, thCut=thCut, package=package, verbose=verbose)

    gRadJProf.GetXaxis().SetTitle("Theta [deg]")
    gRadJProf.GetYaxis().SetTitle(r"dJ/d$\Omega$")

    return gRadJProf, J

def calcConvJ_rad(gJ, energy = None, thCut = 0, package="", theta=[], width=0.004, verbose=False):
    
    thCut = defineThetaCut(package, thCut**2.)

    if len(theta)==0:
        if gJ.Class_Name() == "TGraph2D":
            z, e, theta = getArray(gJ)
        else:
            theta, y = getArray(gJ)
    
    J = 0
    for i in range(len(theta)):
        r = theta[i]
        dr = np.diff(theta)[0]

        if r < thCut:
            if gJ.Class_Name() == "TGraph2D":
                J0 = gJ.Interpolate(np.log10(energy), r)
            else:
                J0 = gJ.Eval(r)


            if J0 >0:
                r = r*TMath.DegToRad()
                dr = dr*TMath.DegToRad()
                J+= J0*2*np.pi*r*dr
        else:
            r0 = r
            dr = thCut-r
            r = r0+dr/2.
            r = r*TMath.DegToRad()
            dr = dr*TMath.DegToRad()
            J+= J0*2*np.pi*r*dr/2.
            break

    if verbose:
        print("[Log] J factor (w/ convolution): ", J)

    return J

def calcConvJ_eng(gJ, theta = None, thCut = 0, package="", verbose=False):
    
    if gJ.Class_Name() != "TGraph2D":
        print("[Error] The input file is not TGraph2D.")
        return
    
    J = 0
    eBinEdges = const.eKnots
    for i in range(len(eBinEdges)-1):
        r = (np.log10(eBinEdges[i+1])+np.log10(eBinEdges[i]))/2.
        dr = eBinEdges[i+1]-eBinEdges[i]
        J0 = gJ.Interpolate(r, theta)

        if not(np.isnan(J0)):
            J+= J0*dr

    if verbose:
        print("[Log] J factor (w/ convolution): ", J)

    return J

def calcConvJ_rad_multi(gConvJ2D, thCut=0, width=0.004, axis="theta", package=""):
    thCut = defineThetaCut(package, thCut**2.)
    
    J_tot = []
    if axis=="theta":
        z, energies, theta = getArray(gConvJ2D)
        for en in energies:
            J = calcConvJ_rad(gConvJ2D, energy=10**en, theta=theta, package=package, thCut = thCut)
            J_tot.append(J)
        return np.asarray(J_tot), energies
    elif axis=="energy":
        z, energies, theta = getArray(gConvJ2D)
        for th in theta:
            J = calcConvJ_eng(gConvJ2D, theta=th, package=package, thCut = thCut)
            J_tot.append(J)
        return np.asarray(J_tot), theta
    
def findMinMaxE(PSF):
    minE = 0
    maxE = 0
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


# Calculate J profile
def calcJProfile_old(dwarf, verbose=False, thCut=-1, seed=-1):
# INPUT
#    dwarf: the name of dwarf
# OUTPUT
#    gL: Luminostiy profile (degree)

    th = np.load(const.SCRIPT_DIR+"/npy/thKnots_jp.npy")
    nBinsTh = len(th)
        
    for i in range(nBinsTh):
        th[i] *= TMath.DegToRad()
        
    th_max = th[-1] # radians
    th_min = th[0] # radians
    if verbose:
        print("[Log] Theta^2 range between {:.2f} and {:.2f}".format(th_min*TMath.RadToDeg(), th_max*TMath.RadToDeg()))

    
    if verbose:
        print("[Log] Truncation radius in pc: ", r_t)

    d = Distance2Dwarf[dwarf]

    nInt = 1000

    gL = TGraph()
    gL.Set(0)
    j = 0
    L = np.zeros(nBinsTh)
    params = dwarfParam(dwarf)
    props = params[seed]
    r_t = TruncationRadius[dwarf]
    
    for i in range(nBinsTh):
    
        if pow(r_t,2.0) - pow(d*sin(th[i]),2.0) < 0:
            if verbose:
                print("[Log] The maximum theta is {:.3f} deg".format(th[i]*TMath.RadToDeg()))
            gL.SetPoint(j, th[i]*TMath.RadToDeg(), 0);
            j+=1
            break
        
        s_temp = np.sqrt(abs(pow(r_t,2.0) - pow(d*sin(th[i]),2.0))) 
        s_min = d*cos(th[i]) - s_temp; 
        s_max = d*cos(th[i]) + s_temp;

        s = s_min;

        ds = (s_max - s_min)/nInt;
        
        # This is for integration
        while(s < s_max):
            r = np.sqrt(s*s + d*d - 2*s*d*cos(th[i])); # pc
            L[i] += pow(calcAlexDMProfile(dwarf, props, r=r)*MsToKg*KgToGeV*pow(CmToPc,3.0), 2.0)*(ds/CmToPc);
            s += ds;
        
        
        L[i] *= 2*pi*sin(th[i]) # Luminosity profile at a particular value of th[i]
        
        if L[i]!=np.nan and L[i]>0:
            if thCut == -1:
                gL.SetPoint(j, th[i]*TMath.RadToDeg(), L[i]);
            else:
                if th[i]*TMath.RadToDeg() < thCut:
                    gL.SetPoint(j, th[i]*TMath.RadToDeg(), L[i]);
                else:
                    gL.SetPoint(j, th[i]*TMath.RadToDeg(), 0);
            j+=1

    return gL
