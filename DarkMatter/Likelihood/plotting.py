import numpy as np
import os

import matplotlib.pyplot as plt
from ..const import REF_DIR, OUTPUT_DIR

from scipy.stats import norm as norm_dist
import re

import matplotlib as mpl

from scipy.interpolate import interp1d

def plotLikelihoodProfiles(gL):
    mass = list(gL.keys())
    for i, m in enumerate(mass):
        if i >=20:
            ls = ":"
        elif i >=10:
            ls = "-."
        else:
            ls = "-"
        if m!=0:
            plt.plot(10**gL[m][:,0], gL[m][:,1], label="{:.0f} GeV".format(m), ls=ls)
    plt.axhline(1.35, color="k", ls=":")
    plt.axhline(0, color="k", ls="-")
    plt.xlim(1e-28, 1e-20)
    plt.ylim(-1, 5)
    plt.xscale("log")
    plt.legend()
    plt.xlabel(r"$\langle \sigma v \rangle$ [cm$^{3}$/s]", fontsize=15)
    plt.ylabel(r"log($\mathcal{L}_{max}$) - log($\mathcal{L}$)", fontsize=15)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)


def plotULcurve(Input, label=None, ax=None, addRelic=False, units="GeV", **kwargs):
    if Input == "MAGIC":
        ul_magic = np.load(REF_DIR+"segue_1_MAGIC.npy")
        ul = np.asarray([ul_magic[:,0], np.log10(ul_magic[:,1])]).T
        if label == None:
            label="MAGIC (2014)"
        kwargs["ls"] = kwargs.get("ls", "-.")
    elif Input == "VERITAS":
        ul_veritas = np.load(REF_DIR+"segue_1_VERITAS.npy")
        ul = np.asarray([ul_veritas[:,0], np.log10(ul_veritas[:,1])]).T
        if label == None:
            label="VERITAS (2017)"
        kwargs["ls"] = kwargs.get("ls", "-.")
    elif Input == "HAWC":
        ul_hawc = np.load(REF_DIR+"segue_1_HAWC.npy")
        ul = np.asarray([ul_hawc[:,0], np.log10(ul_hawc[:,1])]).T
        if label == None:
            label="HAWC (2017)"
        kwargs["ls"] = kwargs.get("ls", "-.")
    elif Input == "HAWC_bb":
        ul_hawc = np.load(REF_DIR+"segue_1_HAWC_bb.npy")
        ul = np.asarray([ul_hawc[:,0], np.log10(ul_hawc[:,1])]).T
        if label == None:
            label="HAWC (2017)"
        kwargs["ls"] = kwargs.get("ls", "-.")
    elif Input == "VEGAS":
        ul_v = np.load(REF_DIR+"segue_1_VEGAS.npy")
        ul = np.asarray([ul_v[:,0], np.log10(ul_v[:,1])]).T
    elif Input != None:
        if type(Input) == str:
            if ".npy" not in Input:
                Input = Input+".npy"

            try:
                ul = np.load(OUTPUT_DIR+Input, allow_pickle=True)
            except:
                ul = np.load(Input, allow_pickle=True)
        elif type(Input) == np.ndarray:    
            ul = Input
    else:
        print("[Error] Upper limits cannot be imported.")
        return

    ul[:,1] = np.nan_to_num(ul[:,1])
    ul = ul[ul[:,1] != 0]
    
    if ax==None:
        ax = plt.gca()

    if units == "TeV":
        ul[:,0] = ul[:,0]/1e3

    if np.average(ul[:,1]) >0 :
        ul[:,1] = np.log10(ul[:,1])

    ax.plot(ul[:,0], 10**ul[:,1], label=label, **kwargs)
    ax.set_title(r"$\langle \sigma v \rangle$ 95% UL curve", fontsize=15)

    ymin = 10**(round(min(ul[:,1]))-1.5)
    ymax = 10**(round(max(ul[:,1]))+1.5)
    if addRelic:
        ax.axhline(1e-26, ls="-.", color="gray", label="Thermal relic DM")
        ax.fill_between([0, 1e5], 0, 1e-26, color="gray", alpha=0.5)
        ax.set_xlim(min(ul[:,0])/2, max(ul[:,0])*1.5)
        ymin = 5e-27

    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    if units == "TeV":
        ax.set_xlabel(r"$M_{\chi}$ [TeV]", fontsize=15)
    else:
        ax.set_xlabel(r"$M_{\chi}$ [GeV]", fontsize=15)
    ax.set_ylabel(r"$\langle \sigma v \rangle$ [cm$^{3}$/s]", fontsize=15)
    ax.grid(b=True, which="major")
    ax.grid(b=True, which="minor", ls=":", lw=0.5)
    ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')


def plotExpULcurve(filename=None, dwarf=None, package=None, channel = None, ax=None, addTheta=False, label="Expected Line", version="all", mean_only=False, add_mean=False, units="GeV", which=[68, 95], export=False, **kwargs):
    if addTheta:
        dim = "2D"
    else:
        dim = "1D"

    if filename == None:
        if version == "all":
            filename = "{}_{}_{}_{}_exp.npy".format(package, dwarf, channel, dim)
        else:
            filename = "{}_{}_{}_{}_{}_exp.npy".format(package, dwarf, channel, dim, version)
    else:
        if ".npy" not in filename:
            filename += ".npy"

    if os.path.isfile(OUTPUT_DIR+filename):
        uls = np.load(OUTPUT_DIR+filename, allow_pickle=True).item()
    else:
        print("[Error] Check your package, dwarf and channel.")
        return

    mass = list(uls.keys())
    mean_val = []
    error_cont = []
    for m in mass:
        if len(uls[m]) == 0:
            mean_val.append(np.nan)
            error_cont.append([np.nan, np.nan, np.nan, np.nan])
        else:
            mean_val.append(10**np.percentile(uls[m], 50))
            error_cont.append([10**np.percentile(uls[m], 16), 10**np.percentile(uls[m], 84), 10**np.percentile(uls[m], 2.5), 10**np.percentile(uls[m], 97.5)])
    
    mass=np.asarray(mass)
    if units == "TeV":
        mass = mass/1e3
    mean_val = np.asarray(mean_val)
    error_cont = np.asarray(error_cont)

    #etc = plt.plot(mass, mean_val)
    if ax == None:
        ax = plt.gca()
    
    if mean_only:
        ax.plot(mass, mean_val, label=label, **kwargs)

        if export:
            data = np.asarray([mass, mean_val]).T
            np.save(filename.split(".")[0]+"_plot", data)
    else:
        if add_mean:
            ax.plot(mass, mean_val, label=label, **kwargs)
        if 95 in which:
            etc = ax.plot(mass, error_cont[:,2], alpha=0.5, ls="--", **kwargs)    
            ax.plot(mass, error_cont[:,3], alpha=0.5, ls="--", color = etc[0].get_color())
        if 68 in which:
            if 95 in which:
                ax.fill_between(mass, error_cont[:,0], error_cont[:,1], color = etc[0].get_color(), alpha=0.2, label=label)
            else:
                ax.fill_between(mass, error_cont[:,0], error_cont[:,1], alpha=0.2, label=label)
        if export:
            data = np.asarray([mass, mean_val, error_cont[:,0], error_cont[:,1], error_cont[:,2], error_cont[:,3]]).T
            np.save(filename.split(".")[0]+"_plot", data)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if units == "GeV":
        ax.set_xlabel(r"$M_{\chi}$ [GeV]", fontsize=15)
    elif units == "TeV":
        ax.set_xlabel(r"$M_{\chi}$ [TeV]", fontsize=15)
    ax.set_ylabel(r"$\langle \sigma v \rangle$ [cm$^{3}$/s]", fontsize=15)
    ax.grid(b=True, which="major")
    ax.grid(b=True, which="minor", ls=":", lw=0.5)
    ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

def plotDeviation(Input, expectedLine=None, ax=None, version="all", **kwargs):

    if Input != None:
        if type(Input) == str:
            if ".npy" not in Input:
                Input = Input+".npy"
            try:
                ul = np.load(OUTPUT_DIR+Input, allow_pickle=True)
            except:
                ul = np.load(Input, allow_pickle=True)
        elif type(Input) == np.ndarray:    
            ul = Input
    else:
        print("[Error] Upper limits cannot be imported.")
        return

    props=re.findall("([a-zA-Z0-9]+)", Input)

    if len(props) == 7:
        package = props[0]
        dwarf = props[1]+"_"+props[2]
        channel = props[3]
        dim = props[4]
    elif len(props) == 6:
        package = props[0]
        dwarf = props[1]
        channel = props[2]
        dim = props[3]

    if expectedLine == None:
        if version == "all":
            filename = "{}_{}_{}_{}_exp.npy".format(package, dwarf, channel, dim)
        else:
            filename = "{}_{}_{}_{}_{}_exp.npy".format(package, dwarf, channel, dim, version)
    else:
        if ".npy" not in expectedLine:
            filename = expectedLine+".npy"
            
    if os.path.isfile(OUTPUT_DIR+filename):
        ul_exp = np.load(OUTPUT_DIR+filename, allow_pickle=True).item()
    else:
        print("[Error] Expected line is not imported. Check your inputs.")
        return

    sigma = []
    for M, ul_m in ul:
        if np.isnan(ul_m):
            continue
        sigma.append([M, norm_dist.ppf(sum(ul_exp[M]<ul_m)/len(ul_exp[M]))])

    sigma = np.asarray(sigma)

    refx = ul[:,0][~np.isnan(ul[:,1])]
    refy = np.zeros(len(ul[:,0]))[~np.isnan(ul[:,1])]

    if ax==None:
        ax = plt.gca()

    ax.plot(refx, refy, color="gray", **kwargs)
    ax.fill_between(refx, refy+1,  refy-1, color="gray", alpha=0.3)
    ax.plot(refx, refy+2, ls=":", c="gray")
    ax.plot(refx, refy-2, ls=":", c="gray")
    ax.plot(sigma[:,0], sigma[:,1]) 
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(r"$M_{\chi}$ [GeV]", fontsize=13)
    ax.set_ylabel(r"Deviation [$\sigma$]", fontsize=13)
    ax.set_xscale("log")
    #ax.legend(fontsize=12)

def plotPublication(channel, print_chan=True, textloc=0.5, **kwargs):
    if channel not in ["bbar", "tt"]:
        print("[Error] Choose other channel: tt, or bbar.")
        return

    plotULcurve(f"fermi_6y_{channel}", label="Fermi-LAT (2015; 6y)", ls="--", **kwargs)
    plotULcurve(f"magic_354h_{channel}", label="MAGIC (2022; 354h)", ls="--", **kwargs)
    plotULcurve(f"veritas_216h_{channel}", label="VERITAS (2017; 216h)", ls="--", **kwargs)
    plotULcurve(f"hess_80h_{channel}", label="H.E.S.S. (2020; 80h)", ls="--", **kwargs)
    plotULcurve(f"hawc_1038d_{channel}", label="HAWC (2020; 1038d)", ls="--", **kwargs)
    if print_chan:
        ax = plt.gca()
        if channel == "bbar":
            plt.text(0.9, textloc, r"$\chi\chi \rightarrow b\bar{b}$", fontsize=15, ha="right", transform=ax.transAxes)
        elif channel == "tt":
            plt.text(0.9, textloc, r"$\chi\chi \rightarrow \tau^{+}\tau^{-}$", fontsize=15, ha="right", transform=ax.transAxes)
    plt.legend(loc=4)
    plt.ylim(8e-27, 2e-20)

def plotUnitarity(composite=[1e-1, 1e-2, 1e-3]):
    vrel = 2.e-5
    TeV2cm3s = 1.1673299710900705e-23

    ### s-wave Unitarity limit ###
    # This is equation (10) of [Griest, Kamionkowski 1990], with J=0 for s-wave
    def slim(m):
        "m [TeV]"
        return TeV2cm3s*(4.*np.pi)/(m**2.*vrel)

    # Add in the finite size, which is (5) in https://arxiv.org/pdf/2203.06029.pdf, or (16) in Kamionkowski and Griest (but they expand in the last step)

    ### Composite Unitarity limit ###
    # This is equation (16) of [Griest, Kamionkowski 1990], but without the approximation that Jmax >> 1 as they use in the final step
    def Rlim(m, Rinv):
        "m [TeV], Rinv [TeV]"
        R = 1./Rinv
        return TeV2cm3s*(4.*np.pi)/(m**2.*vrel)*(1.+m*vrel*R)**2.

    #rescale=1.1
    #plot_h = 10/rescale
    #plot_w = 8/rescale
    ax = plt.gca()

    ax.set_xlabel(r'$M_{\chi}$ [TeV]',fontsize=13)
    ax.set_ylabel(r'$\langle \sigma v \rangle$ [cm$^3$/s]',fontsize=13)

    #mpl.rcParams['lines.dashed_pattern'] = 7.5, 7.5
    ax.axvline(194.,ls="-",c='k',lw=0.8)
    ax.fill_between([10,194],[1.e-28,1.e-28],[1.e-16,1.e-16],color=(0.8,0.8,0.8),alpha=0.3)

    ax.plot([10.,194],[2.4e-26,2.4e-26],c='red',lw=1.5)

    #mpl.rcParams['lines.dotted_pattern'] = 1.1, 2.5
    mv=np.logspace(1.,np.log10(4.e4),100)
    cv="k"

    for com in composite:
        ax.plot(mv,Rlim(mv,com),ls=(0, (3, 1, 1, 1, 1, 1)),c=cv,lw=0.8,zorder=2)
        if com == 1e-1:
            ax.text(3.5e4,8e-25,r'$R=(100~{\rm GeV})^{-1}$',fontsize=11,color="k", ha="right")
        elif com == 1e-2:
            ax.text(3.5e4,5e-23,r'$R=(10~{\rm GeV})^{-1}$',fontsize=11,color="k", ha="right")
        elif com == 1e-3:
            ax.text(3.5e4,5e-21,r'$R=(1~{\rm GeV})^{-1}$',fontsize=11,color="k", ha="right")
    
    ax.plot([10.,4.e4],[slim(10.),slim(4.e4)],c=cv,lw=1.5,zorder=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Restore y ticks
    #locmaj = mpl.ticker.LogLocator(base=10, numticks=1000)
    #ax.yaxis.set_major_locator(locmaj)

    #locmin = mpl.ticker.LogLocator(base=10.0, subs=np.linspace(0, 1.0, 11)[1:-1], numticks=1000)
    #ax.yaxis.set_minor_locator(locmin)
    #ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    # Text
    #plt.text(1.15e1,1.5e-18,r'Unitarity Limits',fontsize=13)

    ax.text(70,6.e-27,r'Thermal',fontsize=13, ha="center")
    ax.text(70,2.e-27,r'$\langle \sigma v \rangle$',fontsize=13, ha="center")
     
    #ax.text(200,3.e-28,r'Non-Thermal Relic',fontsize=13,color=(0.3,0.3,0.3))
    ax.text(1.1e4, 2e-26,r'Partial-Wave Unitarity',fontsize=13,color='k',rotation=339, ha="right")
    ax.text(3e4,4e-18,r'Composite Unitarity',fontsize=13,color="k", ha="right")

    #plt.tight_layout()
    #plt.legend()
    #plt.xlabel(r'$M_{\chi}$ [TeV]',fontsize=15)
    ax.set_xlim(25, 4e4)
    ax.set_ylim(1e-28, 1e-16)
    ax.legend(loc=2, fontsize=10)
    
def plotUnitarityR(Input, label=None, inv=True, vrel=2.e-5, units="GeV", **kwargs):
    TeV2cm3s = 1.1673299710900705e-23
    if type(Input) == str:
        if ".npy" not in Input:
            Input = Input+".npy"

    ### s-wave Unitarity limit ###
    # This is equation (10) of [Griest, Kamionkowski 1990], with J=0 for s-wave
    def slim(m):
        "m [TeV]"
        return TeV2cm3s*(4.*np.pi)/(m**2.*vrel)

    # Add in the finite size, which is (5) in https://arxiv.org/pdf/2203.06029.pdf, or (16) in Kamionkowski and Griest (but they expand in the last step)

    ### Composite Unitarity limit ###
    # This is equation (16) of [Griest, Kamionkowski 1990], but without the approximation that Jmax >> 1 as they use in the final step
    def CSlim(m, CS):
        "m [TeV], CS [cm^3/s]"
        R = ((CS/(4.*np.pi*TeV2cm3s*vrel))**(1/2.)-1./(m*vrel))
        return 1./R

    ul = np.load(OUTPUT_DIR+Input, allow_pickle=True)

    ul[:,1] = np.nan_to_num(ul[:,1])
    ul = ul[ul[:,1] != 0]
    ul[:,0] = ul[:,0]/1e3

    if np.average(ul[:,1]) >0 :
        ul[:,1] = np.log10(ul[:,1])

    intp = interp1d(np.log10(ul[:,0]), ul[:,1])
    intp_M = np.linspace(1, 4.4, 1000)
    mass = intp_M 
    intp_uls = intp(intp_M)
    ax = plt.gca()

    Rinv = CSlim(10**intp_M, 10**intp_uls)
    if units == "GeV":
        if inv:
            intp_M = np.asarray([intp_M[Rinv>0][0]]+intp_M[Rinv>0].tolist())
            y = Rinv[Rinv>0]*1e3
            y = [1e3] + y.tolist()
            ax.plot(10**intp_M,y, label=label, ls=kwargs.get("ls"))
            ax.set_ylabel(r'$1/R$ [GeV]',fontsize=13)
        else:
            intp_M = np.asarray([intp_M[Rinv>0][0]]+intp_M[Rinv>0].tolist())
            y = 1./Rinv[Rinv>0]/1e3
            y = [1e-5] + y.tolist()
        
            ax.plot(10**intp_M,y, label=label, ls=kwargs.get("ls"))
            ax.set_ylabel(r'$R$ [GeV$^{-1}$]',fontsize=13)
    elif units == "fm":
        hbar = 6.582119569e-16
        c = 299792458
        fminv = 1./Rinv*hbar*c*1e3
        intp_M = np.asarray([intp_M[Rinv>0][0]]+intp_M[Rinv>0].tolist())
        fminv = [1e-5] + fminv[Rinv>0].tolist()
        y = fminv
        ax.plot(10**intp_M,y, label=label, ls=kwargs.get("ls"))
        ax.set_ylabel(r'$R$ [fm]',fontsize=13)

    ax.set_xlabel(r'$M_{\chi}$ [TeV]',fontsize=13)


    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(b=True, which="major")
    ax.grid(b=True, which="minor", ls=":", lw=0.5)
    ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    return intp_M, y

