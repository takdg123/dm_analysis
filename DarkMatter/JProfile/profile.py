import numpy as np

from ROOT import TMath
from .. import const
from ..utils import getArray

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import scipy.integrate as integrate

from astropy.table import Table

import random

from ROOT import TGraph

pc = 1.56392308e+32
Msun = 1.1158518958483142e+57

# Navarro-Frenk-White profile (Zhao, H., 1996)
AlexDMProfile = lambda x, rhos, rs, alpha, beta, gamma: rhos*(x/rs)**(-gamma) * (1+(x/rs)**alpha)**((gamma-beta)/alpha)
TruncationRadius = {"Segue_1":138.58, "Draco":1865.59, "Bootes_I":543.66, "UMi":1580.36}
#Distance2Dwarf = {"segue_1":2.3e4, "draco":7.6e4, "bootes":6.6e4, "ursa_minor":7.6e4, "uma_ii":3.2e4}
propsNames=["rhos", "log10(rs)", "alpha", "beta", "gamma"]
goodNum_NEW = {"Segue_1": 295, "Draco_II": 38, "Bootes_I": 54, "UMi": 113}


prevPars = {
    "Segue_1": [1.77958937145, np.log10(310.75663656), 0.544110936127, 4.35536608401, 0.642320059857],    
    "Draco": [0.00820912015408, np.log10(2563.36089842), 1.96440143647, 6.08744867073, 1.11688100724],
    "Bootes_I": [0.000665784127071, np.log10(12105.0798317), 2.80901997094, 4.87104498245, 1.0804979449],
    "UMi": [0.560401316631, np.log10(355.432472672), 2.36557038861, 8.76508618936, 0.0117481249969],
}

convt = {"Segue_1": "segue_1", "Draco": "draco", "Bootes_I": "bootes", "UMi":"ursa_minor"}

def Distance2Dwarf(dwarf):
    if (dwarf == 'Carina'):
        Dis = 105.
    elif (dwarf == 'Draco'):
        Dis = 76.
    elif (dwarf == 'Fornax'):
        Dis = 147.
    elif (dwarf == 'Leo_I'):
        Dis = 254.
    elif (dwarf == 'Leo_II'):
        Dis = 233.
    elif (dwarf == 'Sagittarius'):
        Dis = 26.
    elif (dwarf == 'Sculptor'):
        Dis = 86.
    elif (dwarf == 'Sextans'):
        Dis = 86.
    elif (dwarf == 'UMi'):
        Dis = 76.
    elif (dwarf == 'Aquarius_2'):
        Dis = 107.9
    elif (dwarf == 'Bootes_I'):
        Dis = 66.
    elif (dwarf == 'Bootes_II'):
        Dis = 42.
    elif (dwarf == 'CVn_I'):
        Dis = 218.
    elif (dwarf == 'CVn_II'):
        Dis = 160.
    elif (dwarf == 'Carina_II'):
        Dis = 36.2
    elif (dwarf == 'ComBer'):
        Dis = 44.
    elif (dwarf == 'Draco_II'):
        Dis = 20.
    elif (dwarf == 'Eridanus_II'):
        Dis = 380.
    elif (dwarf == 'Grus_I'):
        Dis = 120.
    elif (dwarf == 'Hercules'):
        Dis = 132.
    elif (dwarf == 'Horologium_I'):
        Dis = 79.
    elif (dwarf == 'Hyrdus_1'):
        Dis = 27.6
    elif (dwarf == 'Leo_IV'):
        Dis = 154.
    elif (dwarf == 'Leo_T'):
        Dis = 417.
    elif (dwarf == 'Leo_V'):
        Dis = 178.
    elif (dwarf == 'Pegasus_III'):
        Dis = 215.
    elif (dwarf == 'Pisces_II'):
        Dis = 182.
    elif (dwarf == 'Reticulum_II'):
        Dis = 30.
    elif (dwarf == 'Segue_1'):
        Dis = 23.
    elif (dwarf == 'Segue_2'):
        Dis = 35.
    elif (dwarf == 'Triangulum_II'):
        Dis = 30.
    elif (dwarf == 'Tucana_II'):
        Dis = 57.
    elif (dwarf == 'Tucana_III'):
        Dis = 25.
    elif (dwarf == 'UMa_I'):
        Dis = 97.
    elif (dwarf == 'UMa_II'):
        Dis = 32.
    elif (dwarf == 'Willman_1'):
        Dis = 38.
    return Dis*1000

def NFWdwarfParam(dwarf):
    pars = np.load(const.REF_DIR+"/JProfile/profile_data.npy", allow_pickle=True).item()
    pars = pars[dwarf]
    pars = np.asarray(pars)
    pars_rearr = [0, 0, 0]
    pars_rearr[0] = pars[1]/(Msun/pc**3)
    pars_rearr[1] = pars[0]/pc
    pars_rearr[2] = pars[2]/pc
    pars_rearr = np.asarray(pars_rearr)
    prop_table = Table(pars_rearr.T, names = ["rhos", "rs", "rt"])
    return prop_table

def dwarfParam(dwarf):
    dwarf = convt[dwarf]
    props = np.genfromtxt(const.REF_DIR+"/JProfile/{}.dat".format(dwarf))
    mask = np.genfromtxt(const.REF_DIR+"JProfile/{}_filtermask.txt".format(dwarf), skip_header=True)
    props_good = (props[mask==1])
    prop_table = Table(props_good[:,[2,3,4,6,5]], names=propsNames)
    return prop_table

def goodPropNum(dwarf):
    dwarf = convt[dwarf]
    mask = np.genfromtxt(const.REF_DIR+"/JProfile/{}_filtermask.txt".format(dwarf), skip_header=True)
    return sum(mask==1)

def calcAlexDMProfile(props, r = 1, plotting = False):
    (rhos, rs, alpha, beta, gamma) = props
    rs = 10**rs
    return AlexDMProfile(r, rhos, rs, alpha, beta, gamma)

def calcNFWProfile(props, r=1.):
    j_val = lambda r, rhos, rs: (rhos/((r/rs)*(1+(r/rs))**2.))

    rhos = props[0]
    rs = props[1]
    rt = props[2]
    
    if (type(r) == np.float64) or (type(r) == float):
        if r>rt:
            return 0
        else:
            return j_val(r, rhos, rs)
    else:
        rad = r
        output = []
        for r in rad:
            if r>rt:
                output.append(0)
            else:
                output.append(j_val(r, rhos, rs))
        return np.asarray(output)

def getThMax(dwarf):
    r_t = TruncationRadius[dwarf]
    d = Distance2Dwarf(dwarf)
    thMax = np.arcsin(r_t/d)*TMath.RadToDeg()
    return thMax

def classicNFW(dwarf, seed, props=[]):
    los=lambda x, b, props: calcNFWProfile(props, np.sqrt(b**2+x**2))**2
    if len(props) == 0:
        if seed == -1:
            seed = random.randrange(0, 100000)
        prop_table = NFWdwarfParam(dwarf)
        props = prop_table[seed]
        return props, los
    else:
        return props, los

def generalizedNFW(dwarf, seed, props = [], verbose=False):
    los=lambda x, b, props: calcAlexDMProfile(props, np.sqrt(b**2+x**2))**2
    
    if seed == -1:
        seed = random.randrange(0, goodPropNum(dwarf)-1)
    elif seed == "b":
        props = prevPars[dwarf]

    if len(props) == 0:
        all_props = dwarfParam(dwarf)
        props = all_props[seed]
        if verbose:
            print("[Log] Dwarf parameters: ")
            print(props)
    elif len(props) == 5:
        props = tuple(props)
        if verbose:
            print("[Log] Dwarf parameters: ")
            print("          {:10} {:10} {:10} {:10} {:10}".format(*propsNames))
            print("     {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}".format(*props))
    else:
        print("[Error] Check your dwarf properties.")

    return props, los

def calcJProfile(dwarf, step=0.004, props = [], seed=-1, return_array=True, verbose=False, **kwargs):

    profile_type = kwargs.pop("general", False)
    if profile_type:
        props, los = generalizedNFW(dwarf, seed, props=props, verbose=verbose)
        r_t = TruncationRadius[dwarf]
    else:
        props, los = classicNFW(dwarf, seed, props=props)
        r_t = props[-1]

    theta = np.asarray([[th, step] for th in np.arange(step/2, 2, step=step)])
    
    theta_rad = theta*TMath.DegToRad()

    dJdOmega=np.zeros_like(theta_rad)
    dJdOmega[:,0] = theta[:,0]

    d = Distance2Dwarf(dwarf)

    for i, th in enumerate(theta_rad[:,0]):
        
        b = d*np.sin(th)
        if b < r_t:
            l = np.sqrt(r_t**2.- b**2.)
        else:
            l = 0

        val=integrate.quad(los, -l, l, args=(b, props))[0] * const.rho2dlToGeV2cm5
        if val <= 0:
            break
        else:
            dJdOmega[i][1]=val

    if verbose:
        J = calcJval(dwarf, gJProf=dJdOmega, props=props)
        print("[Log] J profile is {:.1e}.".format(J[-1][-1]))

    if return_array:
        return dJdOmega
    else:
        gdJdOmega = TGraph()
        for i, jo in enumerate(dJdOmega):
            gdJdOmega.SetPoint(i+1, jo[0], jo[1])
        return gdJdOmega
    
def calcJval(dwarf, seed=-1, gJProf=None, props=[], deg=None, **kwargs):
    if gJProf is None:
        gJProf = calcJProfile(dwarf, seed=seed, props=props, **kwargs)
    th = gJProf[:,0]
    dth = np.diff(gJProf[:,0])[0]
    th_rad = th * TMath.DegToRad()
    dth_rad = dth * TMath.DegToRad()
    J = np.cumsum(gJProf[:,1]*2*np.pi*np.sin(th_rad)*dth_rad)
    J = np.asarray([th, J]).T
    if deg is None:
        return J
    else:
        J_int = interp1d(J[:,0], J[:,1],  kind='slinear')
        return J_int(deg)

