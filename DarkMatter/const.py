import numpy as np
import os
from pathlib import Path
from astropy.table import Table

pcTocm   = 3.086e18                         # 1pc in cm
MSunToGeV= 1.115e57                         # 1 solar equivalent in GeV
rhodlToGeVcm2=MSunToGeV/(pcTocm**2)       # for channel (not yet implemented below)
rho2dlToGeV2cm5=MSunToGeV**2/(pcTocm**5)  # for annihilation

DM_DIR = os.environ.get('DM')
REF_DIR = DM_DIR+"/RefData/"
DATA_DIR = DM_DIR+"/Data/"
OUTPUT_DIR = DM_DIR+"/Output/"
SCRIPT_DIR = str(Path(__file__).parent.absolute())

if not(os.path.isdir(OUTPUT_DIR+"/__temp__")):
	os.system("mkdir "+OUTPUT_DIR+"/__temp__")

if not(os.path.isdir(REF_DIR)):
	os.system("mkdir "+REF_DIR)

if not(os.path.isdir(DATA_DIR)):
	os.system("mkdir "+DATA_DIR)

ListOfDwarf = ['Bootes_I', 'ComBer', 'CVn_I', 'CVn_II', 
 'Draco_II', 'Hercules','Leo_I', 'Leo_II',  'Leo_IV', 'Leo_V',
 'Segue_1', 'Segue_2', 'Sextans', 'Triangulum_II',
 'UMa_I', 'UMa_II', 'UMi']

ListOfChannel = ["ee", "uu", "tt", "ttbar", "bbar", "WW", "ZZ", "gamma", "nue"]
ListOfChannelLabel = [r"$e^{+}e^{-}$", r"$\mu^{+}\mu^{-}$", 
    r"$\tau^{+}\tau^{-}$", r"$t\bar{t}$", r"$b\bar{b}$",
    r"$W^{+}W^{-}$", r"$ZZ$", r"$\gamma\gamma$", r"$\nu \bar{\nu}_e$"]

thKnots = np.load(SCRIPT_DIR+"/npy/thKnots.npy")
eVJbins = np.load(SCRIPT_DIR+"/npy/eVJbins.npy")
eEDJbins = np.load(SCRIPT_DIR+"/npy/eEDJbins.npy")
thKnots_jp = np.load(SCRIPT_DIR+"/npy/thKnots_jp.npy")

eRecBins = np.load(SCRIPT_DIR+"/npy/eRecBin.npy")
eMCBins = np.load(SCRIPT_DIR+"/npy/eMCBin.npy")
eEDBins = np.load(SCRIPT_DIR+"/npy/eEDBin.npy")
eDispBins = np.load(SCRIPT_DIR+"/npy/eDispBins.npy")

mass4gamma = 10**np.arange(1.1, 9.11, step=0.2)
mass4gamma_disp = np.load(SCRIPT_DIR+"/npy/mass4gamma.npy")
mass4gamma_vegas = np.load(SCRIPT_DIR+"/npy/mass4gamma_vegas.npy")


energyEdges = 10**np.arange(1, 7.01, step=0.2)
#energyEdges = 10**np.linspace(1, 7, 61)
defaultNum_NFW = {"Segue_1": 295, "Draco_II": 38, "Bootes_I": 54, "UMi": 113}
defaultNum = np.load(SCRIPT_DIR+"/npy/default_num.npy", allow_pickle=True).item()

HDM_Channel2Num = {
    "bbar": 5,
    "ttbar": 6,
    "ee": 11,
    "uu": 13,
    "tt": 15,
    "WW": 24,
    "ZZ": 23,
    "gamma": 22,
    "nue": 12,
    "numu": 14,
    "nutau": 16,
}

PPPC_Channel2Num = {
    "bbar": 13,
    "ttbar": 14,
    "ee": 4,
    "uu": 7,
    "tt": 10,
    "WW": 17,
    "ZZ": 20,
    "gamma": 22,
    "nue": 24,
    
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

th2cut_ext = 0.012