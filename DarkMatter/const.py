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

ListOfDwarfLabel = [r'B\"ootes', 'Coma Berenices', 'CVn I', 'CVn II', 
 'Draco II', 'Hercules I','Leo I', 'Leo II',  'Leo IV', 'Leo V',
 'Segue 1', 'Segue 2', 'Sextans I', 'Triangulum II',
 'Ursa Major I', 'Ursa Major II', 'Ursa Minor']

ratio_factor = {"Bootes_I": 2.95, "ComBer": 4.47, "CVn_I": 1.15, 
    "CVn_II": 2.88, "Draco_II": 1, "Hercules": 0.85, 
    "Leo_I": 1.38, "Leo_II": 2.69, "Leo_IV": 0.58,
    "Leo_V": 0.62, "Segue_1": 2.82, "Segue_2": 0.10,
    "Sextans": 0.74, "Triangulum_II": 1, "UMa_I":0.48,
    "UMa_II":4.27, "UMi":3.02}

ListOfChannel = ["ee", "uu", "tt", "ttbar", "bbar", "WW", "ZZ", "gamma", "nue"]
ListOfChannelLabel = {"ee": r"$e^{+}e^{-}$", "uu": r"$\mu^{+}\mu^{-}$", 
    "tt": r"$\tau^{+}\tau^{-}$", "ttbar": r"$t\bar{t}$", "bbar": r"$b\bar{b}$",
    "WW": r"$W^{+}W^{-}$", "ZZ": r"$ZZ$", "gamma": r"$\gamma\gamma$", "nue": r"$\nu \bar{\nu}_e$"}

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

wino_mass = [800.0, 1260.0, 2000.0, 3160.0, 5010.0, 7940.0,
12590.0, 20000.0, 32000.0, 50000.0, 79000.0, 126000.0, 200000.0,
300000.0]

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

plot_properties={'Bootes_I': ['Bootes', ':'],
 'ComBer': ['Coma Berenices', ':'],
 'CVn_I': ['CVn I', ':'],
 'CVn_II': ['CVn II', ':'],
 'Draco_II': ['Draco II', ':'],
 'Hercules': ['Hercules I', ':'],
 'Leo_I': ['Leo I', ':'],
 'Leo_II': ['Leo II', ':'],
 'Leo_IV': ['Leo IV', ':'],
 'Leo_V': ['Leo V', ':'],
 'Segue_1': ['Segue 1', '--'],
 'Segue_2': ['Segue 2', '--'],
 'Sextans': ['Sextans', '--'],
 'Triangulum_II': ['Triangulum II', '--'],
 'UMa_I': ['Ursa Major I', '--'],
 'UMa_II': ['Ursa Major II', '--'],
 'UMi': ['Ursa Minor', '--']}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

th2cut_ext = 0.012