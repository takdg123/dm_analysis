from DarkMatter import *
from tqdm import trange

mass = np.logspace(np.log10(200), np.log10(30000), 6)

#mass = np.logspace(np.log10(1e5), np.log10(3e7), 3)
runs = 300
#chan = ["ZZ","nue"]

uls = {}
 
for c in ["ee", "uu", "tt"]:
    filename = const.OUTPUT_DIR+f"EventDisplay_stacked_sys_{c}_1D_final_low_corrected"
    # if os.path.exists(filename+".npy"):
    #     print(c)
    #     continue

    for i in trange(runs):
        
        try:
            ul = Likelihood.combinedUpperLimits(c, mass=mass, package="EventDisplay", ext=True,
                                            dwarfs = const.ListOfDwarf, seed=130,
                                            method=2, averagedIRF=True, DM_spectra="PPPC", 
                                            addTheta=False, useBias=True, correction=True, bkgModel="gaus",
                                            returnTS=True, sys=True, verbosity=False)
        except:
            continue
        for u in ul:
            if u[0] in uls.keys():
                
                uls[int(u[0])].append(u[1])
            else:
                uls[int(u[0])] = [u[1]]

    np.save(const.OUTPUT_DIR+f"EventDisplay_stacked_sys_{c}_1D_final_low_corrected", uls)