import numpy as np

from ..const import OUTPUT_DIR

from scipy.interpolate import interp1d, interp2d

from .ff import forwardFolding

from .. import utils

from numba import njit

@njit
def get_first_index_nb(A, k):
    for i in range(len(A)):
        if A[i] > k:
            return i
    return -1


def fcn(npar, gin, f, par, iflag):
    log = 0
    b = par[0]
    signu = par[1]
    seed = int(par[2])

    args = np.load(OUTPUT_DIR+"/__temp__/temp_args_{}.npy".format(seed), allow_pickle=True)
    args = args.item()
    
    g = sum(args["g"])*10**signu
    logl = args["N_off"]*np.log(b) - g - (args["alpha"]+1)*b

    P_tot = 0
    for i in range(len(args["g"])):
        b_v = (args["tau"][i]*b)

        P = args["alpha"]*b_v*args["p_off"][i] + (args["g"][i]*10**signu)*args["p_on"][i]
        P = P[P>0]
        P_tot += sum(np.log(P))

    logl += P_tot
    f.value = -logl


def simplefcn(npar, gin, f, par, iflag):
    log = 0
    b = par[0]
    signu = par[1]
    seed = int(par[2])

    args = np.load(OUTPUT_DIR+"/__temp__/temp_args_{}.npy".format(seed), allow_pickle=True)
    args = args.item()
    
    g = sum(args["g"])*10**signu
    logl = args["N_off"]*np.log(b) - g - (args["alpha"]+1)*b + args["N_on"]*np.log(g+args["alpha"]*b)
    f.value = -logl
    

def binnedfcn(npar, gin, f, par, iflag):
    log = 0
    b = par[0]
    signu = par[1]
    seed = int(par[2])

    args = np.load(OUTPUT_DIR+"/__temp__/temp_args_{}.npy".format(seed), allow_pickle=True)
    args = args.item()

    g = sum(args["g"])*10**signu
    g_arr = args["hSig"]*10**signu
    b_arr = args["hOff"]/sum(args["hOff"])*b

    valid = (b_arr !=0)

    g_arr[g_arr < 1e-10] = 0
    g_arr = g_arr[valid]
    b_arr = b_arr[valid]

    if args["alpha_array"] is not None:
        logl = - g - sum((args["alpha_array"][valid]+1)*b_arr) + sum(args["hOff"][valid]*np.log(b_arr) + args["hOn"][valid]*np.log((g_arr+args["alpha_array"][valid]*b_arr)))
    else:
        logl = - g - (args["alpha"]+1)*b + sum(args["hOff"][valid]*np.log(b_arr) + args["hOn"][valid]*np.log((g_arr+args["alpha"]*b_arr)))

    f.value = -logl


def stackedfcn(npar, gin, f, par, iflag):
    logl = 0
    signu = par[0]
    numDwarfs = int(par[1])
    

    for i in range(numDwarfs):
        b = int(par[i+2])
        seed = int(par[numDwarfs+2+i])
        
        args = np.load(OUTPUT_DIR+"/__temp__/temp_args_{}.npy".format(seed), allow_pickle=True)
        args = args.item()
        
        g = sum(args["g"])*10**signu
        logl += args["N_off"]*np.log(b) - g - (args["alpha"]+1)*b

        P_tot = 0
        for i in range(len(args["g"])):
            b_v = (args["tau"][i]*b)

            P = args["alpha"]*b_v*args["p_off"][i] + (args["g"][i]*10**signu)*args["p_on"][i]
            P = P[P>0]
            P_tot += sum(np.log(P))

        logl += P_tot

    f.value = -logl

def mfcn(npar, gin, f, par, iflag):
    logl = 0
    b = par[0]
    signu = par[1]
    seed = int(par[2])

    args = np.load(OUTPUT_DIR+"/__temp__/temp_args_{}.npy".format(seed), allow_pickle=True)
    args = args.item()
    
    N_on = args["N_on"]
    N_off = args["N_off"]
    g = args["g"]*10**signu

    N_obs = N_on + N_off
    N_est = (args["alpha"]*N_off+g)+N_off

    logl = N_obs*np.log(N_est)-N_est-N_on*np.log(args["alpha"]*N_off+g)
    
    P = (args["alpha"]*N_off*args["p_off"][args["p_off"]>0] + g*args["p_on"][args["p_off"]>0])
    P = P[P>0]    
    
    logl = logl + sum(np.log(P)) 

    f.value = -logl


def getLikelihood(signu, b, seed=None):
    if seed==None:
        return 0

    log = 0

    args = np.load(OUTPUT_DIR+"/__temp__/temp_args_{}.npy".format(seed), allow_pickle=True)
    args = args.item()
    
    g = sum(args["g"])*10**signu
    logl = args["N_off"]*np.log(b) - g - (args["alpha"]+1)*b

    P_tot = 0
    for i in range(len(args["g"])):
        b_v = (args["tau"][i]*b)

        P = args["alpha"]*b_v*args["p_off"][i] + (args["g"][i]*10**signu)*args["p_on"][i]
        P = P[P>0]
        P_tot += sum(np.log(P))

    logl += P_tot
    return -logl


def fcn_bkg(npar, gin, f, par, iflag):
    log = 0
    
    seed = int(par[2])
    m_num = int(par[3])

    if m_num == 2:
        pars = (par[0], par[1], par[4], par[5])
        model = utils.BKNPOWER
    else:
        pars = (par[0], par[1])
        model = utils.POWERLAW
    
    args = np.load(OUTPUT_DIR+"/__temp__/temp_args_{}.npy".format(seed), allow_pickle=True)
    args = args.item()
    bkg = args["hOff"]

    if args["package"] == "EventDisplay":
        idx = bkg.argmax()+1
    elif args["package"] == "VEGAS":
        idx = get_first_index_nb(args["energies"], 1000)

    signal = forwardFolding(model, pars, args["dwarf"], package=args["package"])
    mx, my = utils.getArray(signal)

    bkg = bkg[idx:]
    my = my[idx:]

    Cash = 2*np.sum(my[bkg!=0]-bkg[bkg!=0]+bkg[bkg!=0]*(np.log(bkg[bkg!=0])-np.log(my[bkg!=0])))
        
    f.value = Cash

