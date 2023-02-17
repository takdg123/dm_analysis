import numpy as np

from ..utils import center_pt

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import scipy

def bkg_sm_1D(events, binEdges, eMin = 0):
	bkg = events[events[:,2]==0.0]
	cnts, binEdges = np.histogram(bkg[:,0], bins=binEdges)
	cnts = cnts*1.
	bin_ctr = center_pt(binEdges)
	max_idx = cnts.argmax()
	eMax = max(events[:,0][events[:,2]==1.0])
	eMax_bkg = max(events[:,0][events[:,2]==0.0])
	N_tot = sum(cnts)

	for i in range(len(binEdges)-1):
		if i < max_idx:
			continue
		if cnts[i] == 0:
			addBins = 1
			while (cnts[i] == 0) and (i+addBins<len(binEdges)-1):
				totCnts = 0
				for m in range(addBins+1):
					totCnts+=cnts[i+m]

				if totCnts == 0:
					addBins+=1
				else:
					for m in range(addBins+1):
						cnts[i+m] = totCnts*1./(addBins+1.)

	if eMax>eMax_bkg:
		n = 0
		for i, e in enumerate(binEdges[:-1]):
		    if i < cnts.argmax():
		        continue
		    if cnts[i] > 0:
		        prev_cnts = cnts[i]
		        prev_idx = i
		    elif e < eMax:
		        n +=1
		        max_idx = i
		    
		for i in range(prev_idx, max_idx+1):
		    cnts[i] = prev_cnts/(n+1)

	N_tot_arr = sum(cnts)

	if abs(N_tot-N_tot_arr)>1e-4:
		print("[Error] The number of re-distributed events ({:.0f}) is different from that of the original ({:.3f}).".format(N_tot_arr, N_tot))
	
	return np.asarray(cnts)


def bkg_sm_2D(events, eBinEdges, tBinEdges):
	cnts = []
	for i in range(len(tBinEdges)-1):
	    sub_events = events[(events[:,1] >= tBinEdges[i])*(events[:,1] < tBinEdges[i+1])]
	    cnts.append(bkg_sm_1D(sub_events, eBinEdges))
	return np.asarray(cnts).T

def pl_model(x, idx, Ntot, maxE, minE):
    E1 = min(x[x>minE])
    E2 = max(x[x<maxE])
    n = len(x[(x>minE)*(x<maxE)])
    w = 10**np.diff(np.log10(x))[0]
    r = w**idx
    fsum = ((r**n-1)/(r-1))
    N0 = Ntot/fsum/(E1**idx)
    return N0*x**idx

def bkg_gaus_1D(events, binEdges, sigma=1, alpha=1):
	bkg = events[events[:,2] == 0.0][:,0]
	y, x = np.histogram(bkg, bins=binEdges)
	y = y*alpha
	y_idx = y.argmax()
	#y_idx = [i for i, f in enumerate(y != 0) if f][0]
	filtered_y = scipy.ndimage.gaussian_filter(y[y_idx:], sigma=sigma)
	cnts = y[:y_idx].tolist() + filtered_y.tolist()
	return np.asarray(cnts)/alpha

def bkg_gaus_2D(events, eBinEdges, tBinEdges, sigma=1, alpha=1):
	cnts = []

	for i in range(len(tBinEdges)-1):
		sub_events = events[(events[:,1] >= tBinEdges[i]) * (events[:,1] < tBinEdges[i+1])]
		output = bkg_gaus_1D(sub_events, eBinEdges, sigma=sigma, alpha=alpha)
		cnts.append(output)
		
	cnts=np.asarray(cnts).T
	return cnts

def bkg_ex_1D(events, binEdges, eMin = 300, eMax = 10000, plotting=False, overlap=False, order=1):
	bkg = events[events[:,2] == 0.0][:,0]
	y, x = np.histogram(bkg, bins=binEdges)
	cumy= np.cumsum(y[::-1])[::-1]
	#cond = (cumy>min((max(cumy)*0.01), 4))*(np.arange(0,len(y))>(y.argmax()+1))
	cond = (cumy>min((max(cumy)*0.05), 4))*(np.arange(0,len(y))>=(y.argmax()+1))
	selectedx = center_pt(x)[cond]
	selectedy = cumy[cond]

	pfit = np.poly1d(np.polyfit(np.log10(selectedx), np.log10(selectedy), order))
	diff = (abs(10**pfit(np.log10(center_pt(x))) - cumy)/(10**pfit(np.log10(center_pt(x)))))[cond]
	eStart = selectedx[diff.argmin()]
	apply_bins = np.asarray(((range(len(y))>y.argmax())*(y==0))+(center_pt(x) > eStart))
	temp = [False]
	for i, b in enumerate(apply_bins[1:]):
		if x[i]>eMax:
			temp.append(True)
		elif (temp[i] == False)+(b==True):
			temp.append(b)
		elif y[i]>y[i-1]:
			temp.append(True)
		else:
			temp.append(not(b))


	apply_bins = np.asarray(temp)

	fCumy1 = list(cumy[~apply_bins])
	fCumy2 = list(10**pfit(np.log10(center_pt(x)))[apply_bins])
	fCumy = np.asarray(fCumy1+fCumy2)
	cnts = list(np.diff(fCumy[::-1])[::-1])+[0]

	if fCumy1[-1]<fCumy2[0]:
		cnts[len(fCumy1)-1] = cnts[len(fCumy1)]

	if plotting:
		f, ax = plt.subplots(1,2, figsize=(10, 4))
		ax[0].step(center_pt(x), fCumy, where="mid", label="Model")
		ax[0].plot(center_pt(x), 10**pfit(np.log10(center_pt(x))), ls="--", alpha=0.3, color="r", label="Fit")
		ax[0].step(center_pt(x), cumy, where="mid", label="Data")
		ax[0].set_xscale("log")
		ax[0].set_yscale("log")
		ax[0].set_xlabel("Energy")
		ax[0].set_ylabel("Cumulative counts")
		ax[0].axvline(center_pt(x)[apply_bins][0], ls=":", color="r", label="Apply model")
		maxy = 2*10**(round(np.log10(max(cumy)))+1)
		ax[0].fill_betweenx([1e-3, maxy], selectedx[0], selectedx[-1], color="r", alpha=0.1)
		ax[0].set_ylim(1e-3, maxy)
		ax[0].legend()

		ax[1].step(center_pt(x), cnts, where="mid", label="Model")
		ax[1].step(center_pt(x), y, where="mid", label="Data")
		ax[1].set_xscale("log")
		ax[1].set_yscale("log")
		ax[1].set_xlabel("Energy")
		ax[1].set_ylabel("Counts")
		ax[1].axvline(center_pt(x)[apply_bins][0], ls=":", color="r", label="Apply model")
		maxy = 2*10**(round(np.log10(max(y)))+1)
		ax[1].fill_betweenx([1e-3, maxy], selectedx[0], selectedx[-1], color="r", alpha=0.1)
		ax[1].set_ylim(1e-3, maxy)
		ax[1].legend()

		plt.show(block=False)

	if overlap:
		return np.asarray([center_pt(x), 10**pfit(np.log10(center_pt(x)))]).T
	else:
		return cnts

def bkg_ex_2D(events, eBinEdges, tBinEdges, eMin = 500, eMax = 2000, plotting=False, overlap=False):
	cnts = []

	for i in range(len(tBinEdges)-1):
		sub_events = events[(events[:,1] >= tBinEdges[i]) * (events[:,1] < tBinEdges[i+1])]
		if plotting:
			print("Theta = {}".format(tBinEdges[i]))

		output = bkg_ex_1D(sub_events, eBinEdges, eMin = eMin, eMax=eMax, plotting=plotting, overlap=overlap)
		
		if overlap:
			plt.plot(output[:,0], output[:,1])
			plt.xscale("log")
			plt.yscale("log")
			plt.xlabel("Energy")
			plt.ylabel("Cumulative counts")
		else:
			cnts.append(output)
		
	cnts=np.asarray(cnts).T
	return cnts

def bkg_alt_1D(events, binEdges, eMin = 500, eMax = 10000):
	bkg = events[events[:,2] == 0.0][:,0]
	cnts_ex = bkg_ex_1D(events, binEdges, eMin = eMin, eMax = eMax)
	cnts, x = np.histogram(bkg, bins=binEdges)

	for i, cnt in enumerate(cnts):
	    if i < cnts.argmax():
	        continue
	    if cnt == 0:
	        startIdx = i
	        break

	absDiff=[]
	for i in range(10):
	    absDiff.append(abs(sum(cnts[startIdx-i:]) - sum(cnts_ex[startIdx-i:])))


	startIdx -= np.asarray(absDiff).argmin()

	cnts_alt = []
	
	for i in range(len(cnts)):
		if i < startIdx:
			cnts_alt.append(cnts[i])
		else:
			cnts_alt.append(cnts_ex[i])
	
	return cnts_alt

def bkg_alt_2D(events, eBinEdges, tBinEdges, eMin = 500, eMax = 3000):
	cnts = []
	for i in range(len(tBinEdges)-1):
		sub_events = events[(events[:,1] >= tBinEdges[i]) * (events[:,1] < tBinEdges[i+1])]
		y_ex = bkg_alt_1D(sub_events, eBinEdges, eMin = eMin, eMax=eMax)
		cnts.append(y_ex)
	cnts=np.asarray(cnts).T
	return cnts
