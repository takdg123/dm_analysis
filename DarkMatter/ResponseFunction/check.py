import numpy as np
import os


def checkIRF(irf, Erec = [], normCheck=False, verbose=False):
	errFlag = False
	hEA = irf.EA
	if hEA.Class_Name() != "TGraph":
		if verbose: print("[Error] Dispersion matrix is wrong")
		errFlag = True

	hBias = irf.Bias
	if hBias.Class_Name() != "TH2D":
		if verbose: print("[Error] Energy bias is wrong")
		errFlag = True

	hPSF = irf.PSF
	if hPSF.Class_Name() != "TH2D":
		if verbose: print("[Error] Point spread function is wrong")
		errFlag = True

	hDisp = irf.Edisp
	if hDisp.Class_Name() != "TH2D":
		if verbose: print("[Error] Dispersion matrix is wrong")
		errFlag = True

	if normCheck:
		norm = 0
		for i in range(1, hDisp.GetNbinsX()+1):
			norm = 0
			Etr = hDisp.GetXaxis().GetBinCenter(i)
			if len(Erec) == 0:
				for j in range(1, hDisp.GetNbinsY()+1):
					E = hDisp.GetYaxis().GetBinCenter(j)
					dE = hDisp.GetYaxis().GetBinWidth(j)

					D = hDisp.GetBinContent(i, j)
					if D>0:
						norm +=D*dE
			else:
				for j in range(len(Erec)-1):
					E = (Erec[j]+Erec[j+1])/2.
					dE = np.diff(Erec)[j]
					D = hDisp.Interpolate(Etr, E)
					if D>0:
						norm +=D*dE

			if norm == 0:
				continue
			elif abs(norm-1) > 0.05:
				if verbose: print("[Error] Normalization error at Etr = {:.3f} TeV. norm = {:.2f}".format(Etr/1000., norm))
				errFlag = True
	return errFlag
