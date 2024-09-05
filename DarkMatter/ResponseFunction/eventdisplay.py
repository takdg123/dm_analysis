import numpy as np
import os
import re

from ROOT import TFile, TH1D, TH2D, TGraph, TDirectoryFile

from ROOT.TObject import kOverwrite, kWriteDelete

import uproot

import ctypes

from ..utils import findIrfFile, findAltFile, getArray, printRunList, getVersion, center_pt

from .. import const

from IPython.display import clear_output

from ..external.v2dl3.IrfInterpolator import IrfInterpolator
#from ..V2DL3.interpolate import IrfInterpolator

from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator

from ..utils import newirf


def getParsFromLog(dwarf, run, ext=False):
	pars = {"ped": None, "livetime": None, "exposure": None, "az": None, "el": None, "deadtime": None, "theta2": None, "alpha": 1/6.}
	if ext:
		filename = const.DATA_DIR+dwarf+"_ext/{}.anasum.log".format(int(run))
	else:
		filename = const.DATA_DIR+dwarf+"/{}.anasum.log".format(int(run))
	
	with open(filename) as f:

		for line in f.readlines():
			pedvar = re.findall("mean pedvars: ([0-9.]+)", line)
			if len(pedvar)>0 and pars["ped"] == None:
				pars["ped"] = float(pedvar[0])

			exposure = re.findall("Exposure ON=([0-9.\+e]+)", line)
			if len(exposure)>0 and pars["exposure"] == None:
				pars["exposure"] = float(exposure[0])

			az = re.findall("mean azimuth: ([0-9.\-]+)", line)
			if len(az)>0 and pars["az"] == None:
				pars["az"] = float(az[0])

			el = re.findall("mean elevation: ([0-9.]+)", line)
			if len(el)>0 and pars["el"] == None:
				if float(el[0])!=0:
					pars["el"] = float(el[0])

			theta2 = re.findall("theta2 cut: ([0-9.]+)", line)
			if len(theta2)>0 and pars["theta2"] == None:
				if float(theta2[0])!=0:
					pars["theta2"] = float(theta2[0])

			alpha = re.findall("alpha=([0-9.]+)", line)
			if len(alpha)>0:
				pars["alpha"] = float(alpha[0])

			if "dead time (fraction of missing events)" in line:
				deadtime = re.findall("([0-9.]+)", line) 
				if len(deadtime)>0 and pars["deadtime"] == None:
					pars["deadtime"] = float(deadtime[0])

	pars["livetime"] = pars["exposure"]*(1-pars["deadtime"]/100.)
	return pars

def th2cut_ext(dwarf=None, ext=False):
	if dwarf is None:
		return 0.012
	else:
		run = printRunList(dwarf, ext=ext)[0]
		pars = getParsFromLog(dwarf, run, ext=ext)
		return pars["theta2"]

def filterRunList(dwarf, az_cut = [-1, 361], zn_cut=[-1, 91], ext=False):
	new_runlist = []
	runlist = printRunList(dwarf, ext=ext)
	if len(runlist) == 0:
		runlist = printRunList(dwarf, log_only=True, ext=ext)

	for run in runlist:
	    pars = ResponseFunction.eventdisplay.getParsFromLog(dwarf, run, ext=ext)
	    az = pars["az"]
	    if az < 0: az +=360
	    if (az_cut[0] != -1) or (az_cut[1] != 361):
	    	if (az >= az_cut[0]) and (az < az_cut[1]):
	    		new_runlist.append(run)

	    zn = pars["zn"]
	    if (zn_cut[0] != -1) or (zn_cut[1] != 91):
	    	if (zn >= zn_cut[0]) and (zn < zn_cut[1]):
	    		new_runlist.append(run)
	
	return new_runlist

class EventDisplay:
	
	def __init__(self, dwarf, runNum, irf_file = None, export_name=None, mode="all", cut="soft", rec=False, export=False, path=None, ext=False, quick=False, from_log=False, verbose=True):

		# mode 0: do all
		# mode 1: import effective area
		# mdoe 2: import energy dispersion
		# mdoe 3: import PSF
		# mode 4: import from a root
		self.package = "EventDisplay"
		self.runNum = runNum
		self.dwarf = dwarf
		if path == None:
			if ext:
				self.path = const.DATA_DIR+"{}_ext/{}.anasum.root".format(self.dwarf, self.runNum)
				self.logpath = const.DATA_DIR+"{}_ext/{}.anasum.log".format(self.dwarf, self.runNum)
			else:
				self.path = const.DATA_DIR+"{}/{}.anasum.root".format(self.dwarf, self.runNum)
				self.logpath = const.DATA_DIR+"{}/{}.anasum.log".format(self.dwarf, self.runNum)
		else:
			self.path = path+"/{}.anasum.root".format(self.runNum)
			self.logpath = path+"/{}.anasum.log".format(self.runNum)

		self.GetRunParameters(ext=ext, from_log=from_log)	
		self.SetInterpolator(filename=self.logpath, irf_file=irf_file, cut=cut, ext=ext)
		
		if export:
			if quick:
				try:
					self.EA, self.Edisp, self.PSF, self.Bias, self.exposure, self.Exp = self.__importIRF__(dwarf, self.runNum, irf_file=irf_file, verbose=verbose, ext=ext)
				except:
					self.EA = self.GetEffectiveArea(verbose=verbose)
					self.Edisp = self.GetEnergyDispersionMatrix(ext=ext, verbose=verbose)
					self.PSF = self.GetPointSpreadFunction(verbose=verbose)
					self.exposure, self.Exp = self.GetExposure(self.dwarf, self.runNum, ext=ext)
					self.__exportIRF__(verbose=verbose, ext=ext, export_name=export_name)
					pass
			else:
				self.EA = self.GetEffectiveArea(verbose=verbose)
				self.Edisp = self.GetEnergyDispersionMatrix(ext=ext, verbose=verbose)
				self.PSF = self.GetPointSpreadFunction(verbose=verbose)
				self.exposure, self.Exp = self.GetExposure(self.dwarf, self.runNum, ext=ext)
				self.__exportIRF__(verbose=verbose, ext=ext, export_name=export_name)
		elif mode=="all":
			self.EA = self.GetEffectiveArea(verbose=verbose)
			self.Edisp = self.GetEnergyDispersionMatrix(ext=ext, verbose=verbose)
			self.PSF = self.GetPointSpreadFunction(verbose=verbose)
			self.exposure, self.Exp = self.GetExposure(self.dwarf, self.runNum, ext=ext)
		elif mode=="EA":
			self.GetRunParameters(ext=ext, verbose=verbose)
			self.EA = self.GetEffectiveArea(verbose=verbose)
		elif mode=="EDisp":
			self.GetRunParameters(ext=ext, verbose=verbose)
			self.Edisp = self.GetEnergyDispersionMatrix(ext=ext, verbose=verbose)
		elif mode=="PSF":
			self.GetRunParameters(ext=ext, verbose=verbose)
			self.PSF = self.GetPointSpreadFunction(verbose=verbose)
		elif mode=="load":
			self.EA, self.Edisp, self.PSF, self.Bias, self.exposure, self.Exp = self.__importIRF__(dwarf, self.runNum, irf_file=path, verbose=verbose, ext=ext)
	

	def GetRunParameters(self, verbose=False, ext=False, from_log=False):

		if from_log:
			pars = getParsFromLog(self.dwarf, self.runNum, ext=ext)
			self.zn = 90-pars["el"]
			self.az = pars["az"] if pars["az"]> 0 else pars["az"]+360
			self.noise = pars["ped"]
			self.offset = 0.5
		else:
			file = uproot.open(self.path)
			
			self.zn = np.abs(90-np.average(file["run_{}/stereo/pointingDataReduced".format(self.runNum)]["TelElevation"].arrays().TelElevation.to_numpy()))
			self.noise = round(file["total_1/stereo/tRunSummary"]["pedvarsOn"].arrays().pedvarsOn[0], 2)
			self.offset = 0.5

			az = file["run_{}/stereo/pointingDataReduced".format(self.runNum)]["TelAzimuth"].arrays().TelAzimuth.to_numpy()
			avAz_rad = np.deg2rad(az)
			avAz = np.rad2deg(np.arctan2(np.sum(np.sin(avAz_rad)),np.sum(np.cos(avAz_rad))))
			avAz = avAz if avAz > 0 else avAz + 360
			self.az = round(np.average(avAz), 2)
			self.exposure, gExp = self.GetExposure(self.dwarf, self.runNum, path = self.path, ext=ext)
			
			file.close()

	def SetInterpolator(self, filename, irf_file=None, cut="soft", ext=False, verbose=False):
		self.version = getVersion(self.runNum)
		eff = findIrfFile(filename)
		if eff:
			self.irfFile = eff
		else:
			print(f"The effArea file is not found for run {self.runNum}. Use an alternative IRF.")	
			self.irfFile = findAltFile(version = self.version, cut = cut, irf_file=irf_file, ext=ext, runNum=self.runNum)
		self.irf_interpolator = IrfInterpolator(self.irfFile, self.az)

	@staticmethod
	def GetExposure(dwarf, runNum, path=None, ext=False):

		gExp = TGraph()
		gExp.SetName("ExposureTime")
		gExp.SetTitle("Exposure Time ({})".format(dwarf))
		gExp.GetXaxis().SetTitle("");
		gExp.GetYaxis().SetTitle("Exposure Time [sec]");

		if path !=None:
			file = uproot.open(path)

			t_start = file["run_{}/stereo/pointingDataReduced".format(runNum)]['Time'].arrays().Time[0]
			t_end = file["run_{}/stereo/pointingDataReduced".format(runNum)]['Time'].arrays().Time[-1]
			on_time = t_end-t_start

			hDeadtime = file["run_{}/stereo/deadTimeHistograms/hScalarDeadTimeFraction_on".format(runNum)].to_numpy()[0]

			deadtime = np.average(hDeadtime[hDeadtime!=0])
			livetime = on_time*(1-deadtime)

			file.close()

			gExp.SetPoint(1, 0, livetime)
			
		else:

			livetime = getParsFromLog(dwarf, runNum, ext=ext)["livetime"]

		Exp = gExp.Clone()

		return livetime, gExp

	def GetEffectiveArea(self, verbose=False):
		if verbose: print("[Log; EventDisplay] Importing an effective area...", end='\r' if verbose!=2 else None)

		self.irf_interpolator.set_irf('eff', use_click=False)
		eff, axis = self.irf_interpolator.interpolate([self.noise, self.zn, self.offset])
		
		gEff = TGraph()
		gEff.SetTitle("Effective area (run={})".format(int(self.runNum)))
		gEff.SetName("EffectiveArea")
		gEff.GetXaxis().SetTitle("log10 Energy [TeV]")
		gEff.GetYaxis().SetTitle("Effective area [m^2]")
		
		for i, (A, E) in enumerate(zip(eff, axis[0])):
			gEff.SetPoint(i, E, A)

		if verbose: print("[Log; EventDisplay] The effectiva area (run={}) is imported.".format(self.runNum))
		return gEff

	def GetEnergyDispersionMatrix(self, ext=False, verbose=False):
		if verbose: print("[Log; EventDisplay] Importing a dispersion matrix...", end='\r' if verbose!=2 else None)
		self.irf_interpolator.set_irf('hEsysMCRelative2D', use_click=False)
		bias, axis = self.irf_interpolator.interpolate([self.noise, self.zn, self.offset])
		
		nBinsX, nBinsY = np.shape(bias.T)

		hBias = TH2D("hBias", "Energy Migration (run={})".format(int(self.runNum)), nBinsX, -2, 4, nBinsY, 0, 3)
		hBias.GetXaxis().SetTitle("log10 True Energy [GeV]");
		hBias.GetYaxis().SetTitle("Ratio (Erec/Etr)");

		eDispBins = 10**np.linspace(1, 7, 61)
		eEDBins = 10**np.arange(1, 7.1, step=0.2)

		hDispNum = TH2D("Count", "Count", len(eEDBins)-1, eEDBins, len(eDispBins)-1, eDispBins)
		hDispNorm = TH2D("DispersionMatrix", "Edisp Matrix (run={})".format(int(self.runNum)), len(eEDBins)-1, eEDBins, len(eDispBins)-1, eDispBins)
		hDispNorm.GetXaxis().SetTitle("True Energy [GeV]");
		hDispNorm.GetYaxis().SetTitle("Reconstructed Energy [GeV]");
		
		for j in range(1, hBias.GetNbinsX()):
		    for k in range(1, hBias.GetNbinsY()):
		        if bias.T[j-1][k-1] > 9:
		            hBias.SetBinContent(j, k, bias.T[j-1][k-1])
		        else:
		            hBias.SetBinContent(j, k, 0)
		
		z, x, y = getArray(hBias)
		f = interp2d(x, y, z, kind='cubic')

		for i in range(1, hDispNorm.GetNbinsX()+1):
			Etr = hDispNorm.GetXaxis().GetBinCenter(i)
			dEtr = hDispNorm.GetXaxis().GetBinWidth(i)
			
			if Etr < 80:
				continue

			norm = 0
			Elog10TeV = np.log10(Etr/1000.0)

			for j in range(1, hDispNorm.GetNbinsY()+1):
				Erec = hDispNorm.GetYaxis().GetBinCenter(j)
				if Erec < 80:
					continue

				ratio_ctr = Erec/Etr
				ratio_Elow = hDispNorm.GetYaxis().GetBinLowEdge(j)/Etr
				ratio_Ehigh = hDispNorm.GetYaxis().GetBinUpEdge(j)/Etr

				ratio = np.linspace(ratio_Elow if ratio_Elow>0.2 else 0.2, ratio_Ehigh if ratio_Ehigh<3 else 3, 10)
				r_ctr = (ratio[1:]+ratio[:-1])/2.
				dratio = np.diff(ratio)
				r_intp = f(Elog10TeV, r_ctr)[:,0]*dratio
				p = sum(r_intp[r_intp>0])
				if p>1:
					hDispNorm.SetBinContent(i, j, p)
					norm+=p
				else:
					hDispNorm.SetBinContent(i, j, 0)

			for j in range(1, hDispNorm.GetNbinsY()+1):
				if norm!=0:
					dE = hDispNorm.GetYaxis().GetBinWidth(j) 
					p = hDispNorm.GetBinContent(i, j)
					hDispNorm.SetBinContent(i, j, p/(norm*dE))
				else:
					hDispNorm.SetBinContent(i, j, 0)

		if verbose:	
			print("[Log; EventDisplay] The dispersion matrix (run={}) is imported.".format(self.runNum))
		
		self.Bias = hBias
		self.Bias.SetDirectory(0)

		hDispNorm.SetDirectory(0)
		return hDispNorm

	def GetPointSpreadFunction(self, verbose=False):
		if verbose: print("[Log; EventDisplay] Importing a point spread function...", end='\r' if verbose!=2 else None)
		self.irf_interpolator.set_irf('hAngularLogDiffEmc_2D', use_click=False)
		psf, axis = self.irf_interpolator.interpolate([self.noise, self.zn, self.offset])
		
		nBinsX, nBinsY = np.shape(psf.T)
		hPSF = TH2D("PSF".format(self.runNum), "PSF (run={})".format(self.runNum), nBinsX, -2, 4, nBinsY, -4, 1)

		for j in range(1, nBinsX+1):
		    for k in range(1, nBinsY+1):
		        if psf.T[j-1][k-1]>25:
		            hPSF.SetBinContent(j, k, psf.T[j-1][k-1])
		if verbose:
			print("[Log; EventDisplay] The point spread function (run={}) is imported.".format(self.runNum))

		hPSF.SetDirectory(0)
		return hPSF
	
	def __importIRF__(self, dwarf, runNum, irf_file=None, verbose=False, ext=False):
		
		if irf_file is None:
			if ext:
				path = const.REF_DIR+'EventDisplay_IRFs_{}_ext.root'.format(dwarf)
			else:
				path = const.REF_DIR+'EventDisplay_IRFs_{}.root'.format(dwarf)
		else:
			path = irf_file

		

		File = TFile(path, "READ")
		IRF = File.Get("irf_{}".format(runNum))
		
		try:
			EA = IRF.Get("EffectiveArea")
		except:
			print(f"[Error] EA is not imported for run {runNum}.")
			raise 

		if EA.Class_Name() != "TGraph":
			EA.SetDirectory(0)
		Edisp = IRF.Get("DispersionMatrix")
		Edisp.SetDirectory(0)
		PSF = IRF.Get("PSF")
		PSF.SetDirectory(0)
		hBias = IRF.Get("hBias")
		hBias.SetDirectory(0)
		#try:
		#	exposure = IRF.Get("ExposureTime").GetPointY(1)
		#except:

		exposure = getParsFromLog(dwarf, runNum, ext=ext)["livetime"]

		gExp = TGraph()
		gExp.SetName("ExposureTime")
		gExp.SetTitle("Exposure Time ({})".format(dwarf))
		gExp.GetXaxis().SetTitle("");
		gExp.GetYaxis().SetTitle("Exposure Time [sec]");
		gExp.SetPoint(1, 0, exposure)

		
		File.Close()

		if verbose:
			print("[Log; EventDisplay] IRFs (run={}) are imported from {}".format(self.runNum, path))

		return EA, Edisp, PSF, hBias, exposure, gExp

	def __exportIRF__(self, export_name=None, ext=False, verbose=False):
		if export_name is None:
			if ext:
				path = const.REF_DIR+'EventDisplay_IRFs_{}_ext.root'.format(self.dwarf)
			else:
				path = const.REF_DIR+'EventDisplay_IRFs_{}.root'.format(self.dwarf)
		else:
			path = export_name

		try:
			File = TFile.Open(path, "UPDATE")
		except:	
			File = TFile.Open(path, "RECREATE")
		try:
			IRF = File.Get("irf_{}".format(self.runNum))
			IRF.Delete("EffectiveArea;1".format(self.runNum))
			IRF.Delete("DispersionMatrix;1".format(self.runNum))
			IRF.Delete("PSF;1".format(self.runNum))
		except:
			IRF = TDirectoryFile("irf_{}".format(self.runNum), "irf files")

		File.cd() 
		EA = self.EA.Clone()
		IRF.Append(EA)
		#EA.SetDirectory(IRF)
		Edisp = self.Edisp.Clone()
		Edisp.SetDirectory(IRF)
		PSF = self.PSF.Clone()
		PSF.SetDirectory(IRF)
		Ebias = self.Bias.Clone()
		Ebias.SetDirectory(IRF)
		Exp = self.Exp.Clone()
		IRF.Append(Exp)

		IRF.Write("irf_{}".format(self.runNum), kWriteDelete)
		File.Close()

		if verbose:
			print("[Log; EventDisplay] IRFs (run={}) are exported to '{}'".format(self.runNum, path))

	@classmethod
	def averagedIRFs(self, dwarf, path=None, irf_file = None, eCut = 80, export=True, verbose=False, check=False, version = "all", runlist=[], norm=False, ext=False, **kwargs):
		if len(runlist) == 0:
			runlist = printRunList(dwarf, path=path, log_only=kwargs.get("log_only", False), ext=ext)
		else:
			export = False


		EA, Edisp, PSF, Bias, exposure, etc = self.__importIRF__(self, dwarf, runlist[0], ext=ext, irf_file=irf_file)

		gEA = TGraph()
		gEA.SetTitle("Effective area ({})".format(dwarf))
		gEA.SetName("EffectiveArea")
		gEA.GetXaxis().SetTitle("log10 True Energy [TeV]");
		gEA.GetYaxis().SetTitle("Effective Area [m^{2}]");
		
		z, x, y = getArray(Edisp, return_edges=True)
		hDisp = TH2D("EnergyDispersion", "Dispersion Matrix ({})".format(dwarf), len(x)-1, x, len(y)-1, y)
		hDisp.GetXaxis().SetTitle("True Energy [GeV]");
		hDisp.GetYaxis().SetTitle("Reconstructed Energy [GeV]");

		hDisp_norm = TH2D("EnergyDispersion_norm", "Dispersion Matrix ({})".format(dwarf), len(x)-1, x, len(y)-1, y)
		hDisp_norm.GetXaxis().SetTitle("True Energy [GeV]");
		hDisp_norm.GetYaxis().SetTitle("Reconstructed Energy [GeV]");

		z, x, y = getArray(Bias, return_edges=True)
		energies = 10**(center_pt(x)+3)
		hBias = TH2D("EnergyBias", "Energy Migration ({})".format(dwarf), len(x)-1, x, len(y)-1, y)
		hBias.GetXaxis().SetTitle("log10 True Energy [GeV]");
		hBias.GetYaxis().SetTitle("Ratio (Erec/Etr)");

		z, x, y = getArray(PSF, return_edges=True)
		hPSF = TH2D("PointSpreadFunction", "PSF ({})".format(dwarf), len(x)-1, x, len(y)-1, y)
		hPSF.GetXaxis().SetTitle("True Energy [GeV]");
		hPSF.GetYaxis().SetTitle("Theta^{2} [deg^{2}]");

		gExp = TGraph()
		gExp.SetName("ExposureTime")
		gExp.SetTitle("Exposure Time ({})".format(dwarf))
		gExp.GetXaxis().SetTitle("");
		gExp.GetYaxis().SetTitle("Exposure Time [sec]");
		
		t_total = 0

		temp_EA = []
		for run in runlist:
			if version == "all":
				pass
			else:
				if version == getVersion(run):
					pass
				else:
					continue
				
			EA, Edisp, PSF, Bias, exposure, etc = self.__importIRF__(self, dwarf, run, ext=ext, irf_file=irf_file)			
			t_total += exposure

			for eng in energies:
				eng = np.log10(eng)-3
				x, y = getArray(EA)
				if eng >= x[-1]:
					y = 0
				else:
					y = EA.Eval(eng)
				if (y >=0):
					temp_EA.append([eng, y*exposure])
				else:
					temp_EA.append([eng, 0])

			for i in range(1, Bias.GetNbinsX()+1):
			    for j in range(1, Bias.GetNbinsY()+1):
			        d = Bias.GetBinContent(i, j)
			        d0 = hBias.GetBinContent(i, j)
			        hBias.SetBinContent(i, j, d0+d*exposure)

			for i in range(1, Edisp.GetNbinsX()+1):
			    for j in range(1, Edisp.GetNbinsY()+1):
			        d = Edisp.GetBinContent(i, j)
			        d0 = hDisp.GetBinContent(i, j)
			        hDisp.SetBinContent(i, j, d0+d*exposure)

			for i in range(1, PSF.GetNbinsX()+1):
			    for j in range(1, PSF.GetNbinsY()+1):
			        dpdr = PSF.GetBinContent(i, j)
			        dpdr0 = hPSF.GetBinContent(i, j)
			        hPSF.SetBinContent(i, j, dpdr0+dpdr*exposure)

		if t_total == 0:
			print("[Error] There is no run corrsponding the entered version.")
			return None

		hDisp.Scale(1/t_total)
		hBias.Scale(1/t_total)
		hPSF.Scale(1/t_total)

		temp_EA = np.asarray(temp_EA)
		for eng, i in zip(energies, range(len(energies))):
			eng = np.log10(eng)-3
			y = sum(temp_EA[:,1][temp_EA[:,0]==eng])/t_total
			gEA.SetPoint(i, eng, y)

		hDisp_norm = hDisp.Clone()
		if norm:
			for i in range(1, hDisp.GetNbinsX()+1):
				norm = 0
				Etr = hDisp.GetXaxis().GetBinCenter(i)
				for j in range(1, hDisp.GetNbinsY()+1):
					E = hDisp.GetYaxis().GetBinCenter(j)
					dE = hDisp.GetYaxis().GetBinWidth(j)

					D = hDisp.GetBinContent(i, j)
					if D>0:
						norm +=D*dE
					else:
						hDisp.SetBinContent(i, j, 0)
				if norm > 0:
					for j in range(1, hDisp.GetNbinsY()+1):
						E = hDisp.GetYaxis().GetBinCenter(j)
						dE = hDisp.GetYaxis().GetBinWidth(j)

						D = hDisp.GetBinContent(i, j)
						hDisp.SetBinContent(i, j, D/norm)

		gExp.SetPoint(1, 0, t_total)

		irf = newirf()
		irf.EA = gEA
		irf.Edisp = hDisp
		irf.Edisp_norm = hDisp_norm
		irf.Bias = hBias
		irf.PSF = hPSF
		irf.exposure = t_total
		irf.package = "EventDisplay"


		if check and norm:
			for i in range(hDisp.GetNbinsX()+1):
				norm = 0
				for j in range(hDisp.GetNbinsY()+1):
					dE = hDisp.GetYaxis().GetBinWidth(j)
					norm += hDisp.GetBinContent(i, j)*dE

				E = hDisp.GetXaxis().GetBinCenter(i)
				
				if norm == 0:
					print("Dispersion normalization is 0 at {:.0f} GeV".format(E))
				else:
					print("Dispersion normalization check at {:.0f} GeV: {:.1f}".format(E, norm))

		if export:
			if ext:
				ofile = const.OUTPUT_DIR+"/EventDisplay_IRFs_{}_ext.root".format(dwarf)
			else:
				ofile = const.OUTPUT_DIR+"/EventDisplay_IRFs_{}.root".format(dwarf)
			try:
				f = TFile.Open(ofile, "UPDATE")
			except:	
				f = TFile.Open(ofile, "RECREATE")
			if version=="all":
				f.Delete("EffectiveArea;1")
				f.Delete("EnergyDispersion;1")
				f.Delete("EnergyDispersion_norm;1")
				f.Delete("EnergyBias;1")
				f.Delete("PointSpreadFunction;1")
				f.Delete("ExposureTime;1")

				gEA.Write("EffectiveArea")
				if norm:
					hDisp.Write("EnergyDispersion_norm")
				else:
					hDisp.Write("EnergyDispersion")
				hBias.Write("EnergyBias")
				hPSF.Write("PointSpreadFunction")
				gExp.Write("ExposureTime")
			else:
				try:
					VD = f.Get("{}".format(version))
					VD.Delete("EffectiveArea;1")
					VD.Delete("EnergyDispersion;1")
					VD.Delete("EnergyDispersion_norm;1")
					VD.Delete("EnergyBias;1")
					VD.Delete("PointSpreadFunction;1")
					VD.Delete("ExposureTime;1")
				except:
					VD = TDirectoryFile("{}".format(version), "irf version")
				
				gEA_S = gEA.Clone()
				VD.Append(gEA_S)
				gExp_S = gExp.Clone()
				VD.Append(gExp_S)
				hDisp_S = hDisp.Clone()
				if norm:
					hDisp_S.SetName("EnergyDispersion_norm")
				else:
					hDisp_S.SetName("EnergyDispersion")
				hDisp_S.SetDirectory(VD)
				hBias_S = hBias.Clone()
				hBias_S.SetDirectory(VD)
				hPSF_S = hPSF.Clone()
				hPSF_S.SetDirectory(VD)
				VD.Write("{}".format(version), kWriteDelete)
			f.Close()

			if verbose:
				print("IRFs (dwarf: {}) are saved in {}.".format(dwarf, ofile))
			
			return irf
		else:
			return irf

	@classmethod
	def readIRFs(self, dwarf, irf_file=None, run=None, norm = False, ext=False, version="all"):
		if irf_file is None:
			if ext:
				path = const.OUTPUT_DIR+"/EventDisplay_IRFs_{}_ext.root".format(dwarf)
			else:
				path = const.OUTPUT_DIR+"/EventDisplay_IRFs_{}.root".format(dwarf)
		else:
			path = irf_file
		
		f = TFile(path, "READ")
		irf = newirf()
		if run == None:
			if version == "all":
				v = f
			elif version in ["v4", "v5", "v6"]:
				try:
					v = f.Get(version)
				except:
					print("[Error] IRFs cannot be loaded.")
			else:
				print("[Error] IRFs cannot be loaded.")

			irf.EA = v.Get("EffectiveArea")
			if norm:
				irf.Edisp = v.Get("EnergyDispersion_norm")
			else:
				irf.Edisp = v.Get("EnergyDispersion")
			irf.Edisp.SetDirectory(0)
			irf.Bias = v.Get("EnergyBias")
			irf.Bias.SetDirectory(0)
			irf.PSF = v.Get("PointSpreadFunction")
			irf.PSF.SetDirectory(0)
			irf.exposure = v.Get("ExposureTime").GetPointY(1)
		else:
			EA, Edisp, PSF, Bias, exposure, etc = self.__importIRF__(self, dwarf, run, ext=ext, irf_file=irf_file)			
			irf.EA = EA
			irf.Edisp = Edisp
			irf.Edisp.SetDirectory(0)
			irf.Bias = Bias
			irf.Bias.SetDirectory(0)
			irf.PSF = PSF
			irf.PSF.SetDirectory(0)
			irf.exposure = exposure

		irf.package = "EventDisplay"
		irf.dwarf = dwarf

		return irf

