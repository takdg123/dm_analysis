import numpy as np
import os

from ROOT import TFile, TGraphAsymmErrors, TH2D, TGraph
import uproot

from ..utils import readBinFile
from .. import const
from ..utils import newirf

DM_DIR = os.environ.get('DM')
REF_DIR = DM_DIR+"/RefData/"
DATA_DIR = DM_DIR+"/Data/"


class VEGAS:

	def __init__(self, dwarf, runNum=0, mode = 0, switch=True, azimuth=180, noise = 150, zenith = 20, verbose=False):
		# mode 0: do both
		# mode 1: import effective area
		# mdoe 2: import energy dispersion
		self.exposure=0
		self.dwarf = dwarf
		self.runNum = runNum
		self.package = "VEGAS"

		if int(mode)==0:
			self.EA = self.GetEffectiveArea(verbose=verbose)
			self.Edisp = self.GetEnergyDispersionMatrix(switch=switch, verbose=verbose)
			self.PSF = self.GetPointSpreadFunction(noise = noise, zenith = zenith, verbose=verbose)
		elif int(mode)==1:
			self.EA = self.GetEffectiveArea(verbose=verbose)
		elif int(mode)==2:
			self.Edisp = self.GetEnergyDispersionMatrix(switch=switch, verbose=verbose)
		elif int(mode)==3:
			self.PSF = self.GetPointSpreadFunction(verbose=verbose)
		elif int(mode)==4:
			self.EA, self.Edisp = self.GetNewIRFs(azimuth=azimuth, zenith = zenith, noise=noise, switch=switch)

		print("RES")
	def GetExposure(self):
		
		runNum = 0

		gExp = TGraph()
		gExp.SetName("ExposureTime")
		gExp.SetTitle("Exposure Time ({})".format(self.dwarf))
		gExp.GetXaxis().SetTitle("");
		gExp.GetYaxis().SetTitle("Exposure Time [sec]");

		if self.dwarf == "Segue_1":
			path = REF_DIR+"/Pass5f/segue1_eventList_pass5f_stg5noise_nZnCorr.txt.mce"
		elif self.dwarf == "Draco":
			path = REF_DIR+"/Pass5f/draco_eventList_pass5f_stg5noise_nZnCorr.txt.mce".format(self.dwarf)
		elif self.dwarf == "UMi":
			path = REF_DIR+"/Pass5f/umi_eventList_pass5f_stg5noise_nZnCorr.txt.mce"
		elif self.dwarf == "Bootes_I":
			path = REF_DIR+"/Pass5f/bootes_eventList_pass5f_stg5noise_nZnCorr.txt.mce".format(self.dwarf)
		
		with open(path) as f:
			i=0
			for line in f.readlines()[2:]:
				line = np.asarray(line.split()).astype("float")
				if int(line[0]) == runNum:
					ExpTime = line[1]

		self.exposure = ExpTime
		gExp.SetPoint(1, 0, ExpTime)

		return gExp

	def GetEffectiveArea(self, verbose=False):

		runNum = 0

		gEA = TGraphAsymmErrors();
		gEA.SetTitle("Effective area ({})".format(self.dwarf))
		gEA.GetXaxis().SetTitle("log10 Energy [TeV]")
		gEA.GetYaxis().SetTitle("Effective area [m^2]")

		if self.dwarf == "Segue_1":
			path = REF_DIR+"/Pass5f/segue1_eventList_pass5f_stg5noise_nZnCorr.txt.mce"
		elif self.dwarf == "Draco":
			path = REF_DIR+"/Pass5f/draco_eventList_pass5f_stg5noise_nZnCorr.txt.mce".format(self.dwarf)
		elif self.dwarf == "UMi":
			path = REF_DIR+"/Pass5f/umi_eventList_pass5f_stg5noise_nZnCorr.txt.mce"
		elif self.dwarf == "Bootes_I":
			path = REF_DIR+"/Pass5f/bootes_eventList_pass5f_stg5noise_nZnCorr.txt.mce".format(self.dwarf)
		
		with open(path) as f:
			i=0
			for line in f.readlines()[2:]:
				line = np.asarray(line.split()).astype("float")
				if int(line[0]) == runNum:
					if line[3] < 0:
						line[3] = 0
					gEA.SetPoint(i, np.log10(line[2]/1000.), line[3])
					gEA.SetPointEYlow(i, -1.0*line[4])
					gEA.SetPointEYhigh(i, line[5])
					i+=1
					ExpTime = line[1]

		self.exposure = ExpTime

		if verbose:
			print("[Log; VEGAS] The averaged effective area is imported.")

		return gEA

	def GetEnergyDispersionMatrix(self, switch=True, verbose=False, check=False):
		if self.dwarf == "Segue_1":
			dwarf = "segue_1"
		elif self.dwarf == "Draco":
			dwarf = "draco"
		elif self.dwarf == "UMi":
			dwarf = "ursa_minor"
		elif self.dwarf == "Bootes_I":
			dwarf = "bootes"
		
		File = TFile(DATA_DIR+'VEGAS_EventFile_{}.root'.format(dwarf), 'READ')
		tEv = File.eventTree

		hDisp = []
		LT = []
		runNum_old = -1
		for i in range(tEv.GetEntries()):
			tEv.GetEntry(i)
			if runNum_old != tEv.runNum:
				hDisp.append(self.__readEdisp__(tEv.runNum, switch=switch))
				LT.append(tEv.runLT)
			runNum_old = tEv.runNum

		hDispNorm = self.__renormEdisp__(hDisp, LT, check=check)
		hDispNorm.SetDirectory(0)

		self.exposure = sum(LT)

		if verbose:
			print("[Log; VEGAS] The averaged energy dispersion matrix is imported.")

		return hDispNorm

	def GetPointSpreadFunction(self, noise = 150, zenith = 20, verbose=False):

		InFile = REF_DIR+"/Pass5f/PSF/No{}_Zn{}_na_thHist_psf_overflow.txt".format(int(noise), int(zenith))

		h = TH2D("h", "Energy_TH Hist.(zn={})".format(int(zenith)), 100, -2, 2, 500, 0, 2)

		with open(InFile) as f:
			PSF = []
			for line in f.readlines():
				temp = []
				for val in line.split():
					try:
						temp.append(float(val))
					except:
						continue
				if temp!=[]:
					PSF.append(temp)

		for line, i in zip(PSF, range(1, len(PSF)+1)):
			for val, j in zip(line, range(1, len(line)+1)):
				h.SetBinContent(i, j, val)

		h.SetDirectory(0)

		return h

	def __readEdisp__(self, runNum, switch=True):

		if self.dwarf == "Segue_1":
			path = REF_DIR+"/Pass5f/segue1_bias/"
		elif self.dwarf == "Draco":
			path = REF_DIR+"/Pass5f/draco_bias/".format(self.dwarf)
		elif self.dwarf == "UMi":
			path = REF_DIR+"/Pass5f/umi_bias/"
		elif self.dwarf == "Bootes_I":
			path = REF_DIR+"/Pass5f/bootes_bias/".format(self.dwarf)

		nBinsMC  = 101
		nBinsRec = 101

		eRecBin = readBinFile(path+str(runNum)+".ErecBin.txt")
		eMCBin  = readBinFile(path+str(runNum)+".EtrBin.txt")

		hDisp = TH2D("{}".format(runNum), "{}".format(runNum), nBinsRec-1, eRecBin, nBinsMC-1, eMCBin)

		with open(path+str(runNum)+".mat.txt") as f:
			j = 1
			for line in f.readlines()[2:]:
				i = 1
				for val in line.split():
					if switch:
						hDisp.SetBinContent(j, i, float(val))
					else:
						hDisp.SetBinContent(i, j, float(val))
					i+=1
				j+=1

		hDisp.SetStats(0);
		hDisp.GetXaxis().SetTitle("True Energy [GeV]");
		hDisp.GetYaxis().SetTitle("Reconstructed Energy [GeV]");

		return hDisp

	def __renormEdisp__(self, h, w, check=False):

		eRecBin = np.asarray(h[0].GetXaxis().GetXbins())
		eMCBin  = np.asarray(h[0].GetYaxis().GetXbins())

		hDispNorm = TH2D("h", "Dispersion Matrix ({})".format(self.dwarf), 100, eRecBin, 100, eMCBin)
		hDispNorm.GetXaxis().SetTitle("True Energy [GeV]");
		hDispNorm.GetYaxis().SetTitle("Reconstructed Energy [GeV]");

		numFilled = np.size(w)

		for i in range(1, hDispNorm.GetNbinsX()+1):
			for j in range(1, hDispNorm.GetNbinsY()+1):
				hDispNorm.SetBinContent(i, j, 0.0)

		for k in range(numFilled):
			if h[k] == None:
				continue
			for i in range(1, hDispNorm.GetNbinsX()+1):
				for j in range(1, hDispNorm.GetNbinsY()+1):
					val = h[k].GetBinContent(i, j)*w[k] + hDispNorm.GetBinContent(i, j)
					hDispNorm.SetBinContent(i, j, val)
		
		for i in range(hDispNorm.GetNbinsX()+1):  # for a given true energy
			norm = 0
			Etr = hDispNorm.GetXaxis().GetBinCenter(i)
			dEtr = hDispNorm.GetXaxis().GetBinWidth(i)
			for j in range(hDispNorm.GetNbinsY()+1):
				E = hDispNorm.GetYaxis().GetBinCenter(j)
				ratio = E/Etr
				norm += hDispNorm.GetBinContent(i, j)

			for j in range(hDispNorm.GetNbinsY()+1):
				E = hDispNorm.GetYaxis().GetBinCenter(j)
				dE = hDispNorm.GetYaxis().GetBinWidth(j)
				val = hDispNorm.GetBinContent(i, j)

				ratio = E/Etr

				if norm !=0:
					hDispNorm.SetBinContent(i, j, val/(dE*norm))


		###### Ben's original ######
		#for j in range(hDispNorm.GetNbinsY()+1):  # for a given reconstructed energy
		#	norm = 0
		#	for i in range(hDispNorm.GetNbinsX()+1):
		#		norm += hDispNorm.GetBinContent(i, j)

		#	for i in range(hDispNorm.GetNbinsX()+1):
		#		dE = hDispNorm.GetXaxis().GetBinWidth(i)
		#		val = hDispNorm.GetBinContent(i, j)

		#		if norm !=0:
		#			hDispNorm.SetBinContent(i, j, val/(dE*norm))

		if check:
			for i in range(hDispNorm.GetNbinsX()+1):
				norm = 0
				for j in range(hDispNorm.GetNbinsY()+1):
					dE = hDispNorm.GetYaxis().GetBinWidth(j)
					norm += hDispNorm.GetBinContent(i, j)*dE

				E = hDispNorm.GetXaxis().GetBinCenter(i)
				
				if norm == 0:
					print("Dispersion normalization is 0 at {:.0f} GeV".format(E))
				else:
					print("Dispersion normalization check at {:.0f} GeV: {:.1f}".format(E, norm))

		return hDispNorm


	def GetNewIRFs(self, azimuth=0, zenith=0, noise=6, switch=False):

		self.GetExposure()

		f = uproot.open("./output/testIRF2.root")
		try:
			irf = f["effective_areas/EffectiveArea_Azimuth_{}_Zenith_{}_Noise_{};1".format(azimuth, zenith, noise)]
		except:
			print("[Error] IRF does not exist. List of IRFs:")
			for k in f.keys():
				if "EffectiveArea" in k:
					print(k[30:-2])

		EA = irf._members["pfEffArea_MC"]
		x, y = EA.values()

		EA = TGraph()
		EA.SetName("Effective Area")
		EA.SetTitle("Effective Area")
		EA.GetXaxis().SetTitle("log10( True Energy ) [TeV]")
		EA.GetYaxis().SetTitle("Effective Area [m^2]")

		for i in range(len(x)):
			EA.SetPoint(i, x[i], y[i])

		Edisp = irf._members["pfEnergy_Rec_VS_MC_2D"]

		z, x, y = Edisp.to_numpy()

		Edisp_log = TH2D("Temporary", "Dispersion Matrix", len(x)-1, x, len(y)-1, y)
		Edisp_log.GetXaxis().SetTitle("log10( True Energy ) [TeV]")
		Edisp_log.GetYaxis().SetTitle("log10( Reconstructed Energy ) [TeV]")

		Etr_edge = 10**(np.asarray(Edisp_log.GetXaxis().GetXbins())+3)
		Erec_edge = 10**(np.asarray(Edisp_log.GetYaxis().GetXbins())+3)

		Edisp = TH2D("Dispersion Matrix", "Dispersion Matrix", len(Etr_edge)-1, Etr_edge, len(Erec_edge)-1, Erec_edge)
		Edisp.GetXaxis().SetTitle("True Energy [GeV]")
		Edisp.GetYaxis().SetTitle("Reconstructed Energy [GeV]")

		for i in range(1, Edisp_log.GetNbinsX()+1):
			for j in range(1, Edisp_log.GetNbinsY()+1):
				if switch:
					Edisp_log.SetBinContent(i, j, z[j-1][i-1])
				else:
					Edisp_log.SetBinContent(i, j, z[i-1][j-1])

		for i in range(1, Edisp.GetNbinsX()+1):
			Etr = Edisp.GetXaxis().GetBinCenter(i)
			norm = 0
			for j in range(1, Edisp.GetNbinsY()+1):
				norm += Edisp_log.GetBinContent(i, j)

			if norm != 0:
				for j in range(1, Edisp.GetNbinsY()+1):
					dE = Edisp.GetYaxis().GetBinWidth(j)
					D = Edisp_log.GetBinContent(i, j)
					Edisp.SetBinContent(i, j, D/(norm*dE))

		return EA, Edisp

	def exportIRFs(self, switch=True, azimuth=180, noise = 150, zenith = 20, verbose=False):
		ofile = const.OUTPUT_DIR+"/VEGAS_IRFs_{}.root".format(self.dwarf)
		
		f = TFile.Open(ofile, "RECREATE")
		
		EA = self.GetEffectiveArea(verbose=verbose)
		Edisp = self.GetEnergyDispersionMatrix(switch=switch, verbose=verbose)
		PSF = self.GetPointSpreadFunction(noise = noise, zenith = zenith, verbose=verbose)
		Exp = self.GetExposure()

		f.cd()

		EA.Write("EffectiveArea")

		Edisp.Write("EnergyDispersion")
		
		PSF.Write("PointSpreadFunction")
		
		Exp.Write("ExposureTime")

		f.Close()

		if verbose:
			print("IRFs (dwarf: {}) are saved in {}.".format(self.dwarf, ofile))
	
	@classmethod
	def readIRFs(self, dwarf, path=None):
		if path == None:
			path = const.OUTPUT_DIR+"/VEGAS_IRFs_{}.root".format(dwarf)
		
		irf = newirf()
		
		f = TFile(path, "READ")
		
		irf.EA = f.Get("EffectiveArea")
		irf.Edisp = f.Get("EnergyDispersion")
		irf.Edisp.SetDirectory(0)
		irf.PSF = f.Get("PointSpreadFunction")
		irf.PSF.SetDirectory(0)
		irf.exposure = f.Get("ExposureTime").GetPointY(1)
		
		irf.package = "VEGAS"
		irf.dwarf = dwarf

		return irf
