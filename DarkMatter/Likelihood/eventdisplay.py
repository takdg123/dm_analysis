import numpy as np

import matplotlib.pyplot as plt

import re

from ROOT import TFile, TH1D, TH2D, TMath, TCanvas, gPad

import copy

from .. import ResponseFunction

from ..utils import getArray, printRunList, convertToPDF, defineTheta2Cut, center_pt, thetaEdges

from .. import const

from ..ResponseFunction.eventdisplay import th2cut_ext

from .bkgmodel import *

from ..ResponseFunction.eventdisplay import getParsFromLog

def initialize(dwarf = None, biasCut=0.2, effCut = 0.15, export=False, filename=None, ext=False, defaultCut=-0.6):
	if dwarf is None:
		for dwarf in const.ListOfDwarf:
			createEventFile(dwarf, biasCut=biasCut, effCut=effCut, export=export, ext=ext, filename=filename, defaultCut=defaultCut)
	else:
		createEventFile(dwarf, biasCut=biasCut, effCut=effCut, export=export, ext=ext, filename=filename, defaultCut=defaultCut)

def createEventFile(dwarf, path=None, export=False, verbose=True, biasCut=0.2, effCut = 0.15, filename=None, ext=False, defaultCut = -0.6):

	try:
		runlist = printRunList(dwarf, path=path, ext=ext)
		if len(runlist) == 0:
			print("[Warning] Data (dwarf: {}) may not exist. Check your folder.".format(dwarf))
			return
	except:
		print("[Warning] Data (dwarf: {}) may not exist. Check your folder.".format(dwarf))
		return

	events = []

	for run in runlist:
		if path==None:
			if ext:
				path = const.DATA_DIR+dwarf+"_ext"
			else:
				path = const.DATA_DIR+dwarf

		try:
			pars = getParsFromLog(dwarf, run, ext=ext)
			alpha = pars["alpha"]

		except:
			print("[Warning] a log file for run {} is not found.".format(run))
			print("[Warning] alpha is assumed to be 1/6.")
			alpha = 1/6.
			pass

		with open(path+"/{}.anasum.log".format(run)) as f:
			for line in f.readlines()[::-1]:
				Non = re.findall("ON:([0-9.]+)", line)
				if len(Non)>0:
					N_log_on = round(float(Non[0]))
					break

		if alpha == 0.0: alpha = 1/6.
		
		with open(path+"/{}.anasum.log".format(run)) as f:
			for line in f.readlines()[::-1]:
				Noff = re.findall("OFF:([0-9.]+)", line)
				if len(Noff)>0:
					N_log_off = round(float(Noff[0])/alpha)
					break
				
		N_log_tot = N_log_on+N_log_off
		try:
			File=TFile(path+"/{}.anasum.root".format(run), "READ")
			onRegion = File.Get("run_{}".format(run)).stereo.data_on
			offRegion = File.Get("run_{}".format(run)).stereo.data_off
		except:
			print("error", run)
			continue
		irf = ResponseFunction.EventDisplay(dwarf, run, mode="load", ext=ext, verbose=False)
		
		# Effective area cut
		try:
			try:
				h_eff = irf.EA
			except:
				try:
					h_eff = File.Get("run_{}".format(run)).stereo.EffectiveAreas.gMeanEffectiveArea
				except:
					h_eff = File.Get("run_{}".format(run)).stereo.EffectiveAreas.gMeanEffectiveArea_off
				
			energies, eff = getArray(h_eff, return_edges=True)
			
			eThreshold_eff = -5
			for a, e in zip(eff, energies[1:]):
			    if a/max(eff) < effCut:
			        eThreshold_eff = e
			    else:
			    	break
			
		except:
			print("[Warning] run {} does not have gMeanEffectiveArea.".format(run))
			print("[Warning] The eff cut is set to 10^{} TeV ({:.0f} GeV)".format(defaultCut, 10**(defaultCut+3)))
			eThreshold_eff = defaultCut
			pass

		
		# Bias cut
		try:
			try:
				h_bias = irf.Bias
				eThreshold_bias = -5
				z, energies, ratio = getArray(h_bias, return_edges=True)

				for i, val in enumerate(z.T):
				    if val.argmax() == 0:
				        continue
				    elif abs(ratio[val.argmax()]-1) < biasCut:
				        eThreshold_bias = e
				        break

			except:
				h_bias = File.Get("run_{}".format(run)).stereo.EffectiveAreas.gMeanEnergySystematicError
				energies, bias = getArray(h_bias, return_edges=True)

				eThreshold_bias = -5
				for b, e in zip(bias, energies[1:]):
				    if abs(b) > biasCut:
				        eThreshold_bias = e
				    else:
				    	break
		except:
			print("[Warning] run {} does not have gMeanEnergySystematicError.".format(run))
			print("[Warning] The bias cut is set to 10^{} TeV ({:.0f} GeV)".format(defaultCut, 10**(defaultCut+3)))
			eThreshold_bias = defaultCut
			pass

		if int(run) > 63407:
			version = 6
		elif int(run) > 46549:
			version = 5
		else:
			version = 4

		N_root_on = 0
		N_root_off = 0


		for i in range(onRegion.GetEntries()):
			onRegion.GetEntry(i)
			if onRegion.IsGamma == 1:
				events.append([onRegion.ErecS*1000, onRegion.theta2, 1, alpha, version, int(run), onRegion.EChi2S, onRegion.ErecS > 10**eThreshold_bias, onRegion.ErecS > 10**eThreshold_eff, pars["theta2"]])
				N_root_on+=1

		for i in range(offRegion.GetEntries()):
			offRegion.GetEntry(i)
			if offRegion.theta2>=0 and offRegion.IsGamma == 1:
				events.append([offRegion.ErecS*1000, offRegion.theta2, 0, alpha, version, int(run), offRegion.EChi2S, offRegion.ErecS > 10**eThreshold_bias, offRegion.ErecS > 10**eThreshold_eff, pars["theta2"]])
				N_root_off+=1

		N_root_tot = N_root_on+N_root_off
		if N_log_tot != N_root_tot:
			print("[Error] The number of events (run={}): ".format(run))
			print("[Error] log {} ({}/{}), anasum {} ({}/{})".format(N_log_tot, N_log_on, N_log_off, N_root_tot, N_root_on, N_root_off))
		
	
	events=np.asarray(events)

	if export:
		ofile = const.OUTPUT_DIR+'/EventDisplay_Events_{}.txt'.format(dwarf)
		selected_events = copy.copy(events)
		with open(ofile, mode="w") as f:
			f.write("Event[GeV]   Theta[Deg]   isOn   alpha\n")
			selected_events = selected_events[selected_events[:,7]==1]
			selected_events = selected_events[selected_events[:,8]==1]
			
			for evt in selected_events:
				f.write("{:.3f}\t{:10.5f}\t{:10.0f}\t{:10.3f}\n".format(evt[0], np.sqrt(evt[1]), evt[2], evt[3]))
	
	events = np.asarray(events)

	if filename is None:
		if ext:
			ofile = const.DATA_DIR+'EventDisplay_Events_{}_ext.npy'.format(dwarf)
		else:
			ofile = const.DATA_DIR+'EventDisplay_Events_{}.npy'.format(dwarf)
	else:
		ofile = filename

	np.save(ofile, events)

	minE = np.log10(min(events[:,0][(events[:,7]==1)*(events[:,8]==1)]))
	maxE = np.log10(max(events[:,0][(events[:,7]==1)*(events[:,8]==1)]))

	energyEdges = 10**np.arange(minE-0.501, maxE+0.1, step=0.1)
	#energyEdges = 10**np.arange(1, np.log10(max(events[:,0]))+0.1, step=0.1)
	np.save(const.OUTPUT_DIR+"/npy/signalBins_{}".format(dwarf), energyEdges)
	
	print("Events (dwarf: {}) are saved in {}.".format(dwarf, ofile))
			
def exportBackground(dwarf):
	hOn_1d, hOff_1d = readData(dwarf, addTheta=False, bkgModel="ex")
	pdf_hOff_1d = convertToPDF(hOff_1d, norm=True)
	hOn_2d, hOff_2d = readData(dwarf, addTheta=True, bkgModel="ex")
	pdf_hOff_2d = convertToPDF(hOff_2d, norm=True)
	
	ofile = const.OUTPUT_DIR+"/EventDisplay_Background_{}.root".format(dwarf)
	f = TFile(ofile, "RECREATE")
	pdf_hOff_1d.Write("Background_1D")
	pdf_hOff_2d.Write("Background_2D")
	f.Close()

	print("Backgrounds (dwarf: {}) are saved in {}.".format(dwarf, ofile))

def readData(dwarf, events=[], addTheta=False, th2Cut=0, eLowerCut=0, eUpperCut=None, full_output=False, 
	bkgModel=None, rawdata=False, rebinned=False, apply_weight=True, version="all", 
	biasFilter=True, effFilter=True, evt_file=None, ext=False, 
	apply_cuts=True, **kwargs):

	if dwarf =="Test":
		th2Cut = th2cut_ext(dwarf="Segue_1", ext=True)
	elif ext and (th2Cut == 0):
		th2Cut = defineTheta2Cut("EventDisplay", th2cut_ext(dwarf=dwarf, ext=ext))
	else:
		th2Cut = defineTheta2Cut("EventDisplay", th2Cut)

	if eLowerCut == None: eLowerCut = 0
	if eUpperCut == None: eUpperCut = 1e5
	
	if ext:
		eBinEdges = kwargs.get("energyEdges", np.logspace(1, 7, 61))
	else:
		eBinEdges = kwargs.get("energyEdges", const.energyEdges)

	tBinEdges = thetaEdges(th2Cut)
	hOn = TH1D("hOn", "hOn_1D", len(eBinEdges)-1, eBinEdges)
	hOn.SetTitle("Count spectrum (on region)")
	hOn.GetXaxis().SetTitle("Energy [GeV]")
	hOn.GetYaxis().SetTitle("Counts")
	hOn.SetDirectory(0)

	hOff = TH1D("hOff", "hOff_1D", len(eBinEdges)-1, eBinEdges)
	hOff.SetTitle("Count spectrum (off region)")
	hOff.GetXaxis().SetTitle("Energy [GeV]")
	hOff.GetYaxis().SetTitle("Counts")
	hOff.SetDirectory(0)

	if rebinned:
		hOn.RebinX(2)
		hOff.RebinX(2)

	if addTheta:
		hOn_2d = TH2D("hOn","hOn_2D", len(eBinEdges)-1, eBinEdges, len(tBinEdges)-1, tBinEdges)
		hOn_2d.SetTitle("2D count spectrum (on region)")
		hOn_2d.GetXaxis().SetTitle("Energy [GeV]")
		hOn_2d.GetYaxis().SetTitle("Theta2 [deg^2]")
		hOn_2d.GetZaxis().SetTitle("Counts")
		hOn_2d.SetDirectory(0)

		hOff_2d = TH2D("hOff","hOff_2D", len(eBinEdges)-1, eBinEdges, len(tBinEdges)-1, tBinEdges)
		hOff_2d.SetTitle("2D count spectrum (off region)")
		hOff_2d.GetXaxis().SetTitle("Energy [GeV]")
		hOff_2d.GetYaxis().SetTitle("Theta2 [deg^2]")
		hOff_2d.GetZaxis().SetTitle("Counts")
		hOff_2d.SetDirectory(0)

		if rebinned:
			hOn_2d.RebinX(2)
			hOff_2d.RebinX(2)

	if len(events)==0:
		if evt_file is None:
			if ext:
				events = np.load(const.DATA_DIR+'EventDisplay_Events_{}_ext.npy'.format(dwarf), allow_pickle=True)
			else:
				events = np.load(const.DATA_DIR+'EventDisplay_Events_{}.npy'.format(dwarf), allow_pickle=True)
		else:
			if ".npy" not in evt_file:
				evt_file += ".npy"
			events = np.load(evt_file, allow_pickle=True)

		if apply_cuts:
			events = events[events[:,1] < th2Cut]
			events = events[(events[:,0] > eLowerCut)*(events[:,0] < eUpperCut)]
			if biasFilter: events = events[events[:,7]==1]
			if effFilter: events = events[events[:,8]==1]
			if version != "all":
				events = events[events[:,4] == int(version[-1])]

	if rawdata:
		return events

	w = []
	Non = 0
	Noff = 0
	for evt in events:
		energy = evt[0]
		theta2 = evt[1]
		isOn = evt[2]
		alpha = evt[3]
		if theta2 < th2Cut:
			if isOn == 1.:
				Non += 1
				if addTheta:
					hOn_2d.Fill(energy, theta2)	
				else:
					hOn.Fill(energy)
			else:
				Noff += 1
				w.append(alpha)
				if not(apply_weight):
					alpha = 1

				if bkgModel==None:
					if addTheta:
						hOff_2d.Fill(energy, theta2, alpha)
					else:
						hOff.Fill(energy, alpha)

	if len(w)>=1:
		w_avg = np.average(w)
	else:
		w_avg = 1

	if Noff !=0:

		if bkgModel == "alt":
			if addTheta:
				eMin = kwargs.pop("eMin", 500)
				eMax = kwargs.pop("eMax", 3000)
				cnts_off = bkg_alt_2D(events, eBinEdges, tBinEdges, eMin=eMin, eMax=eMax)
			else:
				eMin = kwargs.pop("eMin", 500)
				eMax = kwargs.pop("eMax", 3000)
				cnts_off = bkg_alt_1D(events, eBinEdges, eMin=eMin, eMax=eMax)
		elif bkgModel == "sm":
			if addTheta:
				cnts_off = bkg_sm_2D(events, eBinEdges, tBinEdges)
			else:
				cnts_off = bkg_sm_1D(events, eBinEdges)
		elif bkgModel == "ex":
			if addTheta:
				eMin = kwargs.pop("eMin", 500)
				eMax = kwargs.pop("eMax", 2000)
				cnts_off = bkg_ex_2D(events, eBinEdges, tBinEdges, eMin=eMin, eMax=eMax)
			else:
				eMin = kwargs.pop("eMin", 500)
				eMax = kwargs.pop("eMax", 2000)
				cnts_off = bkg_ex_1D(events, eBinEdges, eMin=eMin, eMax=eMax)
		elif bkgModel == "gaus":
			if addTheta:
				cnts_off = bkg_gaus_2D(events, eBinEdges, tBinEdges, alpha = w_avg)
			else:
				cnts_off = bkg_gaus_1D(events, eBinEdges, alpha = w_avg)

		if bkgModel is not None:
			if addTheta:
				for i in range(1, hOff_2d.GetNbinsX()+1):
					for j in range(1, hOff_2d.GetNbinsY()+1):
						hOff_2d.SetBinContent(i, j, cnts_off[i-1][j-1]*w_avg)
			else:
				for i in range(1, hOff.GetNbinsX()+1):
					if i <= len(cnts_off):
						hOff.SetBinContent(i, cnts_off[i-1]*w_avg)

	
	if addTheta:
	    if full_output:
	        return hOn_2d, hOff_2d, Non, Noff, events, w_avg
	    else:
	        return hOn_2d, hOff_2d
	else:
	    if full_output:
	        return hOn, hOff, Non, Noff, events, w_avg
	    else:
	        return hOn, hOff

def plotData(dwarf, addTheta=False, events=[], eEdges = [], th2Cut=0, eLowerCut = 0, eUpperCut = 1e5, bkgModel=None, verbose=False, version="all", individual=False, biasFilter=True, effFilter=True, **kwargs):
	th2Cut = defineTheta2Cut("EventDisplay", th2Cut)
	if bkgModel == None:
		hOn, hOff = readData(dwarf, addTheta=addTheta, eEdges = eEdges, events=events, th2Cut=th2Cut, bkgModel=bkgModel, eLowerCut=eLowerCut, eUpperCut=eUpperCut, version=version, biasFilter=biasFilter, effFilter=effFilter, **kwargs)
	else:
		hOn, hOff = readData(dwarf, addTheta=addTheta, eEdges = eEdges, events=events, th2Cut=th2Cut, bkgModel=None, eLowerCut=eLowerCut, eUpperCut=eUpperCut, version=version, biasFilter=biasFilter, effFilter=effFilter, **kwargs)
		hOn, hOff_m = readData(dwarf, addTheta=addTheta, eEdges = eEdges, events=events, th2Cut=th2Cut, bkgModel=bkgModel, eLowerCut=eLowerCut, eUpperCut=eUpperCut, version=version, biasFilter=biasFilter, effFilter=effFilter, **kwargs)

	if addTheta:
		if individual:
			plotTwoHist_2D(hOn, hOff, hOff_m)
		else:
			if bkgModel in ["sm", "ex", "alt"]:
				hOff = hOff_m
			hDiff = TH1D("chi", "chi", 20, 0, 10)
			hDiff.SetTitle("#chi ^{2} distribution")
			for i in range(1, hOn.GetNbinsX()+1):
			    for j in range(1, hOn.GetNbinsY()+1):
			        if hOff.GetBinContent(i, j)!=0 :
			            diff = (hOn.GetBinContent(i, j)-hOff.GetBinContent(i, j))**2./hOff.GetBinContent(i, j)
			            if diff > 5:
			            	E = hOn.GetXaxis().GetBinCenter(i)
			            	theta2 = hOn.GetYaxis().GetBinCenter(j)
			            	if verbose: print("(E, theta2) = ({:.0f} GeV, {:.3f} deg), on events: {:.0f}, off events: {:.3f}, chi: {:.1f}".format(E, theta2, hOn.GetBinContent(i, j), hOff.GetBinContent(i, j), diff))
			            hDiff.Fill(diff)
			c = TCanvas("Observation", "Observation", 900,300)
			c.Divide(3,1)
			c.cd(1)
			hOn.Draw("colz")
			gPad.SetLogx()
			c.cd(2)
			hOff.Draw("colz")
			gPad.SetLogx()
			c.cd(3)
			hDiff.Draw()
			hOn.SetDirectory(0)
			hOff.SetDirectory(0)
			hDiff.SetDirectory(0)
			c.Draw()
			return c, hOn, hOff, hDiff
	else:
		xOn, yOn = getArray(hOn)
		xOff, yOff = getArray(hOff)
		if bkgModel is not None:
			xOff_m, yOff_m = getArray(hOff_m)
		else:
			yOff_m = []
		plotTwoHist(xOn, yOn, xOff, yOff, yOff_m)

def plotTwoHist_2D(hOn, hOff, hOff_m, save=False):
	output_s, x, y = getArray(hOn, return_edges=True)
	output_bkg, x, y = getArray(hOff, return_edges=True)
	output_bkg_m, x, y = getArray(hOff_m, return_edges=True)
	x_ctr = center_pt(x)
	y_ctr = center_pt(y)

	for i, (out_s, out_bkg, out_bkg_m) in enumerate(zip(output_s, output_bkg, output_bkg_m)):
		ax = plotTwoHist(x_ctr, out_s, x_ctr, out_bkg, out_bkg_m)
		ax[0].set_title("Theta2={:.4f} deg2".format(y_ctr[i]), fontsize=15)
		if save:
			plt.savefig(const.OUTPUT_DIR+"/figure/Theta={:.0f}deg".format(y_ctr[i]*1000))
		plt.show(block=False)

def plotTwoHist(xOn, yOn, xOff, yOff, y_model=[], label=["On region", "Off region", "Off-region model"]):

	dof = len(yOff[yOff!=0])

	f, ax = plt.subplots(2,1, figsize=(7, 7), gridspec_kw={'height_ratios':[5,1]})
	ax[0].step(xOn, yOn,  label=label[0], where="mid")
	ax[0].step(xOff, yOff, label=label[1], where="mid")
	if len(y_model) == len(xOff):
		ax[0].step(xOff, y_model, label=label[2], where="mid")
	
	ax[0].set_xscale("log")
	ax[0].set_yscale("log")
	ax[0].set_xlim(80, 2e5)
	ax[0].set_ylim(8e-3)
	ax[0].set_ylabel("Counts", fontsize=15)
	ax[0].legend(fontsize=12, loc=1, frameon=False)
	ax[0].grid()

	if len(y_model) == 0: y_model = yOff
	if (max(yOn)<1 or max(y_model)<1):
		ratio = np.sign(yOn[y_model!=0]-y_model[y_model!=0])*abs(y_model[y_model!=0]-yOn[y_model!=0])/y_model[y_model!=0]*100
		ax[1].scatter(xOn[y_model!=0], ratio, marker="+", c="k", label="ratio")
		ax[1].set_ylabel(r"ratio (%)", fontsize=15)
		ax[1].set_ylim(-max(abs(ratio)+3), max(abs(ratio)+3))
	else:
		if len(y_model) == len(xOff):
			chisq = sum((yOn[y_model!=0]-y_model[y_model!=0])**2/y_model[y_model!=0])
			chi = np.sign(yOn[y_model!=0]-y_model[y_model!=0])*(y_model[y_model!=0]-yOn[y_model!=0])**2./y_model[y_model!=0]
			ax[1].errorbar(xOn[y_model!=0], chi, yerr= 1, marker="+", ls="", c="k", label="chisq")
		else:
			chisq = sum((yOn[yOff!=0]-yOff[yOff!=0])**2/yOff[yOff!=0])
			chi = np.sign(yOn[yOff!=0]-yOff[yOff!=0])*(yOff[yOff!=0]-yOn[yOff!=0])**2./yOff[yOff!=0]
			ax[1].errorbar(xOn[yOff!=0], chi, yerr= 1, marker="+", ls="", c="k", label="chisq")
		ax[0].text(0.95, 0.6, r"$\chi^2$ / dof = {:.1f} / {} = {:.2f}".format(chisq, dof, chisq/dof), ha="right", fontsize=12, transform=ax[0].transAxes)
		ax[1].set_ylabel(r"$\chi^2$", fontsize=15)
		ax[1].set_ylim(-max(abs(chi)+1.2), max(abs(chi)+1.2))

	ax[1].set_xscale("log")
	ax[1].set_xlabel("Energy [GeV]", fontsize=15)
	ax[1].set_xlim(80, 2e5)
	ax[1].set_ylim(-6, 6)
	ax[1].axhline(0, color="k", ls="--")
	ax[1].grid()

	return ax