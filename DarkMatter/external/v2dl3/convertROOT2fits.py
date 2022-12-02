import os
import glob
from pathlib import Path

REF_DIR = "" # path to the eff file folder

def convertROOT2fits(files, eff = None, **kwargs):
    '''
    This is to convert a root file(s) to a fit file(s).

    Usage:
    from V2DL3.pyV2DL3.eventdisplay.convertROOT2fits import convertROOT2fits
    convertROOT2fits(files)

    Args:
        files (str): a file name or a path to the folder
        eff (optional, pathlib.Path): a path to the eff file
            if None, it will look up a matched eff file from the log.
    '''

    from pyV2DL3.genHDUList import genHDUlist, loadROOTFiles
    from pyV2DL3 import generateObsHduIndex

    if type(files) == str:
        files = [files]
    else:
        files = glob.glob(files + "/*anasum.root")

    full_enclosure = kwargs.pop("full_enclosure", True)
    point_like = kwargs.pop("point_like", True)
    instrument_epoch = kwargs.pop("instrument_epoch", None)
    save_multiplicity = kwargs.pop("save_multiplicity", False)
    filename_to_obsid = kwargs.pop("filename_to_obsid", True)
    evt_filter = kwargs.pop("evt_filter", None)

    if evt_filter is not None:
        evt_filter = Path(evt_filter)

    force_extrapolation = kwargs.get("force_extrapolation", False)
    fuzzy_boundary = kwargs.get("fuzzy_boundary", 0.0)

    if not(full_enclosure) and not(point_like):
        point_like = True
        full_enclosure = False

    irfs_to_store = {"full-enclosure": full_enclosure, "point-like": point_like}

    for file in files:
        if eff is None:
            eff = findIrfFile(file)

        datasource = loadROOTFiles(Path(file), Path(eff), "ED")
        datasource.set_irfs_to_store(irfs_to_store)

        datasource.fill_data(
            evt_filter=evt_filter,
            use_click=False,
            force_extrapolation=force_extrapolation,
            fuzzy_boundary=fuzzy_boundary,
            **kwargs)

        hdulist = genHDUlist(
            datasource,
            save_multiplicity=save_multiplicity,
            instrument_epoch=instrument_epoch,
        )

        fname_base = os.path.basename(file)
        obs_id = int(fname_base.split(".")[0])

        if filename_to_obsid:
            hdulist[1].header["OBS_ID"] = obs_id

        output = kwargs.get("output", file.replace(".root", ".fits"))

        if ".fits" not in output:
            output += ".fits"

        hdulist.writeto(output, overwrite=True)

    datadir = str(Path(file).absolute().parent)
    filelist = glob.glob(f"{datadir}/*anasum.fit*")

    generateObsHduIndex.create_obs_hdu_index_file(filelist, index_file_dir=datadir)

def findIrfFile(filename, return_name=False):
    '''
    This is to find an appropriate eff file for a root file to convert it to a fit file.

    Usage:
    from V2DL3.pyV2DL3.eventdisplay.convertROOT2fits import findIrfFile
    findIrfFile(filename)

    Args:
        filename (str): a file name
        
    Return:
        str: path to the IRF file

    '''

    with open(filename) as file:
        for line in file.readlines():
            effFile = re.findall("effective areas from ([a-zA-Z0-9\-\.\_]+)", line)
            if len(effFile)==1:
                break
    effFile = REF_DIR+effFile[0]

    if return_name:
        return effFile
    elif os.path.exists(effFile):
        return effFile
    else:
        return False

