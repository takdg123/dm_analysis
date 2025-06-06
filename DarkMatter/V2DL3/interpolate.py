import numpy as np
from .util import extract_irf, duplicate_dimensions, WrongIrf
from scipy.interpolate import RegularGridInterpolator

class IrfInterpolator:
    def __init__(self, filename, azimuth, **kwargs):
        self.implemented_irf_names_1d = ['eff', 'Rec_eff', 'effNoTh2', 'Rec_effNoTh2'] #'gEffAreaNoTh2MC', 'gEffAreaNoTh2Rec']
        self.implemented_irf_names_2d = ['hEsysMCRelative2D', 'hEsysMCRelative2DNoDirectionCut',
                                         'hAngularLogDiff_2D', 'hAngularLogDiffEmc_2D']
        self.irf_name = ""
        self.azimuth = azimuth
        

        import os.path
        if os.path.isfile(filename):
            self.filename = filename
        else:
            raise FileNotFoundError

    def set_irf(self, irf_name, **kwargs):

        if irf_name in self.implemented_irf_names_1d or irf_name in self.implemented_irf_names_2d:
            self.irf_name = irf_name
            self.__load_irf(**kwargs)
        else:
            print("The irf you entered: {}  is either wrong or not implemented.".format(irf_name))
            raise WrongIrf

    def __load_irf(self, **kwargs):
        irf_data, irf_axes = extract_irf(self.filename, self.irf_name, azimuth=self.azimuth, 
                                         single_index=True, return_irf_axes=True, **kwargs)
        # This is an important technical step: the regular grid interpolator does not accept 
        # interpolating on a dimension with size = 1.
        # Make sure that there are no size 1 dimensions. Do the same with the axes:
        irf_data = duplicate_dimensions(irf_data)
        # Also the coordinates of the axes need to be in increasing order.
        for i, axis in enumerate(irf_axes):
            if len(axis) == 1:
                irf_axes[i] = np.concatenate((axis.flatten(), axis.flatten() + 0.01), axis=None)
        self.irf_data = irf_data
        self.irf_axes = irf_axes
        self.interpolator = RegularGridInterpolator(self.irf_axes, self.irf_data)

    def interpolate(self, coordinate, **kwargs):
        verbose = kwargs.get("verbose", False)
        if verbose: print("Interpolating coordinates: ", coordinate)
        # The interpolation is slightly different for 1D or 2D IRFs. We do both cases separated:
        if self.irf_name in self.implemented_irf_names_2d:
            # In this case, the interpolator needs to interpolate over 2 dimensions:
            xx, yy = np.meshgrid(self.irf_axes[0], self.irf_axes[1])
            interpolated_irf = self.interpolator((xx, yy, coordinate[0], coordinate[1], coordinate[2]))
            return interpolated_irf, [self.irf_axes[0], self.irf_axes[1]]
        elif self.irf_name in self.implemented_irf_names_1d:
            # In this case, the interpolator needs to interpolate only over 1 dimension (true energy):
            interpolated_irf = self.interpolator((self.irf_axes[0], coordinate[0], coordinate[1],
                                                  coordinate[2]))
            return interpolated_irf, [self.irf_axes[0]]
        else:
            print("The interpolation of the irf you entered: {}  is either wrong or not implemented.".format(self.irf_name))
            raise WrongIrf
