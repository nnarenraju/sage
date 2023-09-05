# Modules
import os
import h5py
import glob
import numpy as np

from pycbc.types import FrequencySeries


def load_psds(data_loc, data_cfg):
    """ PSD Handling (used in whitening) """
    # Store the PSD files here in RAM. This reduces the overhead when whitening.
    # Read all psds in the data_dir and store then as FrequencySeries
    psds = {}
    psd_files = glob.glob(os.path.join(data_loc, "psds/*"))
    for psd_file in psd_files:
        with h5py.File(psd_file, 'r') as fp:
            data = np.array(fp['data'])
            delta_f = fp.attrs['delta_f']
            name = fp.attrs['name']
            
        psd_data = FrequencySeries(data, delta_f=delta_f)
        # Store PSD data into lookup dict
        psds[name] = psd_data

    if data_cfg.dataset == 1:
        psds_data = [psds['aLIGOZeroDetHighPower']]*2
    else:
        psds_data = [psds['median_det1'], psds['median_det2']]

    return psds_data
