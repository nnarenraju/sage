# LOCAL
from aux_data.O3b import get_o3b_data

raw_args =  ['--num-workers', 4]
raw_args += ['--event-buffer', 30]
raw_args += ['--minimum-segment-duration', 3600]
raw_args += ['--data-dir', '/local/scratch/igr/nnarenraju/O3b_real_noise']
# Get O3b data for H1 and L1
get_o3b_data(raw_args)
