#!/bin/bash
#
# Launch all 6 (run x detector) PSD-generation jobs concurrently.
#
# Each job is an independent single-detector process:
#   - CPU-only (the GPUs are not used)
#   - ~1.3 GB peak RAM (streaming PSD writer + chunked median)
#   - writes detector-specific filenames, so they never collide
#
# Each job produces, for its detector:
#   - fiducial PSD   -> runs/<run>/run_export/fiducial_psds/fiducial_<det>_psd.bin
#   - recolour bank  -> data_release/<run>_dataset/data_dir/recolour_psds/raw_<det>_psds.bin
#   - segment PSDs   -> data_release/<run>_dataset/data_dir/segment_psds/data_<det>_psds.bin
#
# We `cd` into each run dir so the relative export_dir (./run_export) resolves
# per-run for the fiducial PSDs.

set -u
SAGE=/home/nnarenraju/Research/sage

echo "Launching 6 PSD jobs (2 runs x 3 detectors)..."
for run in o3a o3b; do
  for det in H1 L1 V1; do
    log="$SAGE/runs/$run/psd_gen_${run}_${det}.log"
    (
      cd "$SAGE/runs/$run" && \
      nohup python3 -c "from dataset import make_psds_single; make_psds_single('$det')" \
        > "$log" 2>&1 &
      echo "  launched $run/$det -> pid $! (log: $log)"
    )
  done
done

echo
echo "Watch progress with:  tail -f $SAGE/runs/o3a/psd_gen_o3a_H1.log"
echo "Check all jobs with:   ps -u \$USER -o pid,rss,etime,cmd | grep make_psds_single | grep -v grep"
