#!/bin/bash
#
# Launch all 6 (run x detector) ASD-generation jobs concurrently.
#
# Each job is an independent single-detector process:
#   - CPU-only (the GPUs are not used)
#   - ~1.3 GB peak RAM (streaming ASD writer + chunked median)
#   - writes detector-specific filenames, so they never collide
#
# Each job produces, for its detector:
#   - fiducial ASD   -> runs/<run>/run_export/fiducial_psds/fiducial_<det>_psd.bin
#   - recolour bank  -> data_release/<run>_dataset/data_dir/recolour_psds/raw_<det>_psds.bin
#   - segment ASDs   -> data_release/<run>_dataset/data_dir/segment_psds/data_<det>_psds.bin
#
# We `cd` into each run dir so the relative export_dir (./run_export) resolves
# per-run for the fiducial ASDs.

set -u
SAGE=/home/nnarenraju/Research/sage

echo "Launching 6 ASD jobs (2 runs x 3 detectors)..."
for run in o3a o3b; do
  for det in H1 L1 V1; do
    log="$SAGE/runs/$run/asd_gen_${run}_${det}.log"
    (
      cd "$SAGE/runs/$run" && \
      nohup python3 -c "from dataset import make_asds_single; make_asds_single('$det')" \
        > "$log" 2>&1 &
      echo "  launched $run/$det -> pid $! (log: $log)"
    )
  done
done

echo
echo "Watch progress with:  tail -f $SAGE/runs/o3a/asd_gen_o3a_H1.log"
echo "Check all jobs with:   ps -u \$USER -o pid,rss,etime,cmd | grep make_asds_single | grep -v grep"
