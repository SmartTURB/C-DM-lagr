#!/bin/bash
# Divide trajectories into non-overlapping 60-day segments
python create_dataset-60days.py

# Remove segments with spurious point(s) of high velocity or acceleration
python outlier_pruner-vit3-dvit4.py

# Save normalized velocities and segments' initial positions
python preprocess-dataset-diffusion.py
python output_initial_positions.py

# Clean up the intermediate output and rename the final dataset
rm dataset-60days.h5 dataset-60days-vit3-dvit4.h5
mv dataset-60days-vit3-dvit4-diffusion.h5 gdp1h_60d-diffusion.h5
