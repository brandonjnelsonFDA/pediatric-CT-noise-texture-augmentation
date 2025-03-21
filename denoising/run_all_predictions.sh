#!/bin/bash

experiment_directory=$1  # Replace with the actual path

# Check if the experiment directory exists
if [ ! -d "$experiment_directory" ]; then
  echo "Error: Experiment directory '$experiment_directory' does not exist."
  exit 1
fi

# Iterate through each folder within the experiment directory
for folder_path in "$experiment_directory"/*/; do
  # Check if it's actually a directory
  if [ -d "$folder_path" ]; then
    # Remove trailing slash for cleaner output and use in run_prediction.sh
    folder_name=$(basename "$folder_path")
    folder_path="${experiment_directory}/${folder_name}"

    echo "Checking folder: $folder_path"

    # Check if the "MayoLDGC" subdirectory exists
    if [ ! -d "$folder_path/MayoLDGC" ]; then
      echo "  Subdirectory 'MayoLDGC' not found. Running run_prediction.sh..."
      # Check if run_prediction.sh exists and is executable
      bash ./run_prediction.sh "$folder_path"
      if [ $? -eq 0 ]; then
        echo "    run_prediction.sh executed successfully."
      else
        echo "    run_prediction.sh failed."
      fi
    else
      echo "  Subdirectory 'MayoLDGC' found. Skipping run_prediction.sh."
    fi
  fi
done

exit 0