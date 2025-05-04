#!/bin/bash

# This script runs the SimuXForm experiment with various parameter combinations.
# Make sure the virtual environment is activated before running this script:
# source .venv/bin/activate

echo "Starting experiment sweep..."

# Define parameter ranges
NORMS=("preln" "postln")
RANDOM_VS=(0)
RANDOM_KQS=(0)

# Get total number of experiments for progress tracking
TOTAL_EXPERIMENTS=$((${#NORMS[@]} * ${#RANDOM_VS[@]} * ${#RANDOM_KQS[@]}))
CURRENT_EXPERIMENT=0

# Loop through all combinations
for norm_val in "${NORMS[@]}"; do
  for v_val in "${RANDOM_VS[@]}"; do
    for kq_val in "${RANDOM_KQS[@]}"; do
      CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
      echo ""
      echo "------------------------------------------------------------"
      echo "Running Experiment $CURRENT_EXPERIMENT / $TOTAL_EXPERIMENTS"
      echo "Config: --norm $norm_val --randomV $v_val --randomKQ $kq_val"
      echo "------------------------------------------------------------"

      # Construct the command
      # Add any other fixed flags here if needed (e.g., --dmodel, --ntokens)
      COMMAND="python runner.py --norm $norm_val --randomV $v_val --randomKQ $kq_val"

      # Print the command to be executed
      echo "Executing: $COMMAND"
      echo ""

      # Execute the command
      $COMMAND

      # Check exit status (optional)
      if [ $? -ne 0 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Experiment $CURRENT_EXPERIMENT failed!" 
        echo "Config: --norm $norm_val --randomV $v_val --randomKQ $kq_val"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # Decide whether to exit or continue
        # exit 1 # Uncomment to stop the script on failure
      else
         echo "Experiment $CURRENT_EXPERIMENT completed successfully."
      fi

    done
  done
done

echo ""
echo "------------------------------------------------------------"
echo "Experiment sweep finished!"
echo "------------------------------------------------------------" 