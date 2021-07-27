#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

mpirun_=$(which mpirun)

RESOL=( 31 63 127 255 511 )

USE_GPU=true

DO_VIZ=false

DO_SAVE=true

DO_SAVE_VIZ=false

if [ "$DO_SAVE" = "true" ]; then
    
    FILE=../../output/out_Stokes3D_ve3.txt

    if [ -f "$FILE" ]; then
        echo "Systematic results (file $FILE) already exists. Remove to continue."
        exit 0
    else
        echo "Launching systematics (saving results to $FILE)."
    fi
fi

for i in "${RESOL[@]}"
do
    $mpirun_ -np 8 -rf gpu_rankfile_node40 ./submit_julia.sh $i $USE_GPU $DO_VIZ $DO_SAVE $DO_SAVE_VIZ
done
