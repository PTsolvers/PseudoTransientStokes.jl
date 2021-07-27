#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

mpirun_=$(which mpirun)

RESOL=( 127 )

USE_GPU=true

DO_VIZ=true

DO_SAVE=false

if [ "$DO_SAVE" = "true" ]; then

    FILE=../../output/out_Stokes3D.txt
    
    if [ -f "$FILE" ]; then
        echo "Systematic results (file $FILE) already exists. Remove to continue."
        exit 0
    else
        echo "Launching systematics (saving results to $FILE)."
    fi
fi

for i in "${RESOL[@]}"
do
    $mpirun_ -np 8 -rf gpu_rankfile_node40 ./submit_julia_viz.sh $i $USE_GPU $DO_VIZ $DO_SAVE
done
