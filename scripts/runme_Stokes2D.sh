#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

module purge > /dev/null 2>&1
module load julia
module load cuda/11.0

julia_=$(which julia)

RESOL=( 31 63 127 255 511 1023 2047 4095 8191 )

USE_GPU=true

DO_VIZ=false

DO_SAVE=true

if [ "$DO_SAVE" = "true" ]; then
    FILE=../output/out_Stokes2D.txt
    if [ -f "$FILE" ]; then
        echo "Systematic results (file $FILE) already exists. Remove to continue."
        exit 0
    else 
        echo "Launching systematics (saving results to $FILE)."
    fi
fi

for i in "${RESOL[@]}"
do
    USE_GPU=$USE_GPU DO_VIZ=$DO_VIZ DO_SAVE=$DO_SAVE NX=$i NY=$i $julia_ --project -O3 --check-bounds=no Stokes2D.jl
done