#!/usr/bin/env bash
echo ""
echo "This script will take a number of days to run"
echo "reduce the size of the 'neurons' and 'layers' variables in the script:"
echo " './code/figure-5-2/build_heatmap.sh'"
echo "to a smaller number (e.g. 10, 5) to run relatively quickly"
echo ""
read -n 1 -s -r -p "       Press any key to continue                                                             "
echo ""
cd code/figure-5-2/
time ./build_heatmap.sh
mv ./heatmap.svg ../..

