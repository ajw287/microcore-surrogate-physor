#!/usr/bin/env bash
echo "Optimises in pygmo using a pretrained CNN "
read -n 1 -s -r -p "       Press any key to continue                                                             "
echo ""
cd code/cnn-pygmo/
python3 pygmo_micro.py test

