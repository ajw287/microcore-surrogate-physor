#!/bin/bash
# Copyright 2018 Andrew Whyte & Zhiyao Xing
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

for neurons in {1..150}
do
  for layers in {1..20}
  do
    for i in {1..1}
    do
      echo "Calculating $layers hidden layers of $neurons:"
# 4000 results
#      python3 heatmap_cnn.py ../simple_nn/results_190501_symcore/runs_4/  $i $first $neurons
# 100 pickle files...
      python3 simple_mlp.py ../../data/results_190501_symcore/runs_4  $i $layers $neurons
    done
  done
done
