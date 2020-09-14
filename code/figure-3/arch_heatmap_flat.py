# Copyright 2018 Andrew Whyte & Zhiyao Xing
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

class heatmapResults:
    pass

fig_font = {'fontname':'Liberation Serif'}

#a = np.eye(10,10)
# shows a heatmap of the eroor of different neural architectures.
with open(sys.argv[1],"rb") as f:
    result = pickle.load(f, encoding='latin1')

heatmap = result.errormap
print (result.freqmap)
#a = deepcopy(result.errormap.mean(axis=2))
#heatmap = np.minimum(result.errormap, 1.0)
#heatmap[heatmap >=1.0] = 0.01
#heatmap= heatmap[10:20,20:40]

# for custom slices:
# heatmap = heatmap[:20,:100]
heatmap[heatmap == 0.0] = 0.01
#heatmap[heatmap == float('nan')] = 0.25
heatmap[heatmap > 0.2] = 0.2
heatmap = np.ma.masked_where(np.isnan(heatmap), heatmap)
heatmap = heatmap *100

print ("data in file:")
#print (result.errormap.mean(axis=2))
print("data in heatmap:")
#print(heatmap.mean(axis=2))
fig, ax = plt.subplots()
print("heatmap:")
#np.set_printoptions(threshold='nan')
for line in heatmap:
    print(list(line))
#print(heatmap)

#mask = heatmap.isnan()
map = ax.imshow(heatmap, interpolation='nearest', origin='lower', cmap=cm.viridis, norm=colors.LogNorm(vmin=heatmap.min(), vmax=heatmap.max()))
#plt.colorbar(heatmap,orientation = 'horizontal')
cbar = fig.colorbar(map, orientation='horizontal')
#cbar.ax.set_yticklabels(['0.0','0.025','0.05','0.075','>0.01'])
cbar.set_label('Error scale / %', rotation=0, **fig_font)
ax.set_title('Prediction error with different MLP topologies', **fig_font)
plt.xlabel('Neurons per hidden layer', **fig_font)
plt.ylabel('Hidden layers', **fig_font)

CONST = 1
#x = np.arange(-9, 150, 10)
#y = np.arange(0, 20, 1)
#labels_x = [1,20,40,60,80,100,120,140]#[i+CONST for i in x]
#labels_y = [20,10,1]#[i+CONST for i in y]
## set custom tick labels
#ax.set_xticklabels(labels_x)
#ax.set_yticklabels(labels_y)

# for custom slices:
#x = range(24, 100, 25)
x = range(24, 150, 25)
y = range(4, 20, 5)
x = [0]+[i for i in x]
y = [0]+[i for i in y]
labels_x = [i+CONST for i in x]
labels_y = [i+CONST for i in y]

# set custom tick labels
plt.xticks(x, labels_x)
plt.yticks(y, labels_y)

plt.show()
fig.savefig('heatmap.svg')
