import numpy as np
import copy
import os.path
import sys
import os
import pickle
import socket
import time
import random as rn
import tensorflow as tf
import math

seed  = 'time'
# Max for PYTHONHASHSEED 4294967295
# Max for numpy 2**32
if seed == 'time':
    # shuffled time initialisation source:
    # https://github.com/schmouk/PyRandLib
    #t = int( time.time() * 1000.0 )
    #seed = str( ( ((t & 0x0f000000) >> 24) +
    #            ((t & 0x00ff0000) >>  8) +
    #            ((t & 0x0000ff00) <<  8) +
    #            ((t & 0x000000ff) << 24)   )  )
    seed = str(int( (time.time()*1000.0) % 2147483647))
#os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(int(seed))
#rn.seed(np.random.randint(0, 2**32 - 1, dtype='l'))
rn.seed(np.random.randint(0, 2147483647, dtype='l'))
#
from keras import backend as K
# these changes are bitrot from older versions of tensorflow
#tf.set_random_seed(np.random.randint(0, 2147483647, dtype='l'))
tf.random.set_seed(np.random.randint(0, 2147483647, dtype='l'))
# these changes are bitrot from older versions of tensorflow
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
#os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
from keras.optimizers import Adam

from parse_detectors import parse_detectors




class problem_serpent_:
    pass

class chromosome:
    pass

class problem_susmicro:
    model_file = "cnn_model.hdf5"
    model = load_model(model_file)

    #originally part of simple_nn.py in conv_nn file (keep in sync!)
    def get_input_image(self, nine_vars, filename='input_img.png', save=False):
        w, h = 55, 55
        control_rods = [[[ 6,  3], [ 9,  3], [12,  3],
                        [ 4,  4], [14,  4],
                        [ 3,  6], [ 6,  6], [ 9,  6], [12,  6], [15,  6],
                        [ 3,  9], [ 6,  9], [12,  9], [15,  9],
                        [ 3, 12], [ 6, 12], [ 9, 12], [12, 12], [15, 12],
                        [ 4, 14], [14, 14], [ 6, 15],
                        [ 9, 15], [12, 15]]
                       ]
        control_rods.append([ [v[0]+18,v[1]+ 0] for v in control_rods[0] ])
        control_rods.append([ [v[0]+36,v[1]+ 0] for v in control_rods[0] ])
        control_rods.append([ [v[0]+ 0,v[1]+18] for v in control_rods[0] ])
        control_rods.append([ [v[0]+18,v[1]+18] for v in control_rods[0] ])
        control_rods.append([ [v[0]+36,v[1]+18] for v in control_rods[0] ])
        control_rods.append([ [v[0]+ 0,v[1]+36] for v in control_rods[0] ])
        control_rods.append([ [v[0]+18,v[1]+36] for v in control_rods[0] ])
        control_rods.append([ [v[0]+36,v[1]+36] for v in control_rods[0] ])

        data = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            for y in range(h):
                assembly_number = int(x/18) + (int(y/18) *3)
                if x%18 == 0 or y%18==0 or assembly_number > 9:  # water gap
                    data[y, x] = [0, 0, 255]
                else:
                    if [x,y] in control_rods[assembly_number] or 0== x or 0==y or w-1 == x or h-1 == y:
                        data[y, x] = [0, 0, 255]
                    else:
                        data[y, x] = [nine_vars[assembly_number]*50, 0, 0]
        if save:
            img = Image.fromarray(data, 'RGB')
            new_p.save(filename, "PNG")
        return data

    #originally part of simple_nn.py in conv_nn file (keep in sync!)
    def get_output_image(self, d, filename='pin_power.png', save=False):
        pass

    #originally part of simple_nn.py in conv_nn file (keep in sync!)
    def get_cnn_output_image(self, d, filename='pin_power.png', save=False):
        w, h = 51, 51
        control_rods = [[[ 6,  3], [ 9,  3], [12,  3],
                        [ 4,  4], [14,  4],
                        [ 3,  6], [ 6,  6], [ 9,  6], [12,  6], [15,  6],
                        [ 3,  9], [ 6,  9], [12,  9], [15,  9],
                        [ 3, 12], [ 6, 12], [ 9, 12], [12, 12], [15, 12],
                        [ 4, 14], [14, 14], [ 6, 15],
                        [ 9, 15], [12, 15]]
                       ]
        # note the difference here as there isn't a water gap as an input
        control_rods.append([ [v[0]+17,v[1]+ 0] for v in control_rods[0] ])
        control_rods.append([ [v[0]+34,v[1]+ 0] for v in control_rods[0] ])
        control_rods.append([ [v[0]+ 0,v[1]+17] for v in control_rods[0] ])
        control_rods.append([ [v[0]+17,v[1]+17] for v in control_rods[0] ])
        control_rods.append([ [v[0]+34,v[1]+17] for v in control_rods[0] ])
        control_rods.append([ [v[0]+ 0,v[1]+34] for v in control_rods[0] ])
        control_rods.append([ [v[0]+17,v[1]+34] for v in control_rods[0] ])
        control_rods.append([ [v[0]+34,v[1]+34] for v in control_rods[0] ])
        #full = np.block([[d[0], d[1], d[2]],
        #                 [d[3], d[4], d[5]],
        #                 [d[6], d[7], d[8]] ])
        full = d

        pin_pows = full
        #ds=data_store.data_store()
        with open( "scale.pickle", 'rb') as f:
            ds = pickle.load(f)
        old_max = ds.scale_data[2]
        old_min = ds.scale_data[3]
        offset = ds.scale_data[4]
        old_range = old_max-old_min
        scaled_set = ( ( (pin_pows -old_min) * 1)/ old_range) + offset
        Y = scaled_set
        scaled = ( ( (pin_pows -old_min) * 1)/ old_range)+0.2

        max = 0.0
        average = 0.0
        n = 0
        for x in range(w):
            for y in range(h):
                assembly_number = math.floor(x/17) + (math.floor(y/17) *3)
                if [x,y] in control_rods[assembly_number] or 0== x or 0==y or w-1 == x or h-1 == y:
                    #print(np.shape(full))
                    #print(str(x)+", "+str(y))
                    #print(full[x,y])
                    full[y, x] = np.nan
                else:
                    if full[y,x] > max:
                        max = full[y,x]
                    n +=1
                    average += full[y,x]
        average /= n
#        print(" ave: " +str(average) + " max " + str(max) +"  ppf : " +str(max/average))
        #full[full == 0.0] = np.nan
        max = np.nanmax(full) # largest value
        min = np.nanmin(full)
        mean = np.nanmean(full)
        ppf = max/mean
        #full[full == np.nan] = 0.0
        full = np.nan_to_num(full)
        pos = np.unravel_index(full.argmax(), full.shape)
        scaled[scaled == np.nan] = 0
        #img.convert('RGB')
        if save:
            img = Image.fromarray(scaled*255)
#            print(full)
            img.show()
            new_p = img.convert("L")
            new_p.save(filename, "PNG")
        return full, pos, ppf

    def aw_round(self,x, base=0.2):
        return base * round(x/base)

    def __deepcopy__(self, memo):
        #print ('performing __deepcopy__(%s)' % str(type(memo)))
        #clone_model(model, input_tensors = NULL)
        return problem_susmicro()
        #return VariableWithoutRE(self.name, self.regexTarget, self.type)

    def __init__(self, saves_dir="serpent"):
        self.dim = 9
        self.nobj = 2 # average enrichment , lppf , k infinity error
        self.run_number = 0
        print("completed init function")

    def get_normed_hot_pin_ppf(self, d):
        inputs = []
#        print (str(d))
        for val in d:
            val = self.aw_round(val) #* 1.0/5
            inputs.append(val)
#        print(str(inputs))
        #X =  np.multiply(np.array(inputs), 1.0/5 )
        X = self.get_input_image(inputs)
        X =  np.multiply(np.array(X), 1.0/5 )
#        print(np.shape(X))
        #X = np.expand_dims(X, 0)
        #inputs2 = np.reshape(np.array(inputs), (9,))
        #scale the inputs to the neural network...
        X = X.reshape(1,55,55,3)#.astype('float')
        pin_vals = self.model.predict(X)
#        print(np.shape(pin_vals))
        pin_vals = pin_vals.reshape(51,51)
#        print(np.shape(pin_vals))
        #scaled = self.model.predict(X)
        scaled, pos, ppf = self.get_cnn_output_image(pin_vals)
        radial_dist = np.sqrt(pos[0]* pos[0] + pos[1]*pos[1])
#        print(str(inputs))
#        print(str(scaled))
#        return [float(scaled[0][0]), float(scaled[0][1]), float(scaled[0][2])]
        #return [float(51-pos[0]), float(51-pos[1]), float(ppf)]
        return [ 51-pos[0],  51-pos[1], float(ppf)]

    def start_fitness(self, x, run=0):
        self.run_number = run
        print("not implemented!")
        hash = hash(tuble(x))
        return hash

    def dist(self, a,b):
        return np.sqrt(np.sum((a-b)**2, axis=0))

    def fitness(self, x, run=0):
        self.run_number = run
        print("getting fitness calculation")
        scaled_x = [float(v)/10.0 if v%2==0 else float(v+1)/10.0 for v in x]
        results = self.get_normed_hot_pin_ppf(scaled_x)
        radial_dist = self.dist(results[0], results[1])
        return [radial_dist, results[2]]

    def get_nobj(self):
        return self.nobj

    def get_bounds(self):
        # chromasonal bounda are assembly enrichments 0.8 -5.0%
        #return [0.8] * 9, [5.0]*9
        return [7] * 9, [50]*9

    def get_nix(self):
        return 9

    def get_name(self):
        return "Fuel design with pyGMO 2.6+ and keras model"

    def get_extra_info(self):
        string = "Relies on keras model: "+str(model_file)+"\tDimensions: " + str(self.dim)
        return string
