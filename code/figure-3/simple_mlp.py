import pandas as pd
import sys, os
import numpy as np
import time
import random as rn
import tensorflow as tf

class heatmapResults:
    pass

seed  = sys.argv[2]
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
tf.random.set_seed(np.random.randint(0, 2147483647, dtype='l'))
## removed from tensorflow API... :(
#tf.set_random_seed(np.random.randint(0, 2147483647, dtype='l'))
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import save_model
#from keras.models import model_from_json
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import data_store as ds
import pickle

class record:
    pass

mlp_layers = int(sys.argv[3])
neurons_per_layer = int(sys.argv[4])

description = "\nA script to build an mlp of assemblies\n\n"
example_usage = "usage : \n" + str(sys.argv[0])+ " directory of pickle files time\n\n eg:\n python3 "+str(sys.argv[0])+" ./results_190430_serp/runs/ time\n\n"

def usage_error(string = ""):
    print ("\n\nUSAGE ERROR: \n"+string)
    print (str(sys.argv[0]) + description + example_usage)
    quit()

#def preprocess (ins, outs):
#    # do I need two instances of this?  I would like to recover the data afterwards?
#    scaler_outputs = MinMaxScaler(feature_range=(0, 1))
#    scaler_inputs = MinMaxScaler(feature_range=(0, 1))
#    outs = scaler_outputs.fit_transform(outs)
#    print("\n\ndebug test:")
#    print (len(ins) )
#    ins = scaler_inputs.fit_transform(ins)
#    return np.array(ins), np.array(outs)

def get_hot_pin_ppf(d):
    full = np.block([[d[0], d[1], d[2]],
                     [d[3], d[4], d[5]],
                     [d[6], d[7], d[8]] ])
    #print(np.shape(full))
    hot_pin = np.unravel_index(full.argmax(), full.shape)
    max = np.max(full)
    full[full == 0.0] = np.nan
    mean_no_zero = np.nanmean(full.ravel())
    ppf = max / mean_no_zero
    #print(str(mean_no_zero))
    #print(str(ppf))
    return (list(hot_pin) + [ppf])
    # takes a list of assembly powers and finds
    # the ppf and the position of the hot pin7


def main():

    if 5 != len(sys.argv) or sys.argv[1] == "--help":
        print(len(sys.argv))
        usage_error()
    pickle_dir = sys.argv[1]
    seed  = sys.argv[2]
    if seed == 'time':
        # shuffled time initialisation source:
        # https://github.com/schmouk/PyRandLib
        #t = int( time.time() * 1000.0 )
        #seed = str( ( ((t & 0x0f000000) >> 24) +
        #            ((t & 0x00ff0000) >>  8) +
        #            ((t & 0x0000ff00) <<  8) +
        #            ((t & 0x000000ff) << 24)   )  )
        seed = str(int( (time.time()*1000.0) % 2147483647))
    # import the specified sheet of the data
    np.random.seed(int(seed))
    #rn.seed(np.random.randint(0, 2**32 - 1, dtype='l'))
    rn.seed(np.random.randint(0, 2147483647, dtype='l'))


    try:
        os.path.exists(pickle_dir)
    except:
        print("\n\ directory not found: " + excel_file + "\n")
        exit()
    #list files (just the pickles)
    file_list = [x for x in os.listdir(pickle_dir) if x.endswith(".pickle")]
    #run through the files in order.
    #print (str(file_list))
    inputs = []
    outputs = []
    for data_file in sorted(file_list, key=lambda a: a.split(".")[0]):
        print(pickle_dir+"/"+data_file)
        try:
            pickle_file = open(pickle_dir+"/"+data_file, "rb")
            print("opened file")
            var = pickle.load(pickle_file, encoding='latin1')
            pickle_file.close()
        except:
            print("error opening '"+data_file+"' pickle file")
            exit()
        else:
            inputs.append(var.enrichments)
            #outputs.append( np.concatenate( var.detector_data, axis=0 ) )
            outputs.append(get_hot_pin_ppf(var.detector_data))
    num_inputs = len(inputs[0])
    num_outputs= len(outputs[0])

    #scales...

    X =  np.multiply(np.array(inputs), 1.0/5 )
    Y =  np.multiply(np.array(outputs), np.array([ 1.0/51, 1.0/51, 0.5]))

    #X, Y = preprocess(inputs, outputs)

    print(X[0].shape)
    print(Y[0].shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    #print(X_test)
    #print(Y_test)
    # Create a neural network: (dimensions will be based on heatmap work)
    model = Sequential()
    model.add(Dense(150, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
    for n in range(mlp_layers):
    	model.add(Dense(neurons_per_layer, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_outputs, kernel_initializer='normal'))
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae']) #
    model.compile(loss='mae', optimizer='adam', metrics=['mae']) #
    # train the model
    model.fit(X_train, Y_train, epochs=500, batch_size=50, verbose=1)

    # test
    scores = model.evaluate(X_test, Y_test)
    print("\nperformance on test set:" + str(scores[1]))
    print("scores:"+str(scores))

    predictions = model.predict(X_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    rms_out = [0] * num_outputs
    for y, y_hat in zip(Y_test, predictions):
        for i, (val, val_hat) in enumerate(zip (y, y_hat)):
            rms_out[i] += np.absolute((val-val_hat))
    rms_out = [x / len(Y_test) for x in rms_out]
    rms_error = [np.sqrt(x) for x in rms_out]
    print("rms error per variable:"+str(rms_error))
    model_file = "mlp_model.hdf5"
    model.save(model_file)
    model.save_weights("weights_"+model_file)

    filename = "stats.pickle"
    if os.path.isfile(filename):
        stats = pickle.load( open( filename, "rb" ) )
        stats.errors.append(scores[1])
        stats.per_category_error.append(rms_error)
        pickle.dump(stats, open( filename, "wb" ) )
    else:
        stats = record()
        stats.errors = [scores[1]]
        stats.per_category_error = [rms_error]
        stats.metrics = model.metrics_names
        pickle.dump(stats, open( filename, "wb" ) )
        print (stats.errors)
        print ("Running average over seeds is: " + str(np.mean(stats.errors))+"\n\n")
    print ("metrics: "+str(model.metrics_names))
    print ("scores:  "+str(scores))

    map_y = mlp_layers-1
    map_x = neurons_per_layer-1
    if os.path.isfile("heatmap.pickle"):
        hm = pickle.load( open( "heatmap.pickle", "rb" ) )
        hm.errormap[map_y][map_x] += scores[1]
        hm.freqmap[map_y][map_x] +=1
        pickle.dump(hm, open( "heatmap.pickle", "wb" ) )
    else:
        hm = heatmapResults()
        hm.errormap = np.zeros((20, 150))
        hm.freqmap = np.zeros((20, 150))
        hm.errormap[map_y][map_x] = scores[1]
        hm.freqmap[map_y][map_x] +=1
        pickle.dump(hm, open( "heatmap.pickle", "wb" ) )
        print (hm.errormap)


if __name__ == "__main__":
    main()
