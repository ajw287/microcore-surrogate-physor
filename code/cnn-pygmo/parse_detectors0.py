import copy
import numpy as np
import sys, os
import re
import data_store as ds
import pickle

num_detector = 9

def parse_detectors(detector_filename):
    detector_data  = [copy.deepcopy(np.empty([17,17])) for x in range(num_detector)]
    detector_error = [copy.deepcopy(np.empty([17,17])) for x in range(num_detector)]
    try:
        file = open(detector_filename, 'r')
    except:
        print("detector file error: (check path and that it exists! file:"+detector_filename+")")
        quit()
    else:
        filedata = file.readlines()
        for i in range(1, num_detector+1):
            temp_data = copy.deepcopy(filedata)
            parsing = False
            for line in temp_data:
                if parsing == False:
                    if "DETa{:03d} = [".format(int(i)) in line:
                        print("found " + "DETa{:03d} = [".format(int(i)))
                        parsing = True
                else:
                    if "];" in line:
                        parsing = False
                    else:
                        data  = [float(x) for x in re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", line)]
                        print("y:"+str(int(data[-4])-1) + " x:"+str(int(data[-3])-1) +" i:"+str(i-1))
                        (detector_data[i-1])[int(data[-4])-1][int(data[-3])-1]  = data[-2]
                        (detector_error[i-1])[int(data[-4])-1][int(data[-3])-1] = data[-1]
    return detector_data

def main():
    if len(sys.argv) == 2:
        detector_filename = sys.argv[1]+"_det0.m"
    else:
        detector_filename = "super_cell_test_det0.m"
    detector_data = parse_detectors(detector_filename)
    try:
        print (sys.argv[1]+".pickle")
        pickle_file = open(sys.argv[1]+".pickle", "rb")
        print("opened file")
        var = pickle.load(pickle_file)
        print(var.enrichments)
    except:
        print("error opening '' pickle file")
        exit()
    else:
        var.detector_data = detector_data
        pickle_file.close()
        try:
            f = open(sys.argv[1]+".pickle", "wb+")  # you should be creating state.pickle here...
        except:
            print ("Error opening output file")
            exit()
        else:
            f.write(pickle.dumps(var))
            f.close()


if __name__ == "__main__":
    main()
