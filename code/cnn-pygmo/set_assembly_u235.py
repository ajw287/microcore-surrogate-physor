import sys, os
import re
import random
import pickle
import data_store as ds

def aw_round(x, base=0.2):
    return base * round(x/base)

def main():
    inputs = ds.data_store()
    num_pins = 9 # number of assemblies (with single pin conc)
    filename = sys.argv[1]
    inputs.enrichments = []
    try:
        file = open("./super_cell", 'r')
    except in_file_error:
        usage_error("input file error: (check path and that it exists!)")
        quit()
    else:
        filedata = file.read()
        for pn in range(1, num_pins+1):
            enrichment = aw_round(random.uniform(0.8, 5.0))
            print ("pin number "+str(pn))
            filedata = filedata.replace('%pin'+str(pn)+'%', "UO2_{:02d}".format(int(enrichment * 10)))
            inputs.enrichments.append(enrichment)
        # Write the file out again
        try:
            file = open(filename, 'w')
        except out_file_error:
            usage_error("file output error")
            quit()
        else:
            file.write(filedata)
            try:
                f = open(filename+".pickle", "wb+")  # you should be creating state.pickle here...
            except:
                print ("Error opening output file")
                exit()
            else:
                f.write(pickle.dumps(inputs))
                f.close()

if __name__ == "__main__":
    main()
