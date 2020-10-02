import sys, os
#from problem_smicro import problem_smicro
import copy
import problem_susmicro as psm
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pygmo import population, problem, algorithm, plot_non_dominated_fronts, moead, nsga2, hypervolume, fast_non_dominated_sorting, plot_non_dominated_fronts

from timeit import default_timer as timer

class plotDataOut:
    pass

class save_data:
    pass

def dist(a,b):
    return np.sqrt(np.sum((a-b)**2, axis=0))

def aw_round(x, base=0.2):
    return base * round(x/base)

def main():
    generation = 0
    if len(sys.argv)>2:
        if(sys.argv[2] == 'show'):
            show = True
        else:
            show=False
    else:
        show=False
    file_base_name = sys.argv[1]
    plotdata = plotDataOut()
    save = save_data()
    prob = problem(psm.problem_susmicro())
    pop = population(prob,  size = 60, seed = 3453412)
    specific_algo = nsga2(gen = 1, seed = 3453213)
#    pop = population(prob, size = 210, seed = 3453412)
#    algo = algorithm(moead(gen = 20)) # 250
    algo = algorithm(specific_algo)
    initial_inputs = pop.get_x()
    initial_outputs = pop.get_f()
    ndf, dl, dc, ndl = fast_non_dominated_sorting(initial_outputs)
    save.initial_ndf = copy.deepcopy(ndf)
    save.initial_inputs = copy.deepcopy(initial_inputs)
    save.initial_outputs = copy.deepcopy(initial_outputs)

    start = timer()
    for generation in range(0, 50):
        save.pop = copy.deepcopy(pop)
        ################# initial pop ##################
        #get a list of the non-dominated front of the first set (random points)
        ndf, dl, dc, ndl = fast_non_dominated_sorting(pop.get_f())
        ndf_x = []
        for val in ndf[0]:
            ndf_x.append(pop.get_x()[val])
        save.initial_surr_ndf_x = copy.deepcopy(ndf_x)
        print("evaluate the initial ndf in surrogate")
        inputs = pop.get_x()
        outputs = pop.get_f()
        save.inputs = copy.deepcopy(inputs)
        save.outputs = copy.deepcopy(outputs)
        data_file = "./pop_"+str(generation).zfill(4)+".pickle"
        save.pop = copy.deepcopy(pop)
        try:
            pickle.dump(save, open(data_file, "wb+" ) )
        except:
            print("error opening '"+data_file+"' pickle file")
            exit()
        pop = algo.evolve(pop)
    end = timer()


    ndf, dl, dc, ndl = fast_non_dominated_sorting(pop.get_f())
    ndf_x = []
    for val in ndf[0]:
        ndf_x.append(pop.get_x()[val])
    save.final_surr_ndf_x = copy.deepcopy(ndf_x)
    final_inputs = pop.get_x()
    final_outputs = pop.get_f()
    save.inputs = copy.deepcopy(final_inputs)
    save.outputs = copy.deepcopy(final_outputs)

    plotdata.final_inputs = copy.deepcopy(final_inputs)
    plotdata.final_outputs = copy.deepcopy(final_outputs)
    with open(file_base_name+"_plot_data.pickle","wb") as f:
        f.write(pickle.dumps(plotdata))
    # Plot
    fig, ax = plt.subplots()
#    x_vals_f = [(row[0] + row[1]) / 2.0 for row in final_outputs]
    x_vals_f = [row[0] for row in final_outputs]#[dist(row[0], row[1]) for row in final_outputs]
    y_vals_f = [row[1] for row in final_outputs]#[row[2] * 2.0 for row in final_outputs]
    ax.scatter(x_vals_f, y_vals_f, c="purple", alpha=0.6, label='Final ndf surrogate model')
#    x_vals_i = [(row[0] + row[1])/2.0 for row in initial_outputs]
    x_vals_i = [row[0] for row in initial_outputs]#[dist(row[0], row[1]) for row in initial_outputs]
    y_vals_i = [row[1] for row in initial_outputs]#[row[2] * 2.0 for row in initial_outputs]
    ax.scatter(x_vals_i, y_vals_i, c="green", alpha=0.6, label='initial evaluation')
    ax.set_title('Initial to Surrogate population')
    ax.set_ylabel('ppf')
    ax.set_xlabel('1/radius')
    ax.legend(loc=1)
    #fig.savefig('surrogate-wims.png')
    #fig.show()
    # if you are debugging probably just show to screen
    if(show):
        print("\a")
        plt.show()
        fig.savefig(file_base_name+'-graph.svg')
    else:
        # if not debugging save the figure
        fig.savefig(file_base_name+'-graph.png')
        fig.savefig(file_base_name+'-graph.svg')

    # Plot only NDF
    #fig, ax = plt.subplots()
    initials = [[a,b] for a,b in zip(x_vals_i,y_vals_i)]
    finals = [[a,b] for a,b in zip(x_vals_f,y_vals_f)]
    plt.ylim([0,6])
    plt.xlim([0,0.6])
    ax = plot_non_dominated_fronts(initials, marker='o')
    plt.ylim([0,6])
    plt.xlim([0,0.6])
    ax = plot_non_dominated_fronts(finals, marker='x',)

    ax.set_title('Surrogate NDF to Serpent NDF')
    ax.set_ylabel('ppf')
    ax.set_xlabel('1/radius (relative)')
#    ax.legend(loc=1)
    if(show):
        print("\a")
        plt.show()
        fig.savefig(file_base_name+'-ndf_only.svg')
    else:
        # if not debugging save the figure
        fig.savefig(file_base_name+'-ndf_only.png')
        fig.savefig(file_base_name+'-ndf_only.svg')


    ndf_simplified = [] # copy.deepcopy(ndf[0])
    for idx in range(len(pop.get_x())):#ndf[0]:
        x = pop.get_x()[idx]
        new_row = [aw_round(val) for val in x]
        ndf_simplified.append(new_row)
    #ndf_simplified = list(set(ndf_simplified))
    print(str(len(ndf_simplified)))
    ndf_no_repeat = np.unique(ndf_simplified, axis=0)
    print("data from populations:"+str(len(pop))+" "+str(len(ndf_no_repeat)))
    print(str(ndf_no_repeat))

    line = "input_test_list = ["
    # get whole final population and print it out...
    for vals in ndf_no_repeat:
        line +="["
        for v in vals:
            line += str(v)+", "
        line += "],\n"
    line += "]\n"
    print(line)
    ndf, dl, dc, ndl = fast_non_dominated_sorting(pop.get_f())
    for idx in ndf_simplified:
        f = ndf_simplified#pop.get_f()[idx]
        print("f: "+str(f))
    print("NDF len:" + str(len(ndf_no_repeat)))
    print("Execution time for evolution: " + str(end-start))


if __name__ == "__main__":
    main()
