import json
import os
from math import ceil, log
from random import random
from time import ctime, time

import numpy as np


class Hyperband:

    # alors ici on utilise la fonction get_params pour générer des configurations aleatoires et puis on fait le trraining avec la fonction main dans la quelle on passe t comme atrgs et nombre d'epochs

    def __init__(self, args, get_params_function, try_params_function):
        self.get_params = get_params_function
        self.try_params = try_params_function
        self.args = args
        self.max_iter = 81  	# maximum iterations per configuration 81
        self.eta = 3			# defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        # max_iter eq à R pour l'algorithme
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.best_result = []
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

    # can be called multiple times
    def run(self, dry_run=False, hb_result_file="/hb_result.json", hb_best_result_file="/hb_best_result.json"):

        # for s in reversed( range( self.s_max + 1 )):
        for s in range(self.s_max, 1, -1):
            # il faut juste comprendre ces paramtres la et tout est bon nrml

            # initial number of configurations

            n = int(ceil((self.B * self.eta ** s) / (self.max_iter * (s + 1))))
            print('n', n)
            print('im in s', s)

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            # get_params est équivalente à la fonction dget hyperparameter configuration(n) dans le papier de la recherche
            T = [self.get_params(self.args) for i in range(n)]

            for i in range((s + 1)):  # SuccessiveHalving loop

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations as the algorithm said
                print(n)

                n_configs = round(n * self.eta ** (-i))  # ni
                n_iterations = round(r * self.eta ** (i))  # ri

                print("*** {} configurations x {:.1f} iterations each".format(
                    n_configs, n_iterations))

                val_losses = []

                for t in T:

                    self.counter += 1

                    print("{} | {} | lowest loss so far: {:.4f} (run {})".format(
                        self.counter, ctime(), self.best_loss, self.best_counter)
                    )
                    start_time = time()

                    if dry_run:
                        result = {'best_val_loss': random(
                        ), 'log_loss': random(), 'auc': random()}
                    else:
                        if t['learning_rate'] < 0 or t['weight_decay'] < 0:
                            continue

                        while True:
                            try:
                                result = self.try_params(t, n_iterations)
                                break
                            except RuntimeError:
                                t['batchsize'] = int(t['batchsize']/1.33)
                                continue
                    assert(type(result) == dict)
                    assert('best_val_loss' in result)

                    seconds = int(round(time() - start_time))
                    print("{} seconds.".format(seconds))

                    loss = result['best_val_loss']
                    val_losses.append(loss)

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter
                        self.best_result = result
                        json.dump(self.best_result, open(
                            hb_best_result_file, 'w'))

                    self.results.append(result)
                    json.dump(self.results, open(hb_result_file, 'w'))
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                ##
                indices = np.argsort(val_losses)
                #T = [ T[i] for i in indices if not early_stops[i]]
                T = [T[i] for i in indices]
                T = T[0:int(n_configs / self.eta)]

        return self.results, self.best_result
