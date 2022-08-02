import numpy as np
from deap import base, algorithms, creator, tools
from ga_subgraph.individual import Individual
from ga_subgraph.fitness import GraphEvaluation, classifier
from ga_subgraph.selections import feasible, Penalty
import random

class GASubX(object):
    def __init__(self, blackbox, classifier, device, IndividualCls, n_gen, CXPB, MUTPB, ) -> None:
        self.model = blackbox
        self.device = device
        self.classifier = classifier
        self.penalty = 10
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.n_gen = n_gen

        self.IndividualCls = IndividualCls
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", self.IndividualCls, fitness=creator.FitnessMin)


    def explain(self, sample, subgraph_size, verbose=False):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
            toolbox.attr_bool, sample.num_nodes)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", GraphEvaluation(subgraph_size, self.model, self.classifier, self.device, sample))
        toolbox.decorate("evaluate", Penalty(feasible, self.penalty, sample))
        toolbox.register("mate", tools.cxPartialyMatched)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.MUTPB)
        toolbox.register("select", tools.selTournament, tournsize=20)


        # keep track of the best individuals
        hof = tools.HallOfFame(5)

        # setting the statistics (displayed for each generation)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register('avg', np.mean, axis=0)
        stats.register('min', np.min, axis=0)
        stats.register('max', np.max, axis=0)

        pop = toolbox.population(200)

        try:
            final_population, logbook = algorithms.eaMuPlusLambda(
                pop, toolbox,
                mu=10, lambda_=50, cxpb=self.CXPB, mutpb=self.MUTPB,
                ngen=self.n_gen, stats=stats, halloffame=hof, verbose=verbose)
        except (Exception, KeyboardInterrupt) as e:
            raise e

        if verbose:
            for individual in hof:
                print(f'hof: {individual.fitness.values[0]:.3f} << {individual}')

        return hof[-1].get_nodes(), logbook
