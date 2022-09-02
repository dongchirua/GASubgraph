import numpy as np
from deap import base, creator, tools
from ga_subgraph.fitness import GraphEvaluation
from ga_subgraph.selections import feasible, Penalty
from ga_subgraph.generator import subgraph
from ga_subgraph.individual import init_population, generate_individual
from ga_subgraph.mating import cxCommonMatched
from ga_subgraph.mutation import mutate
from torch_geometric.utils import get_num_hops
from operator import attrgetter
import random
from tqdm.auto import tqdm
import logging

logging.basicConfig(
    format='%(asctime)s | %(name)s | %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('GASub')


class GASubX(object):
    def __init__(self, blackbox, classifier, device, IndividualCls, n_gen, CXPB, MUTPB, tournsize,
                 subgraph_building_method, max_population=400, offspring_population=100) -> None:
        """
        :param blackbox:
        :param classifier:
        :param device:
        :param IndividualCls:
        :param n_gen:
        :param CXPB:
        :param MUTPB:
        :param tournsize:
        :param subgraph_building_method:
        :param max_population:
        :param offspring_population:
        """
        self.model = blackbox
        self.device = device
        self.classifier = classifier
        self.penalty = 10
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.n_gen = n_gen
        self.l_hops = get_num_hops(blackbox)
        self.tournsize = tournsize
        self.subgraph_building_method = subgraph_building_method
        self.max_population = max_population
        self.offspring_population = offspring_population

        self.IndividualCls = IndividualCls
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", self.IndividualCls, fitness=creator.FitnessMin)

    def explain(self, sample, subgraph_size, verbose=False):
        toolbox = base.Toolbox()
        toolbox.register("subgraph", subgraph, num_hops=self.l_hops, x=sample.x, edge_index=sample.edge_index)
        toolbox.register("individual", generate_individual, subgraph_func=toolbox.subgraph,
                         num_nodes=sample.num_nodes, Individual_Cls=creator.Individual)
        toolbox.register("population", init_population, list, toolbox.individual)
        toolbox.register("feasible", feasible, origin_graph=sample, K=subgraph_size)
        toolbox.register("evaluate", GraphEvaluation(subgraph_size, self.model, self.classifier,
                                                     self.device, sample,
                                                     subgraph_building_method=self.subgraph_building_method))
        toolbox.decorate("evaluate", Penalty(toolbox.feasible, self.penalty))
        toolbox.register("mate", cxCommonMatched)
        toolbox.register("mutate", mutate, indpb=self.MUTPB, origin_graph=sample)
        toolbox.register("select", selTournament, tournsize=self.tournsize)

        # keep track of the best individuals
        hof = tools.HallOfFame(100)

        # setting the statistics (displayed for each generation)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register('avg', np.mean, axis=0)
        stats.register('min', np.min, axis=0)
        stats.register('max', np.max, axis=0)

        pop = toolbox.population(sample.num_nodes)

        try:
            final_population, logbook = eaMuPlusLambda(
                pop, toolbox,
                mu=self.max_population, lambda_=self.offspring_population, cxpb=self.CXPB, mutpb=self.MUTPB,
                ngen=self.n_gen, stats=stats, halloffame=hof, verbose=verbose)
        except (Exception, KeyboardInterrupt) as e:
            raise e

        if verbose:
            for individual in hof:
                logger.info(f'hall of frame: {individual.fitness.values[0]:.4f} << {individual}')

        return hof[0].get_nodes(), logbook


def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.

    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)

    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'psize', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    population = [inv for inv in list(set(population)) if toolbox.feasible(inv)]
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitness = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    def main_func():
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        offspring = list(set(offspring))  # get unique offspring
        offspring = [i for i in offspring if toolbox.feasible(i)]  # remove invalid offspring

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, psize=len(population), nevals=len(invalid_ind), **record)

    if verbose:
        logger.info(logbook.stream)
        for gen in range(1, ngen + 1):
            main_func()
            logger.info(logbook.stream)
    else:
        with tqdm(total=ngen) as t:
            for gen in range(1, ngen + 1):
                main_func()
                t.update()

    return population, logbook


def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: Control the number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    individuals = list(set(individuals))  # only unique individuals
    individuals.sort(key=attrgetter(fit_attr), reverse=True)
    size = max(len(individuals), int(len(individuals) / tournsize))
    chosen = individuals[:tournsize]
    k = min(k, len(individuals))
    for i in range(k):
        random.shuffle(individuals)
        aspirants = individuals[:size]
        sel = max(aspirants, key=attrgetter(fit_attr))
        sel_index = aspirants.index(sel)
        chosen.append(sel)
        del aspirants[sel_index]
        individuals = aspirants + individuals[size:]

    chosen = list(set(chosen))  # get unique individuals
    return chosen


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population.

    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.

    This variation is named *Or* because an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb and len(population) > 2:  # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
            offspring.append(ind1)
            offspring.append(ind2)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:  # Apply reproduction
            offspring.append(random.choice(population))

    return offspring
