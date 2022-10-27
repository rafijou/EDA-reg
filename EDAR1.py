import random

import numpy
import numpy.random as np

from operator import attrgetter

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from myAlgorithms import eaGenerateUpdateW
# from sklearn.covariance import LedoitWolf
from ledoit_wolf import shrinkage

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMin)


class EDAR1(object):
    """
    Work in progress
    """

    def __init__(self, location_, mu, lambda_):
        self.dim = len(location_)
        self.location_ = numpy.array(location_)
        self.cov_ = numpy.diag(self.location_)
        self.lambda_ = lambda_
        self.mu = mu

    def generate(self, ind_init):
        # Generate lambda_ individuals and put them into the provided class

        nrg = np.default_rng()
        arz = [nrg.multivariate_normal(self.location_, self.cov_).clip(-600, 600)
               for i in range(self.lambda_)]

        return list(map(ind_init, arz))

    def update(self, population):
        # Sort individuals so the best is first
        sorted_pop = sorted(population, key=attrgetter("fitness"), reverse=True)

        best = sorted_pop[:self.mu]

        cov, average_cor, shrink = shrinkage(numpy.array(best))

        self.cov_ = cov
        self.location_ = numpy.diag(self.cov_)


def main():
    N, LAMBDA = 100, 25
    MU = int(LAMBDA / 4)
    strategy = EDAR1(location_=[5.0] * N, mu=MU, lambda_=LAMBDA)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", benchmarks.sphere)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=numpy.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # algorithms.eaGenerateUpdate(toolbox, ngen=100, stats=stats, halloffame=hof)
    eaGenerateUpdateW(toolbox, stats=stats, halloffame=hof)

    return hof[0].fitness.values[0]


if __name__ == "__main__":
    main()
