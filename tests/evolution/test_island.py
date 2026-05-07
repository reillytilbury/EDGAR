

import pytest
from src.evolution.island import seed
from src.evolution.population import Population
from tests.evolution.conftest import linear_model_code, linear_param_est_code, quadratic_model_code, quadratic_param_est_code

def test_seed(make_population):
    population = make_population
    seed_programs = [population[0], population[1]]
    n_islands = 3
    islands = seed(population, seed_programs, n_islands)
    assert len(population) == 2
    assert population[0].name == "LinearModel"
    assert population[1].name == "QuadraticModel"
    assert len(islands) == n_islands
    for island in islands:
        assert island == {0, 1}


