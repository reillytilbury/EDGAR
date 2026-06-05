EDGAR is an AI equation/model discovery system for scientific problems.
It is being actively developed and applied to new kinds of problems, so the exact problems it is applied to are changing and the exact algorithmic details may differ from problem to problem (for example evolutionary algorithms may change or the types of models generated could change, ranging from simple equations or neural networks)
It's core functionality is as follows:
- Use an evolutionary algorithm to evolve programs to solve a scientific problem.
- LLMs are used to generate new programs based on the evolutionary algorithm, using previous programs and the scoring of these programs.
- Scores are problem-dependendent and specified by the user, but can include things like the mean-squared error to some experimental data.
- The goal is to find a program which finds the best possible score.