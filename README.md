## GAVulExplainer
genetic algorithm for Finding Subgraph

## Get started (using conda)
    1. create env with command `conda create -f binder/environment.yml`
    2. activate env with `conda activate ga_subgraph`

## using GAVulExplainer
    from ga_subgraph.explainer import GASubX
    from ga_subgraph.fitness import classifier
    from ga_subgraph.individual import Individual

    ga_explainer = GASubX(saved_model, classifier, device, Individual,.)
    ga_subgraph, _ = ga_explainer.explain(foo_sample, k_node, verbose=False)
