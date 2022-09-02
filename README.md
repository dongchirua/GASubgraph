## GAVulExplainer
genetic algorithm for Finding Subgraph

## Get started (Python version >= 3.9)
    1a. (CPU) create env with command `conda create -f binder/environment.yml`
    1b. (GPU) create env with command `conda create -f binder/environment-cu11.3.yml`
    2. activate env with `conda activate ga_subgraph`
    3. Download Reveal dataset: https://bit.ly/3bX30ai
    4. We flowed and used Joern which was provided along with Reveal paper at https://github.com/VulDetProject/ReVeal/blob/master/code-slicer/joern/README.md. Using Joern to parse data

In case you want to install yourself, below are major libs we used
    
    1. PyTorch
    2. PyTorch Geometric
    3. PyTorch Lightning
    4. networkx
    5. DEAP
    6. nltk
    7. gensim

## How to use GAVulExplainer
In case, you want to use our prepared example (`example.py`), download `data.zip` at https://drive.google.com/file/d/1eQBfx3OAOZLJrmX2wby5S_Z_HiWW0BT9/view?usp=sharing, unzip `data.zip`, and `weights.zip` at project level.

In order to ultilize `GAVulExplainer` for other tasks, please follow below instruction

    from ga_subgraph.explainer import GASubX
    from ga_subgraph.fitness import classifier
    from ga_subgraph.individual import Individual
    k_node = 5  # explanation size
    # foo_sample is PyTorch Geometric Data
    ga_explainer = GASubX(saved_model, classifier, device, Individual,.)
    ga_subgraph, _ = ga_explainer.explain(foo_sample, k_node, verbose=False)

Documents of GASubX
```
:param blackbox: PyTorch model
:param classifier: Function to get probability from model, example: `ga_subgraph.fitness.classifier`
:param device: cuda or cpu
:param IndividualCls: Class to store individual representation
:param n_gen: how many generation to perform
:param CXPB: crossover probabitliy
:param MUTPB: mutation probability
:param tournsize: factor control selection function
:param subgraph_building_method: function to construct subgraph
:param max_population: control max individual for every generation
:param offspring_population: control number of offsprint individuals
```

## preproduce our experiments (vary explanation size)
 + unzip `data.zip`, and `weights.zip`
 + run `python do_statistic 4 cuda`. 4 is explanation size, cuda is device
 + the script will ultilize multi-processors to perform explaination parallelly
 + at `do_statistic` line 106: config DataSet 
 + at `do_statistic` line 52: config pretrained model
 + we share raw result for undirect graphs at `statistics` folder, direct graphs at `statistics_undirected`

## project structure
 + weights folder: we stored pretrain classifer here
 + data: store data, word2vec model
 + binder: we locked lib versions for this project
 + ga_subgraph: our implementation for GAVulExplainer
 + visualization: helpers for visualize
 + vulexp: helpers for tranining vulnerability predictor, data processing, and SubgraphX. We demonstrate in `example.py`.
 + `vulexp/reveal_data.py`: class handle Reveal dataset