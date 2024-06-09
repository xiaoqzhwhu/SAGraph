import random
import xflow.method.cosasi as co
import numpy as np
from evaluation import ranking_evaluation

def overall_evaluation(gt_names, predicted_items):
    print(gt_names)
    print(predicted_items)
    precision_1, recall_1, ndcg_1 = ranking_evaluation(gt_names, predicted_items, 1)
    precision_2, recall_2, ndcg_2 = ranking_evaluation(gt_names, predicted_items, 2)
    precision_5, recall_5, ndcg_5 = ranking_evaluation(gt_names, predicted_items, 5)
    precision_10, recall_10, ndcg_10 = ranking_evaluation(gt_names, predicted_items, 10)
    print("P@1=%.3f, R@1=%.3f, NDCG@1=%.3f" % (precision_1, recall_1, ndcg_1))
    print("P@2=%.3f, R@2=%.3f, NDCG@2=%.3f" % (precision_2, recall_2, ndcg_2))
    print("P@5=%.3f, R@5=%.3f, NDCG@5=%.3f" % (precision_5, recall_5, ndcg_5))
    print("P@10=%.3f, R@10=%.3f, NDCG@10=%.3f" % (precision_10, recall_10, ndcg_10))
    return

# TODO make seeds changable
def run (graph, diffusion, seeds, method, eval, epoch, budget, output, names, static_data, gt_names, seed_set):
# def run (graph, diffusion, method, eval, epoch, budget, output):

    print("Running " + eval.upper() + " :")

    for graph_fn in graph:
        try:
            print(graph_fn.__name__)
            g, config = graph_fn()
            print(g)
            seeds = random.sample(list(g.nodes()), 10)

            for method_fn in method:
                try:
                    print(method_fn.__name__)
                    baselines = ['eigen', 'degree', 'pi', 'sigma', 'Netshield', 'IMRank']
                    if method_fn.__name__ in baselines:
                        sims = method_fn(g, config, budget=10)
                        sims = [static_data[names[ids]]["user_name"] for ids in sims if names[ids] in seed_set]
                        overall_evaluation(gt_names, sims)
                        print(sims)
                    baselines = ['RIS']
                    if method_fn.__name__ in baselines:
                        sims = method_fn(g, config, budget=10)
                        sims = [static_data[names[ids]]["user_name"] for ids in sims]
                        overall_evaluation(gt_names, sims)
                        print(sims)
                    baselines = ['greedy', 'celf', 'celfpp']
                    if method_fn.__name__ in baselines:
                        for diffusion_fn in diffusion:
                            try:
                                print(diffusion_fn.__name__)
                                if eval == 'im':
                                    sims = method_fn(g, config, budget, rounds=epoch, model=diffusion_fn.__name__, beta=1)
                                    # print(sims)
                                    sims = [static_data[names[ids]]["user_name"] for ids in sims if names[ids] in seed_set]
                                    print(sims)
                                    overall_evaluation(gt_names, sims)
                                if eval == 'ibm':
                                    sims = method_fn(g, config, budget, seeds, rounds=epoch, model=diffusion_fn.__name__, beta=1)
                            except Exception as e:
                                print(f"Error when calling {diffusion_fn.__name__}: {str(e)}")

                    if method_fn.__name__ == 'netsleuth':
                        # todo seed shoule be changable
                        seed = 10
                        random.seed(seed)
                        np.random.seed(seed)

                        contagion = co.StaticNetworkContagion(
                            G=g,
                            model="si",
                            infection_rate=0.1,
                            # recovery_rate=0.005, # for SIS/SIR models
                            number_infected = 2,
                            seed=seed
                        )

                        contagion.forward(steps = 16)
                        
                        step = 15

                        # This obtains the indices of all vertices in the infected category at the 15th step of the simulation.
                        I = contagion.get_infected_subgraph(step=step)
                        print(I)
                        print("res")
                        sims = method_fn(I=I, G=g, hypotheses_per_step=1)
                        true_source = contagion.get_source()
                        evals = sims.evaluate(true_source)
                        top_dis= evals["distance"]["top score's distance"]
                        print(top_dis)
                        
                except Exception as e:
                    print(f"Error when calling {method_fn.__name__}: {str(e)}")    

        except Exception as e:
            print(f"Error when calling {graph_fn.__name__}: {str(e)}")
