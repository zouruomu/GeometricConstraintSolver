import pickle
from source.utils.evaluation import evaluate

def test_evaluation():
    dataset = pickle.load(open("data/Dataset10000.pkl", "rb"))
    for datapoint in range(min(10, len(dataset))):
        init_score = evaluate(dataset[datapoint]["initial_objects"], dataset[datapoint]["constraints"])
        solved_score = evaluate(dataset[datapoint]["solved_objects"], dataset[datapoint]["constraints"])
        print(init_score, solved_score)
        assert solved_score <= init_score
        if datapoint > 5:
            break