from graphcnn.experiment import *
import sys
import pandas as pd

def load_sa1_dataset():
    keys = []
    features = []
    labels = []
    with open('Data/2018-06-01-NSW-SA1Input-Normalised.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')  # Last value in line is \n
            keys.append(s[0])
            features.extend([float(v) for v in s[1:-1]])  # Last column is the outcome y
            labels.append(int(s[-1]))
        labels = np.array(labels)
        features = np.array(features).reshape((len(keys), -1))
    
    with open('Data/2018-06-01-NSW-NeighbourLinkFeatures.csv', 'r') as file:
        adj_mat = np.zeros((len(labels), 2, len(labels)))
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys.index(s[0])
            b = keys.index(s[1])
            adj_mat[a, 0, b] = float(s[-2]);  # Inverse of drive time
            adj_mat[b, 0, a] = float(s[-2]);
            adj_mat[a, 1, b] = 1;
            adj_mat[b, 1, a] = 1;
    return features, adj_mat, labels

dataset = load_sa1_dataset()

class SA1Experiment():
    def __init__(self, neurons, blocks):
        self.blocks = blocks
        self.neurons = neurons
    
    def create_network(self, net, input):
        net.create_network(input)
        net.make_embedding_layer(self.neurons)
        net.make_dropout_layer()
        
        for _ in range(self.blocks):
            net.make_graphcnn_layer(self.neurons)
            net.make_dropout_layer()
            net.make_embedding_layer(self.neurons)
            net.make_dropout_layer()
        
        net.make_graphcnn_layer(10, name='final', with_bn=False, with_act_func = False)


no_folds = 5
inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)

ls = [1, 2, 3]
ns = [64, 128]
r = int(sys.argv[1])
i = r // 6  # Repetition
j = r % 6  # Parameters
l = ls[j // len(ns)]
n = ns[j % len(ns)]

saveName = '../../../Output/2018-06-07-SemiSupervisedNSW-DriveTime-l={:d}-n={:d}-i={:d}.csv'.format(l,n,i)

max_acc = []
max_acc_iteration = []
layers = []
neurons = []
rep = []

    

exp = SingleGraphCNNExperiment('2018-06-06-SA1', '2018-06-06-sa1', SA1Experiment(neurons = n, blocks = l))

exp.num_iterations = 2000
exp.optimizer = 'adam'

exp.debug = True  # Was True

exp.preprocess_data(dataset)

#exp.train_idx, exp.test_idx = list(inst.split(np.arange(len(dataset[-1]))))[i]
exp.test_idx, exp.train_idx = list(inst.split(np.arange(len(dataset[-1]))))[i]  # Reversed to get more samples in the test set than the training set
results = exp.run()

max_acc.append(results[0])
max_acc_iteration.append(results[1])
layers.append(l)
neurons.append(n)
rep.append(i)

max_acc_df = pd.DataFrame(max_acc, columns = ['Max Val Acc'])
max_acc_iteration_df = pd.DataFrame(max_acc_iteration, columns = ['Max Val Acc Iteration'])
rep_df = pd.DataFrame(rep, columns = ['Repeat'])
l_df = pd.DataFrame(layers, columns = ['Layers'])
n_df = pd.DataFrame(neurons, columns = ['Neurons'])

df = pd.concat([n_df, l_df, rep_df, max_acc_df, max_acc_iteration_df], axis = 1)
df.to_csv(saveName)