from graphcnn.experiment import *
import sys
import pandas as pd

def load_sa1_dataset():
    keys = []
    features = []
    labels = []
    lengths = []
    with open('Data/2018-06-03-SYD-SA1Input-Normalised.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')  # Last value in line is \n
            keys.append(s[0])
            features.extend([float(v) for v in s[1:-1]])  # Last column is the outcome y
            labels.append(int(s[-1]))
        lengths.append(i)
    with open('Data/2018-06-07-MEL-SA1Input-Normalised.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')  # Last value in line is \n
            keys.append(s[0])
            features.extend([float(v) for v in s[1:-1]])  # Last column is the outcome y
            labels.append(int(s[-1]))
        lengths.append(i)
    labels = np.array(labels)
    features = np.array(features).reshape((len(keys), -1))
    
    adj_mat = np.zeros((len(labels), 1, len(labels)))
    with open('Data/2018-06-03-SYD-NeighbourLinkFeatures.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys.index(s[0])
            b = keys.index(s[1])
            adj_mat[a, 0, b] = 1;
            adj_mat[b, 0, a] = 1;
    with open('Data/2018-06-07-MEL-NeighbourLinkFeatures.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys.index(s[0])
            b = keys.index(s[1])
            adj_mat[a, 0, b] = 1;
            adj_mat[b, 0, a] = 1;
    return (features, adj_mat, labels), lengths

dataset, lengths = load_sa1_dataset()

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




try:
    l = int(sys.argv[1])
    n = int(sys.argv[2])
    i = int(sys.argv[3])
except IndexError:
    l = 2
    n = 128
    i = 0


saveName = 'Output/SydneyToMelbourne-l={:d}-n={:d}-i={:d}.csv'.format(l,n,i)

max_acc = []
iteration = []
layers = []
neurons = []
rep = []



exp = SingleGraphCNNExperiment('2018-06-07-SydneyToMel', '2018-06-07-SydneyToMel', SA1Experiment(neurons = n, blocks = l))

exp.num_iterations = 1000
exp.optimizer = 'adam'

exp.debug = True  # Was True

exp.preprocess_data(dataset)

exp.train_idx = np.arange(lengths[0])
exp.test_idx = np.arange(lengths[0], lengths[0] + lengths[1])

results = exp.run()

max_acc.append(results[0])
iteration.append(results[1])
layers.append(l)
neurons.append(n)
rep.append(i)

max_acc_df = pd.DataFrame(max_acc, columns = ['Max Val Acc'])
iteration_df = pd.DataFrame(iteration, columns = ['Iteration'])
rep_df = pd.DataFrame(rep, columns = ['Repeat'])
l_df = pd.DataFrame(layers, columns = ['Layers'])
n_df = pd.DataFrame(neurons, columns = ['Neurons'])

df = pd.concat([n_df, l_df, rep_df, max_acc_df, iteration_df], axis = 1)
df.to_csv(saveName)