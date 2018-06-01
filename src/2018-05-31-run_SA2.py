from graphcnn.experiment import *

def load_sa2_dataset():
    keys = []
    features = []
    labels = []
#    categories = []
    with open('../../../Data/2018-05-31-SA2Input-Normalised.csv', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')  # Last value in line is \n
            keys.append(s[0])
            features.extend([float(v) for v in s[1:-1]])  # Last column is the outcome y
#            if s[-1] not in categories:
#                categories.append(s[-1])
#            labels.append(categories.index(s[-1]))
            ## Just use the imported labels directly, since it is an integer already, not a string like the CORA dataset
            labels.append(int(s[-1]))
        labels = np.array(labels)
        features = np.array(features).reshape((len(keys), -1))
    
    with open('../../../Data/2018-05-31-NeighbourDriveTimes.csv', 'r') as file:
        adj_mat = np.zeros((len(labels), 1, len(labels)))
        for i, line in enumerate(file):
            if i == 0:  # Skip first line (header)
                continue
            s = line[:-1].split(',')
            a = keys.index(s[0])
            b = keys.index(s[1])
            adj_mat[a, 0, b] = 1/(float(s[2]) + 1);  # Inverse of distance
            adj_mat[b, 0, a] = 1/(float(s[2]) + 1);
    return features, adj_mat, labels

dataset = load_sa2_dataset()

class SA2Experiment():
    def create_network(self, net, input):
        net.create_network(input)
#        net.make_embedding_layer(256)
#        net.make_dropout_layer()
        
#        net.make_graphcnn_layer(48)
#        net.make_dropout_layer()
#        net.make_embedding_layer(32)
#        net.make_dropout_layer()
        
        
#        net.make_graphcnn_layer(48)
#        net.make_dropout_layer()
#        net.make_embedding_layer(32)
#        net.make_dropout_layer()
        
        net.make_graphcnn_layer(48)
        net.make_dropout_layer()
        net.make_embedding_layer(32)
        net.make_dropout_layer()
        
        
        net.make_graphcnn_layer(48)
        net.make_dropout_layer()
        net.make_embedding_layer(32)
        net.make_dropout_layer()
        
        net.make_graphcnn_layer(48)
        net.make_dropout_layer()
        net.make_embedding_layer(32)
        net.make_dropout_layer()
        
        
        net.make_graphcnn_layer(10, name='final', with_bn=False, with_act_func = False)
        
exp = SingleGraphCNNExperiment('2018-05-31-SA2', '2018-05-31-sa2', SA2Experiment())

exp.num_iterations = 1000
exp.optimizer = 'adam'
exp.debug = True  # Was True
        
exp.preprocess_data(dataset)

acc, std = exp.run_kfold_experiments(no_folds=10)
print_ext('10-fold: %.2f (+- %.2f)' % (acc, std))
#
#2018-06-01 00:03:47.769377 Result is: 33.35 (+- 4.86)
#2018-06-01 00:03:47.797824 10-fold: 33.35 (+- 4.86)
#


# Doing just one run
#no_folds = 10
#inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)
#exp.train_idx, exp.test_idx = list(inst.split(np.arange(len(dataset[-1]))))[0]
##exp.train_idx, exp.test_idx = np.array(range(270)), list(range(270))  # Gives the same outcome. All nodes are reported regardless of training set or test set (the mask is only for the loss and training)
#max_acc, steps, y_pred = exp.run()



# Checking that this matches the reported values of accuracy
#import pandas as pd
#y = pd.DataFrame(y_pred.T)
#SA2Comb = pd.read_csv('../../../Data/SA2Combined.csv')
#np.sum(SA2Comb.decile.values == y.values.reshape(-1))
#np.mean(SA2Comb.decile.values == y.values.reshape(-1))



## Copied from self.get_max_accuracy
#tf.reset_default_graph()
#with tf.variable_scope('loss') as scope:
#    max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
#saver = tf.train.Saver()
#with tf.Session() as sess:
#    exp.load_model(sess, saver)
#    print(sess.run(max_acc_test))


# Prediction   -- Doesn't work
#with tf.Session() as sess:
##    sess.run(exp.y_pred_cls, feed_dict={exp.net.is_training:0})
#    reports = sess.run([exp.reports], feed_dict={exp.net.is_training:0})


### Getting predictions after training  -- This works. Need to reinitialise and then load weight...
#saver = tf.train.Saver()
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())   # Why do I have to reinitialise?
#    sess.run(tf.local_variables_initializer(), exp.variable_initialization)
#    exp.load_model(sess, saver)
#    y_pred = sess.run(exp.y_pred_cls, feed_dict={exp.net.is_training:0})
