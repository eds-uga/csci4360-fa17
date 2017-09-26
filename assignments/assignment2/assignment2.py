import argparse
import numpy as np

def _load_X(args):
    # Load the data.
    mat = np.loadtxt(args['paths'][0], dtype = int)
    mat_t = np.loadtxt(args['paths'][2],dtype = int)
    train_rows = np.unique(mat[:,0])
    train_features = np.unique(mat[:,1])

    test_rows = np.unique(mat_t[:,0])
    test_features = np.unique(mat_t[:,1])
    features = np.unique(np.concatenate([train_features,test_features]))

    X= get_tfidf(mat, train_rows, features)
    X_test = get_tfidf(mat_t, test_rows, features)
    return X, X_test,features, train_rows

def _load_train(args, labels):
    # Load the labels.
    y = np.loadtxt(labels, dtype = int)
    X, X_test, features, train_rows= _load_X(args)

    # Return.
    return X, y, X_test, features, train_rows

def accuracy (y_pred, y_true):
    #give two one dimensional row arrays of equal length
    accuracy_rate = np.sum(y_pred==y_true)/len(y_true)
    return accuracy_rate

def get_loss(h, y, theta):
    m = len(y)
    print(len(y))
    epsilon =10 **(-25)
    l = 3
    y_t= np.transpose(y)
    cost = ((1/m)*(np.matmul(np.log(h),-y_t) - np.matmul(np.log(1-h+epsilon),np.transpose(1-y)))) + l*(np.sum(np.absolute(theta)))

    return cost

def pred_diff(y_pred, y_true):
    error_terms = y_pred-y_true
    abs_error_terms = np.absolute(error_terms)
    error_rate = np.mean(abs_error_terms)
    return error_rate

def get_tfidf(data, train_rows, features):
    data_tfidf= np.zeros((len(train_rows),len(features)))
    for line in data:
        i = np.where(train_rows==line[0])
        j = np.where(features==line[1])
        tf = line[2]/len(features)
        idf = np.log(len(train_rows)/len(np.where((data[:,0]==line[0])*(data[:,1]==line[1]))))
        data_tfidf[i,j] = tf*idf
    return data_tfidf

def predict(theta, X):
    t = np.matmul(X,np.transpose(theta))
    h = 1/(1 + np.exp(-1*t))
    h = np.round(h)
    return h

def init_wts(features):
    weights = np.random.normal(loc=0, scale=3, size=len(features))
    return weights

def init_pop(features, pop_size):
    pop = [init_wts(features) for i in range(0,100)]
    return pop

def fitness(pop, X, y_true):
    fitness = []
    for theta in pop:
        y_pred = predict(theta, X)
        fitness.append([accuracy(y_pred, y_true),theta])
        # fitness.append([pred_diff(y_pred, y_true),theta])
        # fitness.append([get_loss(y_pred, y_true, theta),theta])
    return fitness

def make_child(parent1, parent2):
    child = np.array([(parent1 + parent2)/2])
    return child

def next_gen(pop_fit, top_percent, mut_rate):
    pop_size = len(pop_fit)
    kept = int(pop_size*top_percent)
    num_kids = pop_size-kept
    parents = sorted(pop_fit, key=lambda pop: -pop[0])
    # print(parents)
    parents = np.array([i[1] for i in parents])
    mean = np.mean(parents, axis=0)
    std_dev = np.std(parents, axis=0)
    parents = parents[:kept,:]
    children = []
    for i in range(0, num_kids):
        mutate = False
        idx1 = np.random.randint(kept)
        idx2 = np.random.randint(kept)
        parent1 = parents[idx1,:]
        parent2 = parents[idx2,:]
        child = make_child(parent1, parent2)
        mut = (np.random.rand()<mut_rate)
        if mut:
            child = mutating(child, mean, std_dev)
        children.append(child[0])
    next_pop = np.concatenate([parents,children])
    return next_pop

def mutating(child, mean, std_dev):
    new_child = np.zeros(child.shape)
    for i in range(0,len(new_child[0])):
        new_child[0,i]= np.random.normal(mean[i], std_dev[i])
    return new_child

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 2",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment2.py [train-data] [train-label] [test-data] <optional args>")
    parser.add_argument("paths", nargs = 3)
    parser.add_argument("-n", "--population", default = 100, type = int,
        help = "Population size [DEFAULT: 100].")
    parser.add_argument("-s", "--survival", default = 0.3, type = float,
        help = "Per-generation survival rate [DEFAULT: 0.3].")
    parser.add_argument("-m", "--mutation", default = 0.01, type = float,
        help = "Point mutation rate [DEFAULT: 0.01].")
    parser.add_argument("-g", "--generations", default = 100, type = int,
        help = "Number of generations to run [DEFAULT: 100].")
    parser.add_argument("-r", "--random", default = -1, type = int,
        help = "Random seed for debugging [DEFAULT: -1].")
    args = vars(parser.parse_args())

    # Do we set a random seed?
    if args['random'] > -1:
        np.random.seed(args['random'])

    # Read in the training data, test data and feautures
    X, y, X_test, features, train_rows = _load_train(args, args["paths"][1])
    pop = init_pop(features, args['population'])

    for i in range(0,args['generations']):
        pop_fit = fitness(pop, X, y)
        next_pop = next_gen(pop_fit, args['survival'], args['mutation'])
        pop = next_pop
    theta = pop[1,:]
    y_pred_test = np.round(predict(theta, X_test))
    # y_test_true= np.loadtxt('./data/test_partial.label', dtype = int)
    # y_test = np.column_stack([y_pred_test, y_test_true])
    for label in y_pred_test :
        print(int(label))


    ### FINISH ME
