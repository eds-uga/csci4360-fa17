import argparse
import numpy as np

STEP_SIZE = .001

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    parser = argparse.ArgumentParser(description = "Assignment 1",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment1.py [train-data] [train-label] [test-data]")
    parser.add_argument("paths", nargs = 3)
    args = vars(parser.parse_args())

    # get data from args paths
    with open(args['paths'][0], 'r') as tr_d_file:
        data = []
        for line in tr_d_file.read().split('\n'):
            if line.strip() != '':
                data.append(line.split(' '))
        train_data = np.array(data, dtype=int)

    with open(args['paths'][1], 'r') as tr_l_file:
        data = []
        for line in tr_l_file.read().split('\n'):
            if line.strip() != '':
                data.append(line)
        train_label = np.array(data, dtype=int)

    with open(args['paths'][2], 'r') as t_d_file:
        data = []
        for line in t_d_file.read().split('\n'):
            if line.strip() != '':
                data.append(line.split(' '))
        test_data = np.array(data, dtype=int)

    #get features for each
    train_rows = np.unique(train_data[:,0])
    train_features = np.unique(train_data[:,1])


    test_rows = np.unique(test_data[:,0])
    test_features = np.unique(test_data[:,1])
    features = np.unique(np.concatenate([train_features,test_features]))

    # print(len(features))
    train = np.zeros((len(train_rows),len(features)))
    test = np.zeros((len(test_rows),len(features)))

    for line in train_data:
        i = np.where(train_rows==line[0])
        j = np.where(features==line[1])
        tf = line[2]/len(features)
        idf = np.log(len(train_rows)/len(np.where((train_data[:,0]==line[0])*(train_data[:,1]==line[1]))))
        train[i,j] = tf*idf
        # print(train[i,j])
    for line in test_data:
        i = np.where(test_rows==line[0])
        j = np.where(features==line[1])
        tf = line[2]/len(features)
        idf = np.log(len(test_rows)/len(np.where((test_data[:,0]==line[0])*(test_data[:,1]==line[1]))))
        test[i,j] = tf*idf
        # print(test[i,j])


    y = train_label
    y_t = np.transpose([train_label])

    x = np.insert(train,0, 1,axis=1)

    x_t = np.transpose(x)
    m = len(train_label)
    epsilon =10 **(-25)

    theta = np.zeros(len(features)+1)
    t = np.matmul(x,np.transpose(theta))
    h = 1/(1 + np.exp(-1*t))
    cost = (1/m)*(np.matmul(np.log(h),-y_t) - np.matmul(np.log(1-h+epsilon),np.transpose([1-y])))
    grad = (STEP_SIZE/m) * np.matmul((h-y),x)
    iterations = 0
    while np.linalg.norm(grad) > .00000001:
        iterations+=1
        t = np.matmul(x,np.transpose(theta))
        h = 1/(1 + np.exp(-1*t))
        cost = (1/m)*(np.matmul(np.log(h),-y_t) - np.matmul(np.log(1-h+epsilon),np.transpose([1-y])))
        print(h.shape)
        grad = (STEP_SIZE/m) * np.matmul((h-y),x)

        theta = theta - grad

        print(iterations)
        print(cost)
        print(grad.shape)
    #pedict

    x_test = np.insert(test, 0,1,axis=1)

    x_test_t = np.transpose(x_test)
    lin_test = np.matmul(x_test, np.transpose(theta))
    h_test = 1/(1+np.exp(-1*lin_test))
    pred = np.round(h_test).astype(int)
    for vall in pred:
        print(vall)

    # Access command-line arguments as such:
    #
    # training_data_file = args["paths"][0]
    # training_label_file = args["paths"][1]
    # testing_data_file = args["paths"][2]
