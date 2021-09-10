import numpy as np
import time
import statistics
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense


def generate_random_training_matrix(matrix_A, matrix_B, n1, n2):
    np.random.shuffle(matrix_A)
    np.random.shuffle(matrix_B)

    a_samples = matrix_A[0:n1]  # pick random n1 samples
    b_samples = matrix_B[0:n2]  # pick random n2 samples

    matrix = np.concatenate((a_samples, b_samples), axis=0)  # combine samples
    np.random.shuffle(matrix)
    return matrix


def main():
    c_list = []  # classification list
    file = open('wpbc.data', "r")
    for line in file:
        values = line.split(',')
        if values[1] == "R":
            c_list.append(1)
        else:
            c_list.append(0)
    y = np.array(c_list)  # classification array

    # modifying dataset for our needs

    matrix = np.genfromtxt('wpbc.data', delimiter=',')
    matrix[np.isnan(matrix)] = 3.2  # average of all last inputs instead of '?'
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 1)

    matrix = StandardScaler().fit_transform(matrix)

    matrix = np.concatenate((matrix, y[:, None]), axis=1)  # add classification column

    h_matrix = np.argsort(matrix[:, -1])  # sort matrix by classification
    new_matrix = matrix[h_matrix]

    N_matrix = new_matrix[0:151]  # matrix of N class (151)
    R_matrix = new_matrix[151:198]  # matrix of R class (47)

    dev_arr = []  # array of all accuracy in testing for standard deviation
    sum_acc = 0.0  # sum of all accuracy in testing for average
    num = 3  # number of training and testing folds
    start_time = time.time()  # record time

    for j in range(num):
        print('\nFold number ', j + 1)

        # ------------- Training 133 samples

        train_matrix = generate_random_training_matrix(N_matrix, R_matrix, 101, 32)
        train_y = train_matrix[:, -1]  # turn last column into classification array
        train_matrix = np.delete(train_matrix, -1, 1)  # remove last column

        model = Sequential()
        model.add(Dense(23, input_dim=32, activation='relu'))  # 32 output neurons + bias, 23 hidden layers neurons
        model.add(Dense(1, activation='sigmoid'))  # sigmoid activation func for hidden layer
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])  # loss func for binary classification

        print("\ntraining phase")
        model.fit(train_matrix, train_y, epochs=50, batch_size=133, verbose=1)

        # ------------- Testing with 65 samples

        test_N = N_matrix[101:151]
        test_R = R_matrix[32:47]
        test_matrix = np.concatenate((test_N, test_R), axis=0)  # combine samples
        np.random.shuffle(test_matrix)

        test_y = test_matrix[:, -1]  # turn last column into classification array
        test_matrix = np.delete(test_matrix, -1, 1)  # remove last column
        predictions = model.predict_classes(test_matrix)


        print("\ntraining phase")

        tp = 0  # true positive
        fp = 0  # false positive
        tn = 0  # true negative
        fn = 0  # false negative

        for i in range(65):

            if test_y[i] == 1 and predictions[i] == 1:
                tp = tp + 1
            if test_y[i] == 0 and predictions[i] == 1:
                fp = fp + 1
            if test_y[i] == 0 and predictions[i] == 0:
                tn = tn + 1
            if test_y[i] == 1 and predictions[i] == 0:
                fn = fn + 1

        print("\ntrue positives count: ", tp)
        print("false positive count: ", fp)
        print("true negatives count: ", tn)
        print("false negatives count: ", fn)

        acc = model.evaluate(test_matrix, test_y, verbose=0)
        print('Test accuracy: ', acc[1])

        sum_acc = sum_acc + acc[1]
        dev_arr.append(acc[1])

    avg = sum_acc / 3.0
    s_dev = statistics.stdev(dev_arr)
    print("\n__Cross-Validation__")
    print("Average: ", avg)
    print("Standard Deviation: ", s_dev)
    end_time = time.time() - start_time
    print("\nTime for training and testing(in seconds): ", end_time)


if __name__ == '__main__':
    main()
