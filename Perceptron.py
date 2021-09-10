import statistics
import time
import numpy as np
from sklearn.preprocessing import StandardScaler


# for each matrix: first column is bias, last column is classification, rest of columns are inputs

def train_weights(matrix, y, weights, n_epoch, l_rate):
    for epoch in range(n_epoch):
        acc = accuracy(matrix, y, weights)
        print("epoch: ", epoch + 1)
        print("Accuracy: ", acc)
        if acc == 1.0:  # stop if accuracy is 1.0 which means that there is linear separation
            break

        for i in range(len(matrix)):
            prediction = predict(matrix[i], weights)  # get predicted classification
            error = y[i] - prediction  # get error from real classification
            for j in range(len(weights)):  # calculate new weight for each node
                weights[j] = weights[j] + (l_rate * error * matrix[i][j])

    return weights


def predict(inputs, weights):
    activation = 0.0
    for i, w in zip(inputs, weights):
        activation += i * w  # dot product of weights and inputs
    return 1.0 if activation >= 0.0 else 0.0  # Heaviside step function


def accuracy(matrix, y, weights):
    num_correct = 0.0
    for i in range(len(matrix)):
        p = predict(matrix[i], weights)  # get predicted classification
        if p == y[i]:
            num_correct += 1.0
    return num_correct / float(len(matrix))  # accuracy of prediction


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

    matrix[np.isnan(matrix)] = 3.2  # average of all last inputs instead of missing input

    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 1)

    matrix = StandardScaler().fit_transform(matrix)

    for i in range(len(matrix)):
        matrix[i][0] = 1.0  # add bias column

    matrix = np.concatenate((matrix, y[:, None]), axis=1)  # add classification column

    h_matrix = np.argsort(matrix[:, -1])  # sort matrix by classification
    new_matrix = matrix[h_matrix]

    N_matrix = new_matrix[0:151]  # matrix of N class (151)
    R_matrix = new_matrix[151:198]  # matrix of R class (47)

    #
    dev_arr = []  # array of all accuracy in testing for standard deviation
    sum_acc = 0.0  # sum of all accuracy in testing for average
    num = 3  # number of training and testing folds
    start_time = time.time()  # record time

    for j in range(num):

        print('\nFold number ', j + 1)

        #  weights initialisation (random very small numbers close to zero)

        rand = np.random.RandomState(1)
        weights = rand.normal(loc=0.0, scale=0.01, size=matrix.shape[1] - 1)

        # ------------- Training 133 samples

        train_matrix = generate_random_training_matrix(N_matrix, R_matrix, 101, 32)
        train_y = train_matrix[:, -1]  # turn last column into classification array
        train_matrix = np.delete(train_matrix, -1, 1)  # remove last column

        print("__training phase__")

        train_weights(matrix=train_matrix, y=train_y, weights=weights, n_epoch=30, l_rate=0.01)

        # ------------- Testing with 65 samples

        test_N = N_matrix[101:151]
        test_R = R_matrix[32:47]
        test_matrix = np.concatenate((test_N, test_R), axis=0)  # combine samples
        np.random.shuffle(test_matrix)

        test_y = test_matrix[:, -1]  # turn last column into classification array
        test_matrix = np.delete(test_matrix, -1, 1)  # remove last column

        print("\n__testing phase__")

        tp = 0  # true positive
        fp = 0  # false positive
        tn = 0  # true negative
        fn = 0  # false negative

        for i in range(len(test_matrix)):
            prediction = predict(test_matrix[i], weights)
            if test_y[i] == 1 and prediction == 1:
                tp = tp + 1
            if test_y[i] == 0 and prediction == 1:
                fp = fp + 1
            if test_y[i] == 0 and prediction == 0:
                tn = tn + 1
            if test_y[i] == 1 and prediction == 0:
                fn = fn + 1

        print("\ntrue positives count: ", tp)
        print("false positive count: ", fp)
        print("true negatives count: ", tn)
        print("false negatives count: ", fn)

        acc = accuracy(test_matrix, test_y, weights)
        print("Accuracy : ", acc)

        sum_acc = sum_acc + acc
        dev_arr.append(acc)
    avg = sum_acc / 3.0
    s_dev = statistics.stdev(dev_arr)
    print("\n__Cross-Validation__")
    print("Average: ", avg)
    print("Standard Deviation: ", s_dev)

    total_time = time.time() - start_time
    print("\nTime for training and testing(in seconds): ", total_time)


if __name__ == '__main__':
    main()
