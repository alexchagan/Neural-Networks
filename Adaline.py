import statistics
import time
import numpy as np
from sklearn.preprocessing import StandardScaler


#  y = classification
def train_weights(matrix, y, weights, l_rate, n_iter):

    for i in range(n_iter):
        acc = accuracy(matrix, y, weights)
        print("epoch: ", i+1)
        print("Accuracy: ", acc)
        if acc == 1.0:  # stop if accuracy is 1.0 which means that there is linear separation
            break

        output = net_input(matrix, weights)

        # calculating errors
        errors = y - output

        # calculating weight update
        weights[1:] += l_rate * matrix.T.dot(errors)
        weights[0] += l_rate * errors.sum()


def net_input(matrix, weights):
    return np.dot(matrix, weights[1:]) + weights[0]


def activation_function(matrix, weights):
    return net_input(matrix, weights)


def predict(matrix, weights):
    return np.where(activation_function(matrix, weights) >= 0.0, 1, -1)


def accuracy(matrix, y, weights):
    m_data_count = 0  # misclassified data
    for xi, target in zip(matrix, y):
        output = predict(xi, weights)
        if (target != output):
            m_data_count += 1

    total_data_count = len(matrix)
    acc = (total_data_count - m_data_count) / total_data_count  # accuracy calculation
    return acc


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
            c_list.append(-1)
    y = np.array(c_list)  # classification array

    # modifying dataset for our needs

    matrix = np.genfromtxt('wpbc.data', delimiter=',')

    matrix[np.isnan(matrix)] = 3.2  # average of all last inputs instead of missing input
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 1)  # final matrix after changes

    matrix = StandardScaler().fit_transform(matrix)

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
        weights = rand.normal(loc=0.0, scale=0.01, size=matrix.shape[1])

        # ------------- Training 133 samples

        train_matrix = generate_random_training_matrix(N_matrix, R_matrix, 101, 32)
        train_y = train_matrix[:, -1]  # turn last column into classification array
        train_matrix = np.delete(train_matrix, -1, 1)  # remove last column

        print("__training phase__")

        train_weights(train_matrix, train_y, weights, 0.001, 11)

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
            # print("Sample %d Expected=%d, Predicted=%d" % (i, test_y[i], prediction))
            if test_y[i] == 1 and prediction == 1:
                tp = tp + 1
            if test_y[i] == -1 and prediction == 1:
                fp = fp + 1
            if test_y[i] == -1 and prediction == -1:
                tn = tn + 1
            if test_y[i] == 1 and prediction == -1:
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
