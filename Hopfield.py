import numpy as np
import random


def train_weights(train_data):
    num_data = len(train_data)

    # initialize weights

    weights = np.zeros((100, 100))

    # Hebb rule

    for i in range(num_data):
        t = train_data[i] - np.sum([np.sum(t) for t in train_data]) / (num_data * 100)
        weights += np.outer(t, t)  # addition of all weight matrices

    # Make diagonal elements of weights into 0
    diagW = np.diag(np.diag(weights))
    weights = weights - diagW
    weights /= num_data

    return weights


def predict(data, num_iter, weights, threshold=0):
    copied_data = np.copy(data)

    # Define vector for prediction

    predicted = []
    for i in range(len(data)):
        predicted.append(run_algo(copied_data[i], num_iter, weights, threshold))

    return predicted


def run_algo(init_state, num_iter, weights, threshold):
    state = init_state
    energy = -0.5 * state @ weights @ state + np.sum(state * threshold)

    for i in range(num_iter):
        random_order = random.sample(range(0, 100), 100)  # random order for asynchronous update

        for j in range(100):
            s_before = state
            #  select random neuron
            idx = random_order[j]
            # update state
            sum = 0
            for k in range(len(state)):
                if (k != idx):
                    sum = sum + (state[k] * weights[idx][k])

            state[idx] = init_state[idx] + sum
            if state[idx] == threshold:
                state[idx] = s_before[idx]
            elif state[idx] > threshold:
                state[idx] = 1
            else:
                state[idx] = -1

        # new state energy

        energy_new = -0.5 * state @ weights @ state + np.sum(state * threshold)

        # if s is converged

        if energy == energy_new:
            print("\nnumber of epochs took to converge", i)

            return state
        # else update energy
        energy = energy_new

    print("\n\nnumber of epochs took to converge", i)
    return state


#  x = test vector, p = noise percentage
def make_noise(x, p):
    v = random.sample(range(0, 99), p)

    for i in range(len(v)):
        if x[v[i]] == -1:
            x[v[i]] = 1
        else:
            x[v[i]] = -1


def show(x):
    n = len(x)
    for i in range(len(x)):
        if i % 10 == 0:
            print("")
        if x[i] == 1:
            print("*", end=" ")
        else:
            print(" ", end=" ")


def main():
    training_sets = [
        np.array([1, -1, -1, -1, -1, -1, -1, -1, 1, -1,
                  -1, 1, -1, -1, -1, -1, -1, -1, 1, -1,
                  -1, 1, 1, -1, -1, -1, -1, -1, 1, -1,
                  -1, 1, -1, 1, -1, -1, -1, -1, 1, -1,
                  -1, 1, -1, -1, 1, -1, -1, -1, 1, -1,
                  -1, 1, -1, -1, -1, 1, -1, -1, 1, -1,
                  -1, 1, -1, -1, -1, -1, 1, -1, 1, -1,
                  -1, 1, -1, -1, -1, -1, -1, 1, 1, -1,
                  -1, 1, -1, -1, -1, -1, -1, -1, 1, -1,
                  -1, 1, -1, -1, -1, -1, -1, -1, -1, 1]),  # alef
        #
        np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),  # bet
        #
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1]),  # gimel
        #
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1]),  # daled
        # #
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1]),  # hey
        # #
        np.array([-1, -1, -1, -1, -1, -1, -1, 1, 1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, 1]),  # vav
        #
        np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, 1, -1, -1]),  # zain

        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1]),  # het

        np.array([1, -1, -1, -1, -1, -1, -1, 1, 1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # tet

    ]

    testing_set = [
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
                  1, -1, -1, -1, -1, -1, -1, -1, -1, 1])  # het


    ]

    weights_ = train_weights(training_sets)

    print("testing set:")
    show(testing_set[0])

    for i in range(1):  # number of epochs
        noise_set = np.copy(testing_set)
        noise = 0  # noise percentage
        make_noise(noise_set[0], noise)
        print("\n\nvector after noise %", noise)
        show(noise_set[0])
        # predict function
        p = predict(noise_set, 1, weights_, 0)
        print("\n\nprediction vector")
        show(p[0])


if __name__ == '__main__':
    main()
