#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdbool.h>
#include "util/matrix_type.h"
#include "util/misc_utils.h"
#include "util/matrix_operations.h"
#include "util/layers.h"
#include "util/fully_connected_net.h"


int main() {
    float alpha = 1e-3;
    float reg = 1e-2;
    float learning_rate = 0.001;
    int minibatch_size = 3;
    int epochs = 100;
    TwoDMatrix* X = matrixMalloc(sizeof(TwoDMatrix));
    X = load2DMatrixFromFile("test_data/X.txt");
    TwoDMatrix* correct_labels = matrixMalloc(sizeof(TwoDMatrix));
    correct_labels = load2DMatrixFromFile("test_data/y.txt");
    parameters* train_params = malloc(sizeof(parameters));
    train_params = initTrainParameters(X, correct_labels,
        minibatch_size, // Size of the minibatch of training examples
        3, // How many scores will the nerual network calculate?
        learning_rate,
        reg,
        alpha,
        epochs,
        2, 10,3);
    train(train_params);
    return 0;
}
