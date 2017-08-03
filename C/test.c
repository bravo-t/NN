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
    float alpha = 0;
    float reg = 1e-2;
    float learning_rate = 0.01;
    int minibatch_size = 300;
    int epochs = 20;
    TwoDMatrix* training_data = matrixMalloc(sizeof(TwoDMatrix));
    training_data = load2DMatrixFromFile("test_data/training_data.txt");
    TwoDMatrix* correct_labels = matrixMalloc(sizeof(TwoDMatrix));
    correct_labels = load2DMatrixFromFile("test_data/correct_labels.txt");
    parameters* train_params = malloc(sizeof(parameters));
    train_params = initTrainParameters(training_data, correct_labels,
        minibatch_size, // Size of the minibatch of training examples
        3, // How many scores will the nerual network calculate?
        learning_rate,
        reg,
        alpha,
        epochs,
        2, 30,10,3);
    train(train_params);
    return 0;
}
