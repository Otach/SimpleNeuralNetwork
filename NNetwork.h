#include "Matrix.h"

typedef struct Layer {
    int size;
    Matrix *weights;
    Matrix *outputs;
} Layer;

typedef struct NeuralNetwork {
    int32_t input;
    int32_t hidden;
    int32_t output;
    float learning_rate;
    Layer *hidden_layers;
    Layer *output_layer;
} NeuralNetwork;

NeuralNetwork *allocNN(int input, int num_hidden, int *hidden_layers, int output, float lrate){
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    int prev_size = 0;
    nn->input = input;
    nn->hidden = num_hidden;
    nn->output = output;
    nn->learning_rate = lrate;

    nn->hidden_layers = (Layer *)malloc(sizeof(Layer) * nn->hidden);
    for (int i = 0; i < nn->hidden; i++) {
        nn->hidden_layers[i].size = hidden_layers[i];
        prev_size = (i == 0) ? input : nn->hidden_layers[i-1].size;
        nn->hidden_layers[i].weights = allocMatrix(nn->hidden_layers[i].size, prev_size);
        randomizeMatrix(nn->hidden_layers[i].weights);
    }
    nn->output_layer = (Layer *)malloc(sizeof(Layer));
    nn->output_layer->weights = allocMatrix(output, nn->hidden_layers[nn->hidden-1].size);
    randomizeMatrix(nn->output_layer->weights);

    return nn;
}

int deallocNN(NeuralNetwork *nn) {

    for (int i = 0; i < nn->hidden; i++) {
        deallocMatrix(nn->hidden_layers[i].weights);
    }
    free(nn->hidden_layers);
    deallocMatrix(nn->output_layer->weights);
    free(nn->output_layer);

    free(nn);
    return 0;
}

float *feedforward(NeuralNetwork *nn, float *input){
    Matrix *inputs = allocMatrix(nn->input, 1);
    float *result;

    if (inputFromArray(input, nn->input, inputs) == -1) {
        return NULL;
    }

    for (int i = 0; i < nn->hidden; i++) {
        if (i == 0) {
            nn->hidden_layers[i].outputs = product(nn->hidden_layers[i].weights, inputs);
        } else {
            nn->hidden_layers[i].outputs = product(nn->hidden_layers[i].weights, nn->hidden_layers[i-1].outputs);
        }
        sigmoid(nn->hidden_layers[i].outputs);
    }

    nn->output_layer->outputs = product(nn->output_layer->weights, nn->hidden_layers[nn->hidden-1].outputs);
    sigmoid(nn->output_layer->outputs);

    result = toArray(nn->output_layer->outputs);

    deallocMatrix(inputs);
    for (int i = 0; i < nn->hidden; i++) {
        deallocMatrix(nn->hidden_layers[i].outputs);
    }
    deallocMatrix(nn->output_layer->outputs);

    return result;
}

int train(NeuralNetwork *nn, float *input, float *target){
    Matrix *inputs = allocMatrix(nn->input, 1);
    Matrix *targets = allocMatrix(nn->output, 1);
    Matrix *output_errors;
    Matrix *gradients;
    Matrix *hidden_T;
    Matrix *weight_ho_deltas;
    Matrix *who_t;
    Matrix *hidden_errors;
    Matrix *hidden_gradient;
    Matrix *inputs_T;
    Matrix *weight_ih_deltas;

    if (inputFromArray(input, nn->input, inputs) == -1) {
        return -1;
    }

    for (int i = 0; i < nn->hidden; i++) {
        if (i == 0) {
            nn->hidden_layers[i].outputs = product(nn->hidden_layers[i].weights, inputs);
        } else {
            nn->hidden_layers[i].outputs = product(nn->hidden_layers[i].weights, nn->hidden_layers[i-1].outputs);
        }
        sigmoid(nn->hidden_layers[i].outputs);
    }

    nn->output_layer->outputs = product(nn->output_layer->weights, nn->hidden_layers[nn->hidden-1].outputs);
    sigmoid(nn->output_layer->outputs);

    if (inputFromArray(target, nn->output, targets) == -1) {
        return -2;
    }

    output_errors = matrixSubtract(targets, nn->output_layer->outputs);
    deallocMatrix(targets);

    gradients = dsigmoid(nn->output_layer->outputs);
    matrixMultiply(gradients, output_errors);
    multiply(gradients, nn->learning_rate);

    hidden_T = transpose(nn->hidden_layers[nn->hidden-1].outputs);
    weight_ho_deltas = product(gradients, hidden_T);
    matrixAdd(nn->output_layer->weights, weight_ho_deltas);
    deallocMatrix(weight_ho_deltas);
    deallocMatrix(hidden_T);
    deallocMatrix(gradients);

    Matrix *previous_errors;
    for (int i = nn->hidden-1; i >= 0; i--) {
        if (i == nn->hidden-1) {
            who_t = transpose(nn->output_layer->weights);
            hidden_errors = product(who_t, output_errors);
            deallocMatrix(output_errors);
        } else {
            who_t = transpose(nn->hidden_layers[i + 1].weights);
            hidden_errors = product(who_t, previous_errors);
            deallocMatrix(previous_errors);
        }
        deallocMatrix(who_t);
        hidden_gradient = dsigmoid(nn->hidden_layers[i].outputs);
        matrixMultiply(hidden_gradient, hidden_errors);
        if (i != 0)
            previous_errors = copyMatrix(hidden_errors);
        deallocMatrix(hidden_errors);
        multiply(hidden_gradient, nn->learning_rate);

        if (i == 0)
            inputs_T = transpose(inputs);
        else
            inputs_T = transpose(nn->hidden_layers[i-1].outputs);

        weight_ih_deltas = product(hidden_gradient, inputs_T);
        matrixAdd(nn->hidden_layers[i].weights, weight_ih_deltas);
        deallocMatrix(hidden_gradient);
        deallocMatrix(inputs_T);
        deallocMatrix(weight_ih_deltas);
    }

    // Dealloc rest of matricies
    deallocMatrix(inputs);
    deallocMatrix(nn->output_layer->outputs);
    for (int i = 0; i < nn->hidden; i++)
        deallocMatrix(nn->hidden_layers[i].outputs);

    return 0;
}
