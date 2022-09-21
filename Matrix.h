#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

typedef struct Matrix {
	int rows;
	int cols;
	int datasize;
	int size;
	float *data;
} Matrix;

// A sufficient seed for the random function
unsigned long long rdtsc() {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long) hi << 32) | lo;
}

// Remove the matrix data from the heap
int deallocMatrix(Matrix *a){
    free(a->data);
    a->data = NULL;
    free(a);
    a = NULL;
    return 0;
}

// Alloc space in heap for matrix data
Matrix *allocMatrix(int32_t rows, int32_t cols){

    Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->size = matrix->rows * matrix->cols;
    matrix->datasize =  matrix->size * sizeof(float);
    matrix->data = (float *)calloc(matrix->size, sizeof(float));
    if (matrix->data == NULL) {
        return NULL;
    }
    return matrix;
}

Matrix *copyMatrix(Matrix *a) {
	Matrix *b = allocMatrix(a->rows, a->cols);
	memcpy(b->data, a->data, a->datasize);
	return b;
}

void printMatrix(Matrix *m) {
    for (int i = 0; i < m->size; i++) {
        if (i % m->cols == 0 && i != 0) {
            printf("\n");
        }
        printf("%.4f ", m->data[i]);
    }
    printf("\n");
}

int sigmoid(Matrix *result) {
    // The activation function that will always give a
    // number between 0 and 1
    for(int i = 0; i < result->size; i++) {
        result->data[i] = 1 / (1 + exp(-result->data[i]));
    }
    return 0;
}

Matrix *dsigmoid(Matrix *a){
    Matrix *result = allocMatrix(a->rows, a->cols);;

    //y * (1-y);
    for (int i = 0; i < result->size; i++){
        result->data[i] = a->data[i] * (1-a->data[i]);
    }
    return result;
}


float randomfloat(void){
    float randomint = -1.0;
    while(randomint == -1.0){
        randomint = ((float)(rand() % 1000) / (float)500) - 1.0;
    }
    return randomint;
}

void randomizeMatrix(Matrix *matrix){
    //randomize the values of the provided matrix
    srand(0);
    for(int i = 0; i < matrix->size; i++) {
        matrix->data[i] = randomfloat();
        }
    return;
}

// int add(struct Matrix *matrix, float a){
//  // Add 'a' to every element in the matrix
//  for(int i = 0; i < matrix->rows; i++){
//      for(int j = 0; j < matrix->cols; j++){
//          matrix->data[i][j] = matrix->data[i][j] + a;
//      }
//  }
//  return 0;
// }

int multiply(Matrix *matrix, float a){
    // Multiply 'a' to every element in the matrix
    for(int i = 0; i < matrix->size; i++){
        matrix->data[i] = matrix->data[i] * a;
    }
    return 0;
}

Matrix *product(Matrix *a, Matrix *b){
    // Perform a matrix product on a and b and store in result
    float sum;
    if (a->cols != b->rows){
        printf("Could not perform matrix product\n");
        printf("Product Col v. Row Check\n");
        return NULL;
    }
    Matrix *result = allocMatrix(a->rows, b->cols);

    // For every row
    for(int i = 0; i < a->rows; i++){
        // For every column
        for(int j = 0; j < b->cols; j++){
        	sum = 0;
        	// For every element in the row-col pair
            for(int k = 0; k < a->cols; k++){
                 sum += a->data[(i * a->cols) + k] * b->data[(k * b->cols) + j];
            }
        	result->data[(i * result->cols) + j] = sum;
        }
    }
    return result;
}

void fromArray(Matrix *matrix, float *arr) {
    memcpy(matrix->data, arr, matrix->datasize);
}

float *toArray(Matrix *matrix) {
    float *arr = (float *)malloc(matrix->datasize);
    memcpy(arr, matrix->data, matrix->datasize);
    return arr;
}

Matrix *transpose(Matrix *input){
    Matrix *t = allocMatrix(input->cols, input->rows);

    for (int i = 0; i < t->rows; i++){
        for(int j = 0; j < t->cols; j++){
            t->data[(i * t->cols) + j] = input->data[(j * input->cols) + i];
        }
    }
    return t;
}

int matrixAdd(Matrix *a, Matrix *b){
    if(a->rows != b->rows || a->cols != b->cols){
        printf("matrixAdd: matricies are not same dimensions\n");
        return -1;
    }

    for(int i = 0; i < a->size; i++){
        a->data[i] = a->data[i] + b->data[i];
    }

    return 0;
}

int matrixMultiply(Matrix *a, Matrix *b){
    if(a->rows != b->rows || a->cols != b->cols){
        printf("matrixAdd: matricies are not same dimensions\n");
        return -1;
    }

    for(int i = 0; i < a->size; i++){
        a->data[i] = a->data[i] * b->data[i];
    }

    return 0;
}

Matrix *matrixSubtract(Matrix *a, Matrix *b){
    Matrix *result;

    if(a->rows != b->rows || a->cols != b->cols){
        printf("matrixSubtract: matricies are not same dimensions\n");
        return NULL;
    }

    result = allocMatrix(a->rows, a->cols);

    for(int i = 0; i < result->size; i++) {
            result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

int inputFromArray(float *arr, int arr_length, Matrix *result) {

    if (arr_length != result->size) {
        return -1;
    }

    memcpy(result->data, arr, result->datasize);

    return 0;
}
