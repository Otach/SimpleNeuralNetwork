/*
Compile with:
gcc xor.c -o build/xor -Ofast -fstack-protector-strong -Wall -lm
*/

#include "NNetwork.h"


int main(void) {
	int hidden_layers[4] = { 5, 4, 10, 9 };
	NeuralNetwork *nn = allocNN(2, 2, hidden_layers, 1, 0.1);
	float arr[][2] = {{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
	float results[][1] = {{0.0}, {1.0}, {1.0}, {0.0}};
	float *res;
	int r;

	printf("Before Training:\n");
	res = feedforward(nn, arr[0]);
	printf("\tShould be close to 0.0\n\t[1.0, 1.0]: %.4f\n\n", res[0]);
	free(res);
	res = feedforward(nn, arr[1]);
	printf("\tShould be close to 1.0\n\t[1.0, 0.0]: %.4f\n\n", res[0]);
	free(res);
	res = feedforward(nn, arr[2]);
	printf("\tShould be close to 1.0\n\t[0.0, 1.0]: %.4f\n\n", res[0]);
	free(res);
	res = feedforward(nn, arr[3]);
	printf("\tShould be close to 0.0\n\t[0.0, 0.0]: %.4f\n\n", res[0]);
	free(res);

	printf("Training...\n\n");
	srand(rdtsc());
	for (int i = 0; i < 100000; i++) {
		r = rand() % 4;
		train(nn, arr[r], results[r]);
	}

	printf("After Training:\n");
	res = feedforward(nn, arr[0]);
	printf("\tShould be close to 0.0\n\t[1.0, 1.0]: %.4f\n\n", res[0]);
	free(res);
	res = feedforward(nn, arr[1]);
	printf("\tShould be close to 1.0\n\t[1.0, 0.0]: %.4f\n\n", res[0]);
	free(res);
	res = feedforward(nn, arr[2]);
	printf("\tShould be close to 1.0\n\t[0.0, 1.0]: %.4f\n\n", res[0]);
	free(res);
	res = feedforward(nn, arr[3]);
	printf("\tShould be close to 0.0\n\t[0.0, 0.0]: %.4f\n\n", res[0]);
	free(res);
	// getchar();
	deallocNN(nn);
	return 0;
}
