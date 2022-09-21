/*
Compile with:
gcc run.c -o build/run -O3 -fstack-protector-strong -Wall -lm
*/

#include <stdint.h>
#include <time.h>
#include "NNetwork.h"

#define TRAIN_IMG_FILE "Datasets/mnist/train-images-idx3-ubyte"
#define TRAIN_LBL_FILE "Datasets/mnist/train-labels-idx1-ubyte"
#define TEST_IMG_FILE "Datasets/mnist/t10k-images-idx3-ubyte"
#define TEST_LBL_FILE "Datasets/mnist/t10k-labels-idx1-ubyte"

#define TO_INT(bytes) (bytes[0] << 24) + (bytes[1] << 16) + (bytes[2] << 8) + bytes[3]

typedef struct Image {
	int32_t rows;
	int32_t cols;
	int8_t label;
	float *img_data;
} Image;

typedef struct DataSet {
	int32_t length;
	Image **data;
} DataSet;

typedef struct LabelSet {
	int32_t num;
	int8_t *labels;
} LabelSet;

typedef struct ImageSet {
	int32_t num;
	int32_t rows;
	int32_t cols;
	int32_t data_size;
	float *data;
} ImageSet;

LabelSet *load_labels(char *filename) {
	LabelSet *labels;
	FILE *label_file;
	unsigned char tmpbuff[4];
	int32_t magic, num;

	memset(tmpbuff, '\0', 4);
	label_file = fopen(filename, "rb");

	// read magic bytes (2049)
	fread(tmpbuff, 4, 1, label_file);
	magic = TO_INT(tmpbuff);
	if (magic != 2049) {
		return NULL;
	}

	// read number of items
	memset(tmpbuff, '\0', 4);
	fread(tmpbuff, 4, 1, label_file);
	num = TO_INT(tmpbuff);

	// load the label data
	labels = malloc(sizeof(LabelSet));
	labels->num = num;
	labels->labels = malloc(sizeof(int8_t) * num);

	fread(labels->labels, 1, labels->num, label_file);

	fclose(label_file);
	return labels;
}

ImageSet *load_images(char *filename) {
	ImageSet *images;
	FILE *image_file;
	unsigned char tmpbuff[4];
	unsigned char *databuff;
	int32_t magic;

	images = malloc(sizeof(ImageSet));
	image_file = fopen(filename, "rb");

	// read magic bytes (2051)
	fread(tmpbuff, 4, 1, image_file);
	magic = TO_INT(tmpbuff);
	if (magic != 2051) {
		return NULL;
	}

	// read number of images
	fread(tmpbuff, 4, 1, image_file);
	images->num = TO_INT(tmpbuff);

	// read number of rows
	fread(tmpbuff, 4, 1, image_file);
	images->rows = TO_INT(tmpbuff);

	// read number of cols
	fread(tmpbuff, 4, 1, image_file);
	images->cols = TO_INT(tmpbuff);

	images->data_size = images->rows * images->cols * images->num;

	// Read the data into a temporary buffer
	databuff = malloc(sizeof(unsigned char) * images->data_size);
	fread(databuff, images->data_size, 1, image_file);

	// convert the bytes to floats (pre-processing)
	images->data = malloc(sizeof(float) * images->data_size);
	for (int i = 0; i < images->data_size; i++) {
		images->data[i] = databuff[i]/256.0;
	}

	free(databuff);
	databuff = NULL;
	fclose(image_file);
	return images;
}

void free_labels(LabelSet *labels) {
	free(labels->labels);
	labels->labels = NULL;
	free(labels);
	labels = NULL;
	return;
}

void free_images(ImageSet *images) {
	free(images->data);
	images->data = NULL;
	free(images);
	images = NULL;
	return;
}

void free_dataset(DataSet *ds) {
	for (int i = 0; i < ds->length; i++) {
		free(ds->data[i]->img_data);
		ds->data[i]->img_data = NULL;
		free(ds->data[i]);
		ds->data[i] = NULL;
	}
	free(ds->data);
	free(ds);
}

DataSet *load_dataset(char *img_filename, char *lbl_filename) {
	DataSet *data = malloc(sizeof(DataSet));
	ImageSet *images = load_images(img_filename);
	LabelSet *labels = load_labels(lbl_filename);

	if (images->num != labels->num) {
		free_labels(labels);
		free_images(images);
		return NULL;
	}
	data->length = images->num;
	data->data = malloc(sizeof(Image *) * data->length);

	for (int i = 0; i < data->length; i++) {
		data->data[i] = malloc(sizeof(Image));
		data->data[i]->rows = images->rows;
		data->data[i]->cols = images->cols;
		data->data[i]->label = labels->labels[i];
		data->data[i]->img_data = malloc(sizeof(float) * images->rows * images->cols);
		memcpy(data->data[i]->img_data, &images->data[i * (images->rows * images->cols)], (sizeof(float) * images->rows * images->cols));
	}

	free_labels(labels);
	free_images(images);
	return data;
}

void print_image_data(Image *img) {
	printf("Image Rows: %d\n", img->rows);
	printf("Image Cols: %d\n", img->cols);
	printf("Image Label: %d\n", img->label);
	for (int i = 0; i < img->rows; i++) {
		for (int j = 0; j < img->cols; j++) {
			if (img->img_data[(i * img->rows) + j] != 0.0) {
				printf("# ");
			} else {
				printf(". ");
			}
		}
		printf("\n");
	}
	return;
}

float *create_expected_return(int label) {
	float *arr = calloc(10, sizeof(float));
	arr[label] = 1.0;
	return arr;
}

int get_prediction_value(float *arr) {
	int max = 0;
	float max_value = 0.0;
	for (int i = 0; i < 10; i++) {
		if (arr[i] > max_value) {
			max_value = arr[i];
			max = i;
		}
	}
	return max;
}

void test_nn(NeuralNetwork *nn, DataSet *ds) {
	int total_correct = 0, prediction;
	float *res;

	for (int i = 0; i < ds->length; i++) {
		res = feedforward(nn, ds->data[i]->img_data);
		prediction = get_prediction_value(res);
		if (prediction == ds->data[i]->label) total_correct += 1;
		free(res);
	}

	printf("Accuracy: %.2f\n", (float)total_correct/ds->length * 100.0);
	return;
}

int main(void) {
	DataSet *training_data, *testing_data;
	NeuralNetwork *nn;
	float *expected_result;
	clock_t tick, tock;

	printf("Loading training data\n");
	training_data = load_dataset(TRAIN_IMG_FILE, TRAIN_LBL_FILE);
	printf("Loading testing data\n");
	testing_data = load_dataset(TEST_IMG_FILE, TEST_LBL_FILE);
	tick = clock();
	int hidden_layers[2] = { 16, 16 };
	nn = allocNN(784, 2, hidden_layers, 10, 0.1);

	printf("Testing before Training...\n");
	test_nn(nn, testing_data);

	printf("Training...\n");
	for (int j = 0; j < 10; j++) {
		printf("Epoch %d\n", j);
		for (int i = 0; i < training_data->length; i++) {
			expected_result = create_expected_return(training_data->data[i]->label);
			train(nn, training_data->data[i]->img_data, expected_result);
			free(expected_result);
		}
	}

	printf("Testing after Training...\n");
	test_nn(nn, testing_data);

	free_dataset(training_data);
	free_dataset(testing_data);
	deallocNN(nn);
	tock = clock();
	printf("Took %fs\n", (double)(tock-tick) / CLOCKS_PER_SEC);
	return 0;
}
