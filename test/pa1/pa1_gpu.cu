// 3D Ultrasound beamforming baseline code for EECS 570
// Created by: Richard Sampson, Amlan Nayak, Thomas F. Wenisch
// Revision 1.0 - 11/15/16

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define TX_X 0
#define TX_Y 0
#define TX_Z -0.001
#define TRANS_X 32
#define TRANS_Y 32
#define RX_Z 0
#define DATA_LEN 12308 // Number for pre-processed data values per channel
#define IDX_CONST 0.000009625 // Speed of sound and sampling rate, converts dist to index
#define FILTER_DELAY 140		 // Constant added to index to account filter delay (off by 1 from MATLAB)

void __global__ transmit_distance(const float *point_x, const float *point_y, const float *point_z, float *dist_tx);
void __global__ reflected_distance(const float *point_x, const float *point_y, const float *point_z, const float *rx_x, const float *rx_y, const float *rx_data, const float *dist_tx, float *image);

int main(int argc, char **argv)
{

	int size = atoi(argv[1]);

	/* Variables for transducer geometry */

	float *rx_x;	// Receive transducer x position
	float *rx_y;	// Receive transducer y position

	int offset = 0;		  // Offset into rx_data
	float *rx_data;		  // Pointer to pre-processed receive channel data

	float tx_x = 0;		 // Transmit transducer x position
	float tx_y = 0;		 // Transmit transducer y position
	float tx_z = -0.001; // Transmit transducer z position

	/* Variables for image space points */
	int point; // Index into image space

	float *point_x; // Point x position
	float *point_y; // Point y position
	float *point_z; // Point z position

	int pts_r = 1560; // Radial points along scanline
	int sls_t = size; // Number of scanlines in theta
	int sls_p = size; // Number of scanlines in phi

	float *image_pos; // Pointer to current position in image
	float *image;	  // Pointer to full image (accumulated so far)

	/* Iterators */
	int it_rx; // Iterator for recieve transducer
	int it_r;  // Iterator for r
	int it_t;  // Iterator for theta
	int it_p;  // Iterator for phi

	/* Variables for distance calculation and index conversion */
	float x_comp; // Itermediate value for dist calc
	float y_comp; // Itermediate value for dist calc
	float z_comp; // Itermediate value for dist calc

	float *dist_tx;						 // Transmit distance (ie first leg only)
	float dist;							 // Full distance
	
	int index;							 // Index into transducer data

	FILE *input;
	FILE *output;

	/* Allocate space for data */
	rx_x = (float *)malloc(TRANS_X * TRANS_Y * sizeof(float));
	if (rx_x == NULL)
		fprintf(stderr, "Bad malloc on rx_x\n");
	rx_y = (float *)malloc(TRANS_X * TRANS_Y * sizeof(float));
	if (rx_y == NULL)
		fprintf(stderr, "Bad malloc on rx_y\n");
	rx_data = (float *)malloc(DATA_LEN * TRANS_X * TRANS_Y * sizeof(float));
	if (rx_data == NULL)
		fprintf(stderr, "Bad malloc on rx_data\n");

	point_x = (float *)malloc(pts_r * sls_t * sls_p * sizeof(float));
	if (point_x == NULL)
		fprintf(stderr, "Bad malloc on point_x\n");
	point_y = (float *)malloc(pts_r * sls_t * sls_p * sizeof(float));
	if (point_y == NULL)
		fprintf(stderr, "Bad malloc on point_y\n");
	point_z = (float *)malloc(pts_r * sls_t * sls_p * sizeof(float));
	if (point_z == NULL)
		fprintf(stderr, "Bad malloc on point_z\n");

	// dist_tx = (float *)malloc(pts_r * sls_t * sls_p * sizeof(float));
	// if (dist_tx == NULL)
	// 	fprintf(stderr, "Bad malloc on dist_tx\n");

	image = (float *)malloc(pts_r * sls_t * sls_p * sizeof(float));
	if (image == NULL)
		fprintf(stderr, "Bad malloc on image\n");
	memset(image, 0, pts_r * sls_t * sls_p * sizeof(float));

	/* validate command line parameter */
	if (argc < 1 || !(strcmp(argv[1], "16") || strcmp(argv[1], "32") || strcmp(argv[1], "64")))
	{
		printf("Usage: %s {16|32|64}\n", argv[0]);
		fflush(stdout);
		exit(-1);
	}

	char buff[128];
#ifdef __MIC__
	sprintf(buff, "/beamforming_input_%s.bin", argv[1]);
#else // !__MIC__
	sprintf(buff, "../../src/beamforming_input_%s.bin", argv[1]);
#endif

	input = fopen(buff, "rb");
	if (!input)
	{
		printf("Unable to open input file %s.\n", buff);
		fflush(stdout);
		exit(-1);
	}

	/* Load data from binary */
	fread(rx_x, sizeof(float), TRANS_X * TRANS_Y, input);
	fread(rx_y, sizeof(float), TRANS_X * TRANS_Y, input);

	fread(point_x, sizeof(float), pts_r * sls_t * sls_p, input);
	fread(point_y, sizeof(float), pts_r * sls_t * sls_p, input);
	fread(point_z, sizeof(float), pts_r * sls_t * sls_p, input);

	fread(rx_data, sizeof(float), DATA_LEN * TRANS_X * TRANS_Y, input);
	fclose(input);

	printf("Beginning computation\n");
	fflush(stdout);
	/* get start timestamp */
	// struct timeval tv;
	// gettimeofday(&tv, NULL);
	// uint64_t start = tv.tv_sec * (uint64_t)1000000 + tv.tv_usec;

	/* --------------------------- COMPUTATION ------------------------------ */

	const int data_num = pts_r * sls_t * sls_p;
	const int data_size = data_num * sizeof(float);
	const int block_size = 128;
	const int grid_size = data_num / block_size;

	// TODO: copy rx_x, rx_y, point_xyz into gpu mem space
	float *d_rx_x, *d_rx_y, *d_rx_data;
	float *d_point_x, *d_point_y, *d_point_z;
	float *d_dist_tx, *d_image;

	cudaMalloc((void **)&d_rx_x, TRANS_X * TRANS_Y * sizeof(float));
	cudaMalloc((void **)&d_rx_y, TRANS_X * TRANS_Y * sizeof(float));
	cudaMalloc((void **)&d_rx_data, DATA_LEN * TRANS_X * TRANS_Y * sizeof(float));
	cudaMalloc((void **)&d_point_x, data_size);
	cudaMalloc((void **)&d_point_y, data_size);
	cudaMalloc((void **)&d_point_z, data_size);
	cudaMalloc((void **)&d_dist_tx, data_size);
	cudaMalloc((void **)&d_image, 	data_size);

	cudaMemcpy(d_rx_x, rx_x, TRANS_X * TRANS_Y * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rx_y, rx_y, TRANS_X * TRANS_Y * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rx_data, rx_data, DATA_LEN * TRANS_X * TRANS_Y * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_point_x, point_x, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_point_y, point_y, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_point_z, point_z, data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_image, image, data_size, cudaMemcpyHostToDevice);

	transmit_distance<<<grid_size, block_size>>>(d_point_x, d_point_y, d_point_z, d_dist_tx);
	reflected_distance<<<grid_size, block_size>>>(d_point_x, d_point_y, d_point_z, d_rx_x, d_rx_y, d_rx_data, d_dist_tx, d_image);

	cudaMemcpy(image, d_image, data_size, cudaMemcpyDeviceToHost);

	/* --------------------------------------------------------------------- */

	/* get elapsed time */
	// gettimeofday(&tv, NULL);
	// uint64_t end = tv.tv_sec * (uint64_t)1000000 + tv.tv_usec;
	// uint64_t elapsed = end - start;

	// printf("@@@ Elapsed time (usec): %lld\n", elapsed);
	printf("Processing complete.  Preparing output.\n");
	fflush(stdout);

	

	/* Write result to file */
	char *out_filename;
#ifdef __MIC__
	out_filename = "/home/micuser/beamforming_output.bin";
#else // !__MIC__
	out_filename = "beamforming_output.bin";
#endif
	output = fopen(out_filename, "wb");
	fwrite(image, sizeof(float), pts_r * sls_t * sls_p, output);
	fclose(output);

	printf("Output complete.\n");
	fflush(stdout);

	/* Cleanup */
	free(rx_x);
	free(rx_y);
	free(rx_data);
	free(point_x);
	free(point_y);
	free(point_z);
	//free(dist_tx);
	free(image);

	cudaFree(d_rx_x);
	cudaFree(d_rx_y);
	cudaFree(d_rx_data);
	cudaFree(d_point_x);
	cudaFree(d_point_y);
	cudaFree(d_point_z);
	cudaFree(d_dist_tx);
	cudaFree(d_image);

	return 0;
}

void __global__ transmit_distance(const float *point_x, const float *point_y, const float *point_z, float *dist_tx)
{
	const int point = blockDim.x * blockIdx.x + threadIdx.x;

	float x_comp = TX_X - point_x[point];
	x_comp = x_comp * x_comp;
	float y_comp = TX_Y - point_y[point];
	y_comp = y_comp * y_comp;
	float z_comp = TX_Z - point_z[point];
	z_comp = z_comp * z_comp;

	dist_tx[point] = (float)sqrt(x_comp + y_comp + z_comp);
}

void __global__ reflected_distance(const float *point_x, const float *point_y, const float *point_z, const float *rx_x, const float *rx_y, const float *rx_data, const float *dist_tx, float *image)
{
	const int point = blockDim.x * blockIdx.x + threadIdx.x;

	float x_comp, y_comp, z_comp;
	int offset = 0;
	int index;
	for (int it_rx = 0; it_rx < TRANS_X * TRANS_Y; it_rx++)
	{

		x_comp = rx_x[it_rx] - point_x[point];
		x_comp = x_comp * x_comp;
		y_comp = rx_y[it_rx] - point_y[point];
		y_comp = y_comp * y_comp;
		z_comp = RX_Z - point_z[point];
		z_comp = z_comp * z_comp;

		float dist = dist_tx[point] + (float)sqrt(x_comp + y_comp + z_comp);
		index = (int)(dist / IDX_CONST + FILTER_DELAY + 0.5);
		image[point] += rx_data[index + offset];
		offset += DATA_LEN;
	}
}