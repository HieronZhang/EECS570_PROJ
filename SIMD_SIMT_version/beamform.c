// 3D Ultrasound beamforming baseline code for EECS 570 
// Created by: Richard Sampson, Amlan Nayak, Thomas F. Wenisch
// Revision 1.0 - 11/15/16

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include <immintrin.h>

#define NUM_THREADS 260

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

struct thread_args{
	int start;
	int end;
};


__m512i float_to_int32(__m512 x){
    x = _mm512_add_round_ps(x, _mm512_set1_ps(8388608), 3);
    return _mm512_sub_epi32(
        _mm512_castps_si512(x),
        _mm512_castps_si512(_mm512_set1_ps(8388608))
    );
}

/* Variables for transducer geometry */
	int trans_x = 32; // Transducers in x dim
	int trans_y = 32; // Transducers in y dim
	
	float *rx_x; // Receive transducer x position
	float *rx_y; // Receive transducer y position
	float rx_z = 0; // Receive transducer z position

	int data_len = 12308; // Number for pre-processed data values per channel
	int offset = 0; // Offset into rx_data
	float *rx_data; // Pointer to pre-processed receive channel data

	float tx_x = 0; // Transmit transducer x position
	float tx_y = 0; // Transmit transducer y position
	float tx_z = -0.001; // Transmit transducer z position

	/* Variables for image space points */
	int point; // Index into image space

	float *point_x; // Point x position
	float *point_y; // Point y position
	float *point_z; // Point z position

	int pts_r = 1560; // Radial points along scanline
	int sls_t; // Number of scanlines in theta
	int sls_p; // Number of scanlines in phi

	float *image_pos; // Pointer to current position in image
	float *image;  // Pointer to full image (accumulated so far)

	/* Iterators */
	int it_rx; // Iterator for recieve transducer
	int it_r; // Iterator for r
	int it_t; // Iterator for theta
	int it_p; // Iterator for phi

	/* Variables for distance calculation and index conversion */
	float x_comp; // Itermediate value for dist calc
	float y_comp; // Itermediate value for dist calc
	float z_comp; // Itermediate value for dist calc

	float *dist_tx; // Transmit distance (ie first leg only)
	float dist; // Full distance
	const float idx_const = 0.000009625; // Speed of sound and sampling rate, converts dist to index
	const int filter_delay = 140; // Constant added to index to account filter delay (off by 1 from MATLAB)



void *step2( void *arg){

	/* Local variables */
	
	int it_rx; // Iterator for recieve transducer
	int it_r; // Iterator for r
	int it_t; // Iterator for theta
	int it_p; // Iterator for phi
	float *image_pos; // Pointer to current position in image
	int point; // Index into image space
	float x_comp; // Itermediate value for dist calc
	float y_comp; // Itermediate value for dist calc
	float z_comp; // Itermediate value for dist calc
	float dist; // Full distance
	int index; // Index into transducer data
	int offset = 0; // Offset into rx_data



	/* Input variables */
	
	struct thread_args * input = (struct thread_args *) arg;

	/* Now compute reflected distance, find index values, add to image */
	for (it_rx = 0; it_rx < 1024; it_rx++) {

		image_pos = image; // Reset image pointer back to beginning
		point = 0; // Reset 
		float ori_x = rx_x[it_rx];
		float ori_y = rx_y[it_rx];


		// Iterate over entire image space
	
		for (it_r = input->start; it_r < input->end; it_r += 16) {
			point = it_r;
			
			__m512 Vec0 = _mm512_setzero_ps ();
			__m512 VecX = _mm512_set1_ps(ori_x);

			
			__m512 VecPX = _mm512_load_ps (point_x+point);


			VecX = _mm512_sub_ps(VecX, VecPX);
			VecX = _mm512_mul_ps(VecX, VecX);

			__m512 VecY = _mm512_set1_ps(ori_y);
			__m512 VecPY = _mm512_load_ps (point_y+point);
			VecY = _mm512_sub_ps(VecY, VecPY);
			VecY = _mm512_mul_ps(VecY, VecY);

			__m512 VecZ = _mm512_load_ps(point_z+point);
			VecZ = _mm512_mul_ps(VecZ, VecZ);

			/*
			x_comp = rx_x[it_rx] - point_x[point];  //let comp value be totally local
			x_comp = x_comp * x_comp;
			y_comp = rx_y[it_rx] - point_y[point];  //point need to be calculated
			y_comp = y_comp * y_comp;
			z_comp = rx_z - point_z[point];
			z_comp = z_comp * z_comp;
			*/
			
			VecX = _mm512_add_ps(VecX, VecY);
			VecX = _mm512_add_ps(VecX, VecZ);
			VecX = _mm512_sqrt_ps(VecX);
			VecY = _mm512_load_ps(dist_tx+point);
			VecX = _mm512_add_ps(VecX, VecY);   //dist

			VecY = _mm512_set1_ps((float)filter_delay);
			VecZ = _mm512_set1_ps(0.5);
			VecPX = _mm512_set1_ps(idx_const);
			VecX = _mm512_div_ps(VecX, VecPX);
			VecX = _mm512_add_ps(VecX, VecY);
			VecX = _mm512_add_ps(VecX, VecZ);



			__m512i VecI;
			VecI = float_to_int32 (VecX); //index   
			

			/*
			dist = dist_tx[point] + (float)sqrt(x_comp + y_comp + z_comp);   // dist should be local
			index = (int)(dist/idx_const + filter_delay + 0.5);  //index local
			*/
			


			VecY = _mm512_i32gather_ps (VecI, rx_data+offset, 4);  //KNC has it

			VecX = _mm512_load_ps(image_pos+point);


			VecX = _mm512_add_ps(VecX, VecY);

			_mm512_store_ps (image_pos+point, VecX);

			/*
			*(image_pos+point) += rx_data[index+offset];     //offset local
			*/

		}
		offset += data_len;
	}
}


int main (int argc, char **argv) {

	int size = atoi(argv[1]);

	sls_t = size; // Number of scanlines in theta
	sls_p = size; // Number of scanlines in phi

        FILE* input;
        FILE* output;

	/* Allocate space for data */
	rx_x = (float*) memalign(512, trans_x * trans_y * sizeof(float));
	if (rx_x == NULL) fprintf(stderr, "Bad malloc on rx_x\n");
	rx_y = (float*) memalign(512, trans_x * trans_y * sizeof(float));
	if (rx_y == NULL) fprintf(stderr, "Bad malloc on rx_y\n");
	rx_data = (float*) memalign(512, data_len * trans_x * trans_y * sizeof(float));
	if (rx_data == NULL) fprintf(stderr, "Bad malloc on rx_data\n");

	point_x = (float *) memalign(512, pts_r * sls_t * sls_p * sizeof(float));
	if (point_x == NULL) fprintf(stderr, "Bad malloc on point_x\n");
	point_y = (float *) memalign(512, pts_r * sls_t * sls_p * sizeof(float));
	if (point_y == NULL) fprintf(stderr, "Bad malloc on point_y\n");
	point_z = (float *) memalign(512, pts_r * sls_t * sls_p * sizeof(float));
	if (point_z == NULL) fprintf(stderr, "Bad malloc on point_z\n");

	dist_tx = (float*) memalign(512, pts_r * sls_t * sls_p * sizeof(float));
	if (dist_tx == NULL) fprintf(stderr, "Bad malloc on dist_tx\n");

	image = (float *) memalign(512, pts_r * sls_t * sls_p * sizeof(float));
	if (image == NULL) fprintf(stderr, "Bad malloc on image\n");
	memset(image, 0, pts_r * sls_t * sls_p * sizeof(float));

	/* validate command line parameter */
	if (argc < 1 || !(strcmp(argv[1],"16") || strcmp(argv[1],"32") || strcmp(argv[1],"64"))) {
	  printf("Usage: %s {16|32|64}\n",argv[0]);
	  fflush(stdout);
	  exit(-1);
	}

	char buff[128];
        #ifdef __MIC__
	  sprintf(buff, "/beamforming_input_%s.bin", argv[1]);
        #else // !__MIC__
	  sprintf(buff, "/n/typhon/data1/home/eecs570/beamforming_input_%s.bin", argv[1]);
        #endif

        input = fopen(buff,"rb");
	if (!input) {
	  printf("Unable to open input file %s.\n", buff);
	  fflush(stdout);
	  exit(-1);
	}	

	/* Load data from binary */
	fread(rx_x, sizeof(float), trans_x * trans_y, input); 
	fread(rx_y, sizeof(float), trans_x * trans_y, input); 

	fread(point_x, sizeof(float), pts_r * sls_t * sls_p, input); 
	fread(point_y, sizeof(float), pts_r * sls_t * sls_p, input); 
	fread(point_z, sizeof(float), pts_r * sls_t * sls_p, input); 

	fread(rx_data, sizeof(float), data_len * trans_x * trans_y, input); 
        fclose(input);

	printf("Beginning computation\n");
	fflush(stdout);

	/* get start timestamp */
 	struct timeval tv;
    	gettimeofday(&tv,NULL);
    	uint64_t start = tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
 
	/* --------------------------- COMPUTATION ------------------------------ */
	/* First compute transmit distance */
	point = 0;
	for (it_t = 0; it_t < sls_t; it_t++) {

		for (it_p = 0; it_p < sls_p; it_p++) {
			for (it_r = 0; it_r < pts_r; it_r++) {

				x_comp = tx_x - point_x[point];
				x_comp = x_comp * x_comp;
				y_comp = tx_y - point_y[point];
				y_comp = y_comp * y_comp;
				z_comp = tx_z - point_z[point];
				z_comp = z_comp * z_comp;

				dist_tx[point++] = (float)sqrt(x_comp + y_comp + z_comp);
			}
		}
	}
	/* get step time */
		gettimeofday(&tv,NULL);
    	uint64_t interm = tv.tv_sec*(uint64_t)1000000+tv.tv_usec;

	/* Now compute reflected distance, find index values, add to image */
	pthread_t child_threads[NUM_THREADS];
	struct thread_args input_ranges[NUM_THREADS];
	int range = 1560*sls_t*sls_p/NUM_THREADS;
	if(1560*sls_t*sls_p%NUM_THREADS) range++;
	int current_start = 0;
	int i;
	for (i = 0; i < NUM_THREADS; i++)
	{
		/*
		input_ranges[i].dist_tx = dist_tx;
		input_ranges[i].image = image;
		input_ranges[i].point_x = point_x;
		input_ranges[i].point_y = point_y;
		input_ranges[i].point_z = point_z;
		input_ranges[i].rx_data = rx_data;
		input_ranges[i].rx_x = rx_x;
		input_ranges[i].rx_y = rx_y;
		input_ranges[i].sls_p = sls_p;
		input_ranges[i].sls_t = sls_t;
		*/
		
		input_ranges[i].start = current_start;
		input_ranges[i].end = current_start + range;
		current_start += range;
	}
	
	
	for (i = 0; i < NUM_THREADS; i++)
	{
		pthread_create(&child_threads[i], NULL, step2, &input_ranges[i]);
	}
	
	for (i = 0; i < NUM_THREADS; i++)
	{
		pthread_join(child_threads[i], NULL);
	}
	



	/* --------------------------------------------------------------------- */

	/* get elapsed time */
    	gettimeofday(&tv,NULL);
    	uint64_t end = tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
    	uint64_t elapsed = end - start;
		uint64_t first = interm - start;
		uint64_t second = end - interm;

	printf("@@@ Elapsed time (usec): %lld\n", elapsed);
	printf("@@@ Elapsed first step time (usec): %lld\n", first);
	printf("@@@ Elapsed second step time (usec): %lld\n", second);
	printf("Processing complete.  Preparing output.\n");
	printf("I have %ld logical cores.\n", sysconf(_SC_NPROCESSORS_ONLN));
	fflush(stdout);

	/* Write result to file */
	char* out_filename;
        #ifdef __MIC__
	  out_filename = "/home/micuser/beamforming_output.bin";
        #else // !__MIC__
	  out_filename = "beamforming_output.bin";
        #endif
        output = fopen(out_filename,"wb");
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
	free(dist_tx);
	free(image);

	return 0;
}
