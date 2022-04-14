// 3D Ultrasound beamforming baseline code for EECS 570 
// Created by: Richard Sampson, Amlan Nayak, Thomas F. Wenisch
// Revision 1.0 - 11/15/16

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>// for AVX512
#include <pthread.h>

// #define DEBUG
#define THREAD_NUM 200

// Global variables

/* Variables for transducer geometry */
const int trans_x = 32; // Transducers in x dim
const int trans_y = 32; // Transducers in y dim

float *point_x; // Point x position
float *point_y; // Point y position
float *point_z; // Point z position

float *rx_data; // Pointer to pre-processed receive channel data
float *rx_x; // Receive transducer x position
float *rx_y; // Receive transducer y position
float rx_z = 0; // Receive transducer z position

const int data_len = 12308; // Number for pre-processed data values per channel

int pts_r = 1560; // Radial points along scanline
int sls_t = 0; // Number of scanlines in theta
int sls_p = 0; // Number of scanlines in phi
int pts_all = 0; // = r*p*p

float *image_pos; // Pointer to current position in image
float *image;  // Pointer to full image (accumulated so far)
float *m256_debug_buf;
int *m256i_debug_buf;

float *dist_tx; // Transmit distance (ie first leg only)

const float idx_const = 0.000009625; // Speed of sound and sampling rate, converts dist to index
const int filter_delay = 140; // Constant added to index to account filter delay (off by 1 from MATLAB)
const int avx_num = 256;
const int avx_step = avx_num/32;
uint64_t mystart, myend;

typedef struct _pthread_args_t
{
	int id;
	int start;
	int end;
} pthread_args_t;

// uint64_t get_current_time()
// {
// 	struct timeval tv;
// 	gettimeofday(&tv, NULL);
// 	uint64_t result = tv.tv_sec * (uint64_t)1000000 + tv.tv_usec;
// 	return result;
// }

// __m256i _mm256_float_to_int32(__m256 x)
// {
// 	x = _mm256_add_ps(x, _mm256_set1_ps(8388608));
// 	return _mm256_sub_epi32(
// 		_mm256_castps_si256(x),
// 		_mm256_castps_si256(_mm256_set1_ps(8388608)));
// }

void print_m256_debug_buf()
{
	int i;
	printf("m256 Debug Buf: [ ");
	fflush(stdout);
	for (i = 0; i < avx_step; i++)
	{
		printf("%lf ", m256_debug_buf[i]);
		fflush(stdout);
	}
	printf("]\n\n");
	fflush(stdout);
}

void print_m256i_debug_buf()
{
	int i;
	printf("m256 Debug Buf: [ ");
	fflush(stdout);
	for (i = 0; i < avx_step; i++)
	{
		printf("%d ", m256i_debug_buf[i]);
		fflush(stdout);
	}
	printf("]\n");
	fflush(stdout);
}
void *reconst_image(void * args_in){
	pthread_args_t * args = (pthread_args_t *)args_in;
	int offset = 0;
	int point, it_rx;// local iterator
	// local temp value
	float dist;
	int index;
	float x_comp;	  // Itermediate value for dist calc
	float y_comp;	  // Itermediate value for dist calc
	float z_comp;	  // Itermediate value for dist calc
	int start = args->start;
	int end = args->end;
	int id = args->id;
#ifdef DEBUG
	myend = get_current_time();
	printf("Thread[%d] \t\t start at time: %lu Range[%d\t%d]\tLen=%d\n ",
		   id, myend - mystart, start, end, end - start);
	fflush(stdout);
#endif

	// Preparation for SIMD loop
	int presimd_len = ((start / avx_step + 1) * avx_step - start) % avx_step;
	int simd_start = start + presimd_len;
	int simd_len = ((end - simd_start) / avx_step) * avx_step;
	int postsimd_start = simd_start + simd_len;

	// Iterate from point [start] to [end]
	// pre-SIMD loop
	offset = 0;
	for (it_rx = 0; it_rx < trans_x * trans_y; it_rx++)
	{
		for (point = start; point < start + presimd_len; point++)
		{
			x_comp = rx_x[it_rx] - point_x[point];
			x_comp = x_comp * x_comp;
			y_comp = rx_y[it_rx] - point_y[point];
			y_comp = y_comp * y_comp;
			z_comp = rx_z - point_z[point];
			z_comp = z_comp * z_comp;

			dist = dist_tx[point] + (float)sqrt(x_comp + y_comp + z_comp);
			index = (int)(dist / idx_const + filter_delay + 0.5);
			image[point] += rx_data[index + offset];
		}
		offset += data_len;
	}

	// SIMD loop
	offset = 0;
	__m256 _rx_z = _mm256_set1_ps(rx_z);
	for (it_rx = 0; it_rx < trans_x * trans_y; it_rx++)
	{
		__m256 _rx_x = _mm256_set1_ps(rx_x[it_rx]);
		__m256 _rx_y = _mm256_set1_ps(rx_y[it_rx]);
		for (point = simd_start; point < simd_start + simd_len; point += avx_step)
		{
			// load data
			__m256 _x = _mm256_load_ps(point_x + point);
			__m256 _y = _mm256_load_ps(point_y + point);
			__m256 _z = _mm256_load_ps(point_z + point);
			// sub
			_x = _mm256_sub_ps(_rx_x, _x);
			_y = _mm256_sub_ps(_rx_y, _y);
			_z = _mm256_sub_ps(_rx_z, _z);
			// mult
			_x = _mm256_mul_ps(_x, _x);
			_y = _mm256_mul_ps(_y, _y);
			_z = _mm256_mul_ps(_z, _z);
			// add + sqrt
			_x = _mm256_add_ps(_x, _y);
			_x = _mm256_add_ps(_x, _z);
			_x = _mm256_sqrt_ps(_x);
			// dist = dist_rx + sqrt
			__m256 _dist_rx = _mm256_load_ps(dist_tx + point);
			_x = _mm256_add_ps(_x, _dist_rx); //_x holds dist
			// index = (int)(dist / idx_const + filter_delay + 0.5);
			__m256 _idx_const = _mm256_set1_ps(idx_const);
			__m256 _half_const = _mm256_set1_ps((float)0.5);
			__m256i _filter_delay = _mm256_set1_epi32(filter_delay);
			__m256i _offset = _mm256_set1_epi32(offset);

			_x = _mm256_div_ps(_x, _idx_const);
			// _x = _mm256_add_ps(_x, _half_const); // _x = dist / idx_const + 0.5

			__m256i _xi = _mm256_cvtps_epi32(_x);
			// __m256i _xi = _mm256_float_to_int32(_x);
			_xi = _mm256_add_epi32(_xi, _filter_delay);// _xi = index
			// // ! error occurs at this stage, int index has err of 1
			// // !debug here
			// if (id == 100)
			// 	_mm256_store_epi32(m256i_debug_buf, _xi);
			_xi = _mm256_add_epi32(_xi, _offset);// _xi holds index + offset

			__m256 _rx_data = _mm256_i32gather_ps(rx_data, _xi, 4);	// load data from rx_data
			__m256 _img = _mm256_load_ps(image + point);				// load image data
			_img = _mm256_add_ps(_img, _rx_data);						// add image and rx_data
			_mm256_store_ps(image + point, _img);						// store back to image

			// // ! use normal method to compute values and compare with values above
			// if(id == 100){
			// 	int point_debug = point;
			// 	float image_debug;
			// 	int flag = 0;
			// 	int cnt=0;
			// 	for (point_debug = point; point_debug < point + 16; point_debug++)
			// 	{
			// 		x_comp = rx_x[it_rx] - point_x[point_debug];
			// 		x_comp = x_comp * x_comp;
			// 		y_comp = rx_y[it_rx] - point_y[point_debug];
			// 		y_comp = y_comp * y_comp;
			// 		z_comp = rx_z - point_z[point_debug];
			// 		z_comp = z_comp * z_comp;
			// 		dist = dist_tx[point_debug] + (float)sqrt(x_comp + y_comp + z_comp);
			// 		index = (int)(dist / idx_const + filter_delay + 0.5);
			// 		image_debug = image[point_debug] + rx_data[index + offset];
			// 		if ( index != m256i_debug_buf[cnt]  || flag)
			// 		{
			// 			printf("%d ", index);
			// 			fflush(stdout);
			// 			flag = 1;
			// 		}
			// 		cnt++;
			// 	}
			// 	if(flag)
			// 		print_m256i_debug_buf();
			// }
		}
		offset += data_len;
	}
	// post-SIMD loop
	offset = 0;
	for (it_rx = 0; it_rx < trans_x * trans_y; it_rx++)
	{
		for (point = postsimd_start; point < end; point++)
		{
			x_comp = rx_x[it_rx] - point_x[point];
			x_comp = x_comp * x_comp;
			y_comp = rx_y[it_rx] - point_y[point];
			y_comp = y_comp * y_comp;
			z_comp = rx_z - point_z[point];
			z_comp = z_comp * z_comp;

			dist = dist_tx[point] + (float)sqrt(x_comp + y_comp + z_comp);
			index = (int)(dist / idx_const + filter_delay + 0.5);
			image[point] += rx_data[index + offset];
		}
		offset += data_len;
	}
	
#ifdef DEBUG
	myend = get_current_time();
	printf("Thread[%d] end at time: %lu \n", id, myend - mystart);
	fflush(stdout);
#endif
	pthread_exit(0);
	return NULL;
}

int main (int argc, char **argv) {

	const int size = atoi(argv[1]);
	
	const float tx_x = 0; // Transmit transducer x position
	const float tx_y = 0; // Transmit transducer y position
	const float tx_z = -0.001; // Transmit transducer z position
	// Create const vectors for vec operation
	__m256 _tx_x_vec = _mm256_setzero_ps();
	__m256 _tx_y_vec = _mm256_setzero_ps();
	__m256 _tx_z_vec = _mm256_set1_ps(tx_z);

	/* Variables for image space points */
	int point; // Index into image space 

	pts_r = 1560; // Radial points along scanline
	sls_t = size; // Number of scanlines in theta
	sls_p = size; // Number of scanlines in phi
	pts_all = pts_r * sls_t * sls_p;

	/* Iterators */
	int it_all;

	/* Variables for distance calculation and index conversion */
	float x_comp; // Itermediate value for dist calc
	float y_comp; // Itermediate value for dist calc
	float z_comp; // Itermediate value for dist calc

        FILE* input;
        FILE* output;

	/* Allocate space for data */
	posix_memalign((void **)(&rx_x), avx_num, trans_x * trans_y * sizeof(float));
		rx_x = (float *)rx_x;
		if (rx_x == NULL) fprintf(stderr, "Bad malloc on rx_x\n");
	
	posix_memalign((void **)(&rx_y), avx_num, trans_x * trans_y * sizeof(float));
		rx_y = (float *)rx_y;
		if (rx_y == NULL) fprintf(stderr, "Bad malloc on rx_y\n");

	posix_memalign((void **)(&rx_data), avx_num, data_len * trans_x * trans_y * sizeof(float));
		rx_data = (float *)rx_data;
		if (rx_data == NULL) fprintf(stderr, "Bad malloc on rx_data\n");

	posix_memalign((void **)(&point_x), avx_num, pts_all * sizeof(float));
		point_x = (float *)point_x;
		if (point_x == NULL) fprintf(stderr, "Bad malloc on point_x\n");
	posix_memalign((void **)(&point_y), avx_num, pts_all * sizeof(float));
		point_y = (float *)point_y;
		if (point_y == NULL) fprintf(stderr, "Bad malloc on point_y\n");
	posix_memalign((void **)(&point_z), avx_num, pts_all * sizeof(float));
		point_z = (float *)point_z;
		if (point_z == NULL) fprintf(stderr, "Bad malloc on point_z\n");

	posix_memalign((void **)(&dist_tx), avx_num, pts_all * sizeof(float));
		dist_tx = (float *)dist_tx;
		if (dist_tx == NULL) fprintf(stderr, "Bad malloc on dist_tx\n");

	posix_memalign((void **)(&image), avx_num, pts_all * sizeof(float));
		image = (float *) image;
		if (image == NULL) fprintf(stderr, "Bad malloc on image\n");
		memset(image, 0, pts_all * sizeof(float));// image init to be 0

	posix_memalign((void **)(&m256_debug_buf), avx_num, avx_step * sizeof(float));
		m256_debug_buf = (float *)m256_debug_buf;
		if (m256_debug_buf == NULL) fprintf(stderr, "Bad malloc on m256_debug_buf\n");

	posix_memalign((void **)(&m256i_debug_buf), avx_num, avx_step * sizeof(int));
		m256i_debug_buf = (int *)m256i_debug_buf;
		if (m256i_debug_buf == NULL) fprintf(stderr, "Bad malloc on m256i_debug_buf\n");

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
	  sprintf(buff, "../../src/beamforming_input_%s.bin", argv[1]);
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
 	// struct timeval tv;
    // 	gettimeofday(&tv,NULL);
    // 	uint64_t start = tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
 
	/* --------------------------- COMPUTATION Part 1------------------------------ */
	/* First compute transmit distance */
    // mystart = get_current_time();

	point = 0;
	int loop_simd = (pts_all / avx_step) * avx_step;
	for (it_all = 0; it_all < loop_simd; it_all += avx_step)
	{
		// load data
		__m256 _x_vec = _mm256_load_ps(point_x + point);
		__m256 _y_vec = _mm256_load_ps(point_y + point);
		__m256 _z_vec = _mm256_load_ps(point_z + point);
		// sub
		_x_vec = _mm256_sub_ps(_tx_x_vec, _x_vec);
		_y_vec = _mm256_sub_ps(_tx_y_vec, _y_vec);
		_z_vec = _mm256_sub_ps(_tx_z_vec, _z_vec);
		// mul
		_x_vec = _mm256_mul_ps(_x_vec, _x_vec);
		_y_vec = _mm256_mul_ps(_y_vec, _y_vec);
		_z_vec = _mm256_mul_ps(_z_vec, _z_vec);
		// add and sqrt
		_x_vec = _mm256_add_ps(_x_vec, _y_vec);
		_x_vec = _mm256_add_ps(_x_vec, _z_vec);
		_x_vec = _mm256_sqrt_ps(_x_vec);
		// store to dist_tx[point: point+15]
		_mm256_store_ps(dist_tx + point, _x_vec);
		// This iteration should store val into
		point += avx_step;
	}

	for (it_all = loop_simd; it_all < pts_all; it_all++)
	{
		x_comp = tx_x - point_x[point];
		x_comp = x_comp * x_comp;
		y_comp = tx_y - point_y[point];
		y_comp = y_comp * y_comp;
		z_comp = tx_z - point_z[point];
		z_comp = z_comp * z_comp;

		dist_tx[point++] = (float)sqrt(x_comp + y_comp + z_comp);
	}

	/* --------------------------- COMPUTATION Part 2------------------------------ */
	/* Now compute reflected distance, find index values, add to image */

	// myend = get_current_time();
	// printf("Transmit distance time: %lu\n", myend - mystart);
	// mystart = get_current_time();


	pthread_t thr[THREAD_NUM];
	pthread_args_t thr_args[THREAD_NUM];
	int avg_work_len = pts_all/THREAD_NUM;
	// should only tune add work
	int addwork = 4*THREAD_NUM;

	int cooldown = addwork * 2 / THREAD_NUM;
	int work_start = 0;
	int work_len = avg_work_len + addwork;
	int it_thr;
	for (it_thr = 0; it_thr < THREAD_NUM; it_thr++)
	{
		thr_args[it_thr].id    = it_thr;
		thr_args[it_thr].start = work_start;
		thr_args[it_thr].end   = work_start + work_len;
		work_start += work_len;
		work_len   -= cooldown;
	}
	thr_args[THREAD_NUM - 1].end = pts_all;

	for (it_thr = 0; it_thr < THREAD_NUM; it_thr++)
		pthread_create(&thr[it_thr], NULL, reconst_image, &thr_args[it_thr]);

	for (it_thr = 0; it_thr < THREAD_NUM; it_thr++)
		pthread_join(thr[it_thr], NULL);

	/* --------------------------------------------------------------------- */

	/* get elapsed time */
    // 	gettimeofday(&tv,NULL);
    // 	uint64_t end = tv.tv_sec*(uint64_t)1000000+tv.tv_usec;
    // 	uint64_t elapsed = end - start;

	// printf("@@@ Elapsed time (usec): %ld\n", elapsed);
	// printf("Processing complete.  Preparing output.\n");
	// fflush(stdout);

	/* Write result to file */
	const char* out_filename;
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
