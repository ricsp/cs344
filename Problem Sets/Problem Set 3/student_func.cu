/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "stdio.h"
#include <float.h>

#define MAX(A,B) (A>B ? A : B)
#define MIN(A,B) (A<B ? A : B)

__global__
void reduce_min(float * d_out, const float * d_in, const unsigned int size){

  extern __shared__ float smin[];

  const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int tid = threadIdx.x;

  if (id<size){
	  smin[tid] = d_in[id];
  } else {
	  smin[tid] = FLT_MAX;
  }
  __syncthreads();

  for (unsigned int s = blockDim.x/2 ; s>0; s>>=1){
    if (tid < s){
    	smin[tid] = MIN(smin[tid], smin[tid+s]);
    }
    __syncthreads();
  }

  if (tid == 0){
      d_out[blockIdx.x] = smin[tid];
  }

}

__global__
void reduce_max(float * d_out, const float * d_in, const unsigned int size){

  extern __shared__ float smin[];

  const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int tid = threadIdx.x;

  if (id<size){
	  smin[tid] = d_in[id];
  } else {
	  smin[tid] = FLT_MIN;
  }
  __syncthreads();

  for (unsigned int s = blockDim.x>>1; s>0; s>>=1){
    if (tid < s){
    	smin[tid] = MAX(smin[tid], smin[tid+s]);
    }
    __syncthreads();
  }

  if (tid == 0){
      d_out[blockIdx.x] = smin[tid];
  }

}


__global__
void histogram(unsigned int * d_histogram, const float * d_logLuminance,
		const float min_logLum, const float range_logLum,
		const unsigned int numBins,
		const unsigned int size){

  const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id<size){
	  const unsigned int bin = MIN(numBins-1, (d_logLuminance[id] - min_logLum) / range_logLum * numBins);
	  atomicAdd(&d_histogram[bin],1);
  }

}

__global__
void hillis(unsigned int *g_odata, size_t n)  {
	extern __shared__ unsigned int data[];  // allocated on invocation

	int thid = threadIdx.x;
	unsigned int *g_idata = g_odata;

	// load input into shared memory
	data[2*thid] = g_idata[2*thid];
	data[2*thid+1] = g_idata[2*thid+1];

	// pre-scan
	unsigned int dp = 1;
	for (unsigned int s = n>>1; s>0; s>>=1){
		__syncthreads();
		if(thid<s){
			unsigned int i = dp*(2*thid+1)-1;
			unsigned int j = dp*(2*thid+2)-1;
			data[j] += data[i];
		}
		dp <<= 1;
	}

	if(thid == 0) data[n - 1] = 0;

	// down-sweep
	for(unsigned int s = 1; s < n; s <<= 1) {
		dp >>= 1;
		__syncthreads();

		if(thid < s) {
			unsigned int i = dp*(2*thid+1)-1;
			unsigned int j = dp*(2*thid+2)-1;

			const unsigned int t = data[j];
			data[j] += data[i];
			data[i] = t;
		}
	}

	// write results to device memory
	g_odata[2*thid] = data[2*thid];
	g_odata[2*thid+1] = data[2*thid+1];

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
//Compute correct grid size (i.e., number of blocks per kernel launch)
//from the image size and and block size.

  const unsigned int numPixel = numCols*numRows;
  unsigned int numThreads = 512;
  unsigned int numBlocks = (numPixel+numThreads-1)/numThreads;

//    1) find the minimum and maximum value in the input logLuminance channel
//       store in min_logLum and max_logLum

  float * d_min, * d_max, *d_intermediate;
  checkCudaErrors(cudaMalloc(&d_min, sizeof(float) * numBlocks));
  checkCudaErrors(cudaMalloc(&d_max, sizeof(float) * numBlocks));
  checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float) * numBlocks));

  reduce_min<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(d_intermediate, d_logLuminance, numPixel);
  reduce_min<<<1, numBlocks, numBlocks * sizeof(float)>>>(d_min, d_intermediate, numBlocks);
  reduce_max<<<numBlocks, numThreads, numThreads * sizeof(float)>>>(d_intermediate, d_logLuminance, numPixel);
  reduce_max<<<1, numBlocks, numBlocks * sizeof(float)>>>(d_max, d_intermediate, numBlocks);

  cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_intermediate);
  cudaFree(d_min);
  cudaFree(d_max);

//  2) subtract them to find the range
  const float range_logLum = max_logLum - min_logLum;
  printf("gpu min: %f\ngpu max: %f\ngpu range: %f\n", min_logLum, max_logLum, range_logLum);

  //    3) generate a histogram of all the values in the logLuminance channel using
  //       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  unsigned int *h_histo = new unsigned int[numBins];
  checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int) * numBins));

  histogram<<<numBlocks, numThreads>>>(d_cdf, d_logLuminance, min_logLum, range_logLum, numBins, numPixel);
  checkCudaErrors(cudaMemcpy(h_histo, d_cdf, sizeof(unsigned int)*numBins, cudaMemcpyDeviceToHost));

  //    4) Perform an exclusive scan (prefix sum) on the histogram to get
  //       the cumulative distribution of luminance values (this should go in the
  //       incoming d_cdf pointer which already has been allocated for you)
  numThreads = numBins/2;
  numBlocks = 1;
  hillis<<<numBlocks, numThreads, 2*numThreads*sizeof(unsigned int)>>>(d_cdf, 2*numThreads);

  checkCudaErrors(cudaMemcpy(h_histo, d_cdf, sizeof(unsigned int)*numBins, cudaMemcpyDeviceToHost));





}
