#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


const int THREADS_PER_BLOCK = 1024;

int COMB(const unsigned int n){
        unsigned int res = 0;
        for(unsigned int i=1; i<n; i++) res += n - i;
        return res;
}

__device__ void warpReduce(volatile float* cache, int tid){
        cache[tid] += cache[tid + 32];
        cache[tid] += cache[tid + 16];
        cache[tid] += cache[tid+8];
        cache[tid] += cache[tid+4];
        cache[tid] += cache[tid+2];
        cache[tid] += cache[tid+1];
}



static void HandleError(cudaError_t err,
                        const char *file,
                        int line){
        if (err != cudaSuccess){
                printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
                exit(EXIT_FAILURE);
        }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void inner_prods_cuda_kernel(
		const float *kernels_ptr,
		float *inner_prods_ptr,
		const int SIZE,
		const int M,
		const int C, 
		const int N){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid < SIZE){
		int i, j, c, b;
		i = tid/C/(2*N-1);
		j = i+1;
		c = (tid%((2*N-1)*C))/(2*N-1);	
		b = ((tid%((2*N-1)*C))%(2*N-1))-N+1;
		for(int n=0; n<N; n++) if(n+b < N && n+b >= 0) inner_prods_ptr[tid] += kernels_ptr[i*N*C + c*N + n] * kernels_ptr[j*N*C + c*N + (n+b)];
		__syncthreads();
		tid += blockDim.x * gridDim.x;	
	}
}

__global__ void convs_2D_cuda_kernel(
		const float *kernels_ptr,
		float *convs_ptr,
		const int SIZE,
		const int M,
		const int C,
		const int N){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while(tid < SIZE){
		int i, j, c, b1, b2;
		int dim4 = 2*N-1;
		i = tid/C/dim4/dim4;
		j = i+1;
		c = (tid%(dim4*C*dim4))/dim4/dim4;
		b1 = (tid%(dim4*C*dim4))%(dim4*dim4)/dim4 - N+1;
		b2 = (tid%(dim4*C*dim4))%(dim4*dim4)%dim4 - N+1;

		for(int n=0; n<N; n++){
			for(int m=0; m<N; m++){
				if(n+b1 < N && n+b1 >= 0 && m+b2 < N && m+b2 >= 0){
					convs_ptr[tid] += kernels_ptr[i*N*N*C + c*N*N + n*N + m] * kernels_ptr[j*N*N*C + c*N*N + (n+b1)*N + (m+b2)]; 
				}
			}
		}
		tid += blockDim.x * gridDim.x;
	}
}


__global__ void conv_sim_cuda_kernel(
                const float *inner_prods,
                float* output,
                const unsigned int SIZE){

        extern __shared__ float cache[];
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int cacheIndex = threadIdx.x;
        float temp = 0;
        while(tid<SIZE){
                temp += inner_prods[tid] * inner_prods[tid];
                tid += blockDim.x * gridDim.x;
        }
        cache[cacheIndex] = temp;
        __syncthreads();

        for(unsigned int s=blockDim.x/2; s>32; s>>=1){
                if(cacheIndex < s){
                        cache[cacheIndex] += cache[cacheIndex +s];
                }
                __syncthreads();
        }
        if(cacheIndex < 32) warpReduce(cache, cacheIndex);
        if(cacheIndex == 0) output[blockIdx.x] = cache[0];
}



__global__ void conv_sim_grad_cuda_kernel(
		const float *inner_prods,
		const float *kernels,
		float *grad,
		const int SIZE,
		const int M,
		const int C,
		const int N){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid < SIZE){
		int a = tid/N/C;
		int c = (tid%(N*C))/N;
		int x = (tid%(N*C))%N;

		for(int b=1-N; b < N; b++){
			if(a != M-1 && x+b < N && x+b >= 0) grad[tid] += inner_prods[a*(2*N-1)*C + c*(2*N-1) + b+N-1] * kernels[(a+1)*N*C + c*N + x+b];
		       	if(a != 0 && x-b < N && x-b >= 0) grad[tid] += inner_prods[(a-1)*(2*N-1)*C + c*(2*N-1) + b+N-1] * kernels[(a-1)*N*C + c*N + x-b];	
		
		}
		grad[tid] *= 2.;
		tid += blockDim.x * gridDim.x;
	}
}


__global__ void conv_sim_2D_grad_cuda_kernel(
		const float *convs,
		const float *kernels,
		float *grads,
		const int SIZE,
		const int M,
		const int C,
		const int N){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(tid < SIZE){
		int a = tid/N/N/C;
		int c = (tid%(N*N*C))/N/N;
		int x1 = ((tid%(N*N*C))%(N*N))/N;
		int x2 = ((tid%(N*N*C))%(N*N))%N;
		
		int dim4 = 2*N-1;
		for(int n=1-N; n < N; n++){
			for(int m=1-N; m < N; m++){
				if(a != M-1 && x1+n >= 0 && x1+n < N && x2+m >= 0 && x2+m < N) grads[tid] += convs[a*dim4*dim4*C + c*dim4*dim4+ (n+N-1)*dim4 + m+N-1] * kernels[(a+1)*N*N*C + c*N*N + (x1+n)*N + (x2+m)];
				if(a != 0 && x1-n >= 0 && x1-n < N && x2-m >= 0 && x2-m < N) grads[tid] += convs[(a-1)*dim4*dim4*C + c*dim4*dim4 + (n+N-1)*dim4 + m+N-1] * kernels[(a-1)*N*N*C + c*N*N + (x1-n)*N + (x2-m)];
			}
		}
		grads[tid] *= 2.;
		tid += blockDim.x * gridDim.x;
	}
}


torch::Tensor inner_prods_cuda(torch::Tensor kernels){
	const unsigned int M = kernels.sizes()[0];
	const unsigned int C = kernels.sizes()[1];
	const unsigned int N = kernels.sizes()[2] * kernels.sizes()[3];

	
	const int SIZE =  (M-1)*(2*N-1)*C;
	int blocksPerGrid = (SIZE+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
	
	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	torch::Tensor result = torch::zeros({SIZE}, options);

	float *kernels_ptr = (float*) kernels.data_ptr();
	float *inner_prods_ptr = (float*) result.data_ptr();

	inner_prods_cuda_kernel<<<blocksPerGrid, THREADS_PER_BLOCK >>>(kernels_ptr, inner_prods_ptr, SIZE, M, C, N);
	HANDLE_ERROR(cudaDeviceSynchronize());
	return result;
}


torch::Tensor convs_2D_cuda(torch::Tensor kernels){
	const unsigned int M = kernels.sizes()[0];
	const unsigned int C = kernels.sizes()[1];
	const unsigned int N = kernels.sizes()[2];

	const int SIZE = (M-1)*C*(2*N-1)*(2*N-1);
	const int BLOCKS_PER_GRID = (SIZE + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	torch::Tensor result = torch::zeros({SIZE}, options);

	float *kernels_ptr = (float*) kernels.data_ptr();
	float *convs_ptr = (float*) result.data_ptr();
	
	convs_2D_cuda_kernel<<< BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(kernels_ptr, convs_ptr, SIZE, M, C, N);
	HANDLE_ERROR(cudaDeviceSynchronize());
	return result;
}

float conv_sim_cuda(torch::Tensor inner_prods, const unsigned int M, const unsigned int C, const unsigned int N){
	float *inner_prods_ptr = (float*) inner_prods.data_ptr();
	const int SIZE = (M-1)*(2*N-1)*C;

	int BLOCKS_PER_GRID = (SIZE+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
	int o_size = SIZE/(THREADS_PER_BLOCK/2);
	if(SIZE % (THREADS_PER_BLOCK/2)) o_size++;

	float *output = (float*) malloc(o_size * sizeof(float));
	float *output_dev;
	float result = 0.0;

	HANDLE_ERROR(cudaMalloc((void**) &output_dev, o_size * sizeof(float)));
	conv_sim_cuda_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float)>>>(inner_prods_ptr, output_dev, SIZE);
	HANDLE_ERROR(cudaMemcpy(output, output_dev, sizeof(float)*o_size, cudaMemcpyDeviceToHost));
	for(int i=0; i<o_size; i++) result += output[i];
	free(output);
	HANDLE_ERROR(cudaFree(output_dev));
	return result;
}


float conv_sim_2D_cuda(torch::Tensor convs, const unsigned int M, const unsigned int C, const unsigned int N){
	float *convs_ptr = (float*) convs.data_ptr();
	const int SIZE = (M-1) * (2*N-1) * (2*N-1) * C;
	const int BLOCKS_PER_GRID = (SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	int o_size = SIZE / (THREADS_PER_BLOCK/2);
	if(SIZE % (THREADS_PER_BLOCK/2)) o_size++;

	float *output = (float*) malloc(o_size * sizeof(float));
	float *output_dev;
	float result = 0.0;

	HANDLE_ERROR(cudaMalloc((void**) &output_dev, o_size * sizeof(float)));
	conv_sim_cuda_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float) >>>(convs_ptr, output_dev, SIZE);
	HANDLE_ERROR(cudaMemcpy(output, output_dev, sizeof(float)*o_size, cudaMemcpyDeviceToHost));
	for(int i=0; i<o_size; i++) result += output[i];
	free(output);
	HANDLE_ERROR(cudaFree(output_dev));
	return result;
}

torch::Tensor conv_sim_grad_cuda(torch::Tensor inner_prods, torch::Tensor kernels){
	const unsigned int M = kernels.sizes()[0];
	const unsigned int C = kernels.sizes()[1];
	const unsigned int N = kernels.sizes()[2] * kernels.sizes()[3];

	const unsigned int SIZE = N*C*M;
	const int BLOCKS_PER_GRID = (SIZE + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	torch::Tensor result = torch::zeros({M, C, kernels.sizes()[2], kernels.sizes()[3]}, options);

	float *kernels_ptr = (float*) kernels.data_ptr();
	float *inner_prods_ptr = (float*) inner_prods.data_ptr();
	float *grad_ptr = (float*) result.data_ptr();

	conv_sim_grad_cuda_kernel<<< BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(inner_prods_ptr, kernels_ptr, grad_ptr, SIZE, M, C, N);
	HANDLE_ERROR(cudaDeviceSynchronize());
	return result;
}


torch::Tensor conv_sim_2D_grad_cuda(torch::Tensor convs, torch::Tensor kernels){
	const unsigned int M = kernels.sizes()[0];
	const unsigned int C = kernels.sizes()[1];
	const unsigned int N = kernels.sizes()[2];

	const unsigned int SIZE = N*N*C*M;
	const int BLOCKS_PER_GRID = (SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

	auto options = torch::TensorOptions().device(torch::kCUDA, 0);
	torch::Tensor result = torch::zeros({M, C, N, N}, options);

	float *kernels_ptr = (float*) kernels.data_ptr();
	float *convs_ptr = (float*) convs.data_ptr();
	float *grads_ptr = (float*) result.data_ptr();

	conv_sim_2D_grad_cuda_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK >>>(convs_ptr, kernels_ptr, grads_ptr, SIZE, M, C, N);
	HANDLE_ERROR(cudaDeviceSynchronize());
	return result;
}




