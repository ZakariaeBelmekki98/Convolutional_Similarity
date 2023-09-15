#include <stdlib.h>
#include <stdio.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor inner_prods_cuda(torch::Tensor kernels);
float conv_sim_cuda(torch::Tensor inner_prods, const unsigned int M, const unsigned int C, const unsigned int N);
torch::Tensor conv_sim_grad_cuda(torch::Tensor inner_prods, torch::Tensor kernels);


torch::Tensor convs_2D_cuda(torch::Tensor kernels);
float conv_sim_2D_cuda(torch::Tensor convs, const unsigned int M, const unsigned int C, const unsigned int N);
torch::Tensor conv_sim_2D_grad_cuda(torch::Tensor convs, torch::Tensor kernels);


torch::Tensor inner_prods(torch::Tensor kernels){
	CHECK_INPUT(kernels);
	return inner_prods_cuda(kernels);
}


float conv_sim(torch::Tensor inner_prods, const unsigned int M, const unsigned int C, const unsigned int N){
	CHECK_INPUT(inner_prods);
	return conv_sim_cuda(inner_prods, M, C, N);
}

torch::Tensor conv_sim_grad(torch::Tensor inner_prods, torch::Tensor kernels){
	CHECK_INPUT(inner_prods);
	CHECK_INPUT(kernels);
	return conv_sim_grad_cuda(inner_prods, kernels);
}

torch::Tensor convs_2D(torch::Tensor kernels){
	CHECK_INPUT(kernels);
	return convs_2D_cuda(kernels);
}

float conv_sim_2D(torch::Tensor convs, const unsigned int M, const unsigned int C, const unsigned int N){
	CHECK_INPUT(convs);
	return conv_sim_2D_cuda(convs, M, C, N);
}

torch::Tensor conv_sim_2D_grad(torch::Tensor convs, torch::Tensor kernels){
	CHECK_INPUT(convs);
	CHECK_INPUT(kernels);
	return conv_sim_2D_grad_cuda(convs, kernels);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("inner_prods", &inner_prods, "Returns a vector containing the inner products of all possible shifts");
	m.def("conv_sim", &conv_sim, "Returns the value of the convolutional similarity");
	m.def("conv_sim_grad", &conv_sim_grad, "Returns the gradients for the convolutional similarity");

	m.def("convs_2D", &convs_2D, "Returns a vector containing the convolutions");
	m.def("conv_sim_2D", &conv_sim_2D, "Returns the value of the convolutional similarity 2D case");
	m.def("conv_sim_2D_grad", &conv_sim_2D_grad, "Returns the grads for the convolutional similarity");
}



