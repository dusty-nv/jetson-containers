#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/coord.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/util/host_tensor.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass_kernels/fpA_intB_gemm.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/util/device_rmsnorm.h>
#include <cutlass/layout/matrix.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/util/device_rmsnorm.h>
#include <cutlass/layout/matrix.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass_kernels/fpA_intB_gemm.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass_kernels/fpA_intB_gemm.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass_kernels/fpA_intB_gemm.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass_kernels/fpA_intB_gemm.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass_kernels/fpA_intB_gemm.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass_kernels/fpA_intB_gemm.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <kernel_forward.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass_kernels/fpA_intB_gemm.h>
#include <tvm/runtime/registry.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <kernel_forward.h>


void fused_decode8_relax_matmul1_cutlass_(DLTensor* model_layers_0_mlp_gate_up_proj_weight_int8_01, DLTensor* model_layers_0_mlp_gate_up_proj_weight_float16_11, DLTensor* lv66, DLTensor* out0){

  
  
  using namespace fastertransformer;
  constexpr auto QuantOp = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;

  int m = 1;
  int n = model_layers_0_mlp_gate_up_proj_weight_int8_01->shape[1] * 2;
  int k = model_layers_0_mlp_gate_up_proj_weight_int8_01->shape[0];

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
    
  gemm_fp16_int_bias_act<cutlass::uint4b_t, QuantOp>(static_cast<cutlass::half_t*>(lv66->data),
                static_cast<cutlass::uint4b_t*>(model_layers_0_mlp_gate_up_proj_weight_int8_01->data),
                static_cast<cutlass::half_t*>(model_layers_0_mlp_gate_up_proj_weight_float16_11->data),
                nullptr,
                static_cast<cutlass::half_t*>(out0->data),
                "identity",
                m, n, k, k, 0, nullptr, 0, stream);

}

int fused_decode8_relax_matmul1_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* out0) {
  fused_decode8_relax_matmul1_cutlass_(arg0,
  arg1,
  arg2,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_decode8_relax_matmul1_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* ret3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  fused_decode8_relax_matmul1_cutlass_wrapper_(arg0,arg1,arg2,ret3);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_rms_norm_cutlass_(DLTensor* lv2, DLTensor* model_layers_0_input_layernorm_weight, DLTensor* out0){

  
    using data_type = cutlass::half_t;
    using namespace cutlass::layout;

    int M = lv2->shape[0] * lv2->shape[1];
    int N = 4096;
    cutlass::MatrixCoord size(M, N);
    auto layout_2D = RowMajor::packed(size);
    auto layout_channels = RowMajor::packed({1, N});

    cutlass::TensorRef<data_type, RowMajor> _input((data_type*)lv2->data, layout_2D);
    cutlass::TensorRef<data_type, RowMajor> _weight((data_type*)model_layers_0_input_layernorm_weight->data, layout_channels);
    cutlass::TensorRef<data_type, RowMajor> _output((data_type*)out0->data, layout_2D);

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    cutlass::rmsnorm(size, _output, _input, _weight, stream, 1e-05);
    
}

int fused_rms_norm_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* out0) {
  fused_rms_norm_cutlass_(arg0,
  arg1,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_rms_norm_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* ret2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  fused_rms_norm_cutlass_wrapper_(arg0,arg1,ret2);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_rms_norm1_cutlass_(DLTensor* lv1613, DLTensor* model_layers_0_input_layernorm_weight1, DLTensor* out0){

  
    using data_type = cutlass::half_t;
    using namespace cutlass::layout;

    int M = 1;
    int N = 4096;
    cutlass::MatrixCoord size(M, N);
    auto layout_2D = RowMajor::packed(size);
    auto layout_channels = RowMajor::packed({1, N});

    cutlass::TensorRef<data_type, RowMajor> _input((data_type*)lv1613->data, layout_2D);
    cutlass::TensorRef<data_type, RowMajor> _weight((data_type*)model_layers_0_input_layernorm_weight1->data, layout_channels);
    cutlass::TensorRef<data_type, RowMajor> _output((data_type*)out0->data, layout_2D);

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    cutlass::rmsnorm(size, _output, _input, _weight, stream, 1e-05);
    
}

int fused_rms_norm1_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* out0) {
  fused_rms_norm1_cutlass_(arg0,
  arg1,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_rms_norm1_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* ret2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  fused_rms_norm1_cutlass_wrapper_(arg0,arg1,ret2);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_decode9_relax_matmul_relax_add1_cutlass_(DLTensor* model_layers_0_mlp_down_proj_weight_int8_01, DLTensor* model_layers_0_mlp_down_proj_weight_float16_11, DLTensor* lv1661, DLTensor* lv1653, DLTensor* out0){

  
  
  using namespace fastertransformer;
  constexpr auto QuantOp = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;

  int m = 1;
  int n = model_layers_0_mlp_down_proj_weight_int8_01->shape[1] * 2;
  int k = model_layers_0_mlp_down_proj_weight_int8_01->shape[0];

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
    
  gemm_fp16_int_bias_act<cutlass::uint4b_t, QuantOp>(static_cast<cutlass::half_t*>(lv1661->data),
                static_cast<cutlass::uint4b_t*>(model_layers_0_mlp_down_proj_weight_int8_01->data),
                static_cast<cutlass::half_t*>(model_layers_0_mlp_down_proj_weight_float16_11->data),
                static_cast<cutlass::half_t*>(lv1653->data),
                static_cast<cutlass::half_t*>(out0->data),
                "identity",
                m, n, k, k, 0, nullptr, 0, stream);

}

int fused_decode9_relax_matmul_relax_add1_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* arg3,
	DLTensor* out0) {
  fused_decode9_relax_matmul_relax_add1_cutlass_(arg0,
  arg1,
  arg2,
  arg3,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_decode9_relax_matmul_relax_add1_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* arg3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  DLTensor* ret4 = (DLTensor*)(((TVMValue*)args)[4].v_handle);
  fused_decode9_relax_matmul_relax_add1_cutlass_wrapper_(arg0,arg1,arg2,arg3,ret4);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_decode9_relax_matmul_relax_add_cutlass_(DLTensor* model_layers_0_mlp_down_proj_weight_int8_0, DLTensor* model_layers_0_mlp_down_proj_weight_float16_1, DLTensor* lv52, DLTensor* lv44, DLTensor* out0){

  
  
  using namespace fastertransformer;
  constexpr auto QuantOp = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;

  int m = lv52->shape[0] * lv52->shape[1];
  int n = model_layers_0_mlp_down_proj_weight_int8_0->shape[1] * 2;
  int k = model_layers_0_mlp_down_proj_weight_int8_0->shape[0];

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
    
  gemm_fp16_int_bias_act<cutlass::uint4b_t, QuantOp>(static_cast<cutlass::half_t*>(lv52->data),
                static_cast<cutlass::uint4b_t*>(model_layers_0_mlp_down_proj_weight_int8_0->data),
                static_cast<cutlass::half_t*>(model_layers_0_mlp_down_proj_weight_float16_1->data),
                static_cast<cutlass::half_t*>(lv44->data),
                static_cast<cutlass::half_t*>(out0->data),
                "identity",
                m, n, k, k, 4096, nullptr, 0, stream);

}

int fused_decode9_relax_matmul_relax_add_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* arg3,
	DLTensor* out0) {
  fused_decode9_relax_matmul_relax_add_cutlass_(arg0,
  arg1,
  arg2,
  arg3,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_decode9_relax_matmul_relax_add_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* arg3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  DLTensor* ret4 = (DLTensor*)(((TVMValue*)args)[4].v_handle);
  fused_decode9_relax_matmul_relax_add_cutlass_wrapper_(arg0,arg1,arg2,arg3,ret4);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_decode7_relax_matmul_relax_add1_cutlass_(DLTensor* model_layers_0_self_attn_o_proj_weight_int8_01, DLTensor* model_layers_0_self_attn_o_proj_weight_float16_11, DLTensor* lv1650, DLTensor* lv1613, DLTensor* out0){

  
  
  using namespace fastertransformer;
  constexpr auto QuantOp = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;

  int m = 1;
  int n = model_layers_0_self_attn_o_proj_weight_int8_01->shape[1] * 2;
  int k = model_layers_0_self_attn_o_proj_weight_int8_01->shape[0];

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
    
  gemm_fp16_int_bias_act<cutlass::uint4b_t, QuantOp>(static_cast<cutlass::half_t*>(lv1650->data),
                static_cast<cutlass::uint4b_t*>(model_layers_0_self_attn_o_proj_weight_int8_01->data),
                static_cast<cutlass::half_t*>(model_layers_0_self_attn_o_proj_weight_float16_11->data),
                static_cast<cutlass::half_t*>(lv1613->data),
                static_cast<cutlass::half_t*>(out0->data),
                "identity",
                m, n, k, k, 0, nullptr, 0, stream);

}

int fused_decode7_relax_matmul_relax_add1_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* arg3,
	DLTensor* out0) {
  fused_decode7_relax_matmul_relax_add1_cutlass_(arg0,
  arg1,
  arg2,
  arg3,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_decode7_relax_matmul_relax_add1_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* arg3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  DLTensor* ret4 = (DLTensor*)(((TVMValue*)args)[4].v_handle);
  fused_decode7_relax_matmul_relax_add1_cutlass_wrapper_(arg0,arg1,arg2,arg3,ret4);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_decode7_relax_matmul_relax_add_cutlass_(DLTensor* model_layers_0_self_attn_o_proj_weight_int8_0, DLTensor* model_layers_0_self_attn_o_proj_weight_float16_1, DLTensor* lv41, DLTensor* lv2, DLTensor* out0){

  
  
  using namespace fastertransformer;
  constexpr auto QuantOp = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;

  int m = lv41->shape[0] * lv41->shape[1];
  int n = model_layers_0_self_attn_o_proj_weight_int8_0->shape[1] * 2;
  int k = model_layers_0_self_attn_o_proj_weight_int8_0->shape[0];

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
    
  gemm_fp16_int_bias_act<cutlass::uint4b_t, QuantOp>(static_cast<cutlass::half_t*>(lv41->data),
                static_cast<cutlass::uint4b_t*>(model_layers_0_self_attn_o_proj_weight_int8_0->data),
                static_cast<cutlass::half_t*>(model_layers_0_self_attn_o_proj_weight_float16_1->data),
                static_cast<cutlass::half_t*>(lv2->data),
                static_cast<cutlass::half_t*>(out0->data),
                "identity",
                m, n, k, k, 4096, nullptr, 0, stream);

}

int fused_decode7_relax_matmul_relax_add_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* arg3,
	DLTensor* out0) {
  fused_decode7_relax_matmul_relax_add_cutlass_(arg0,
  arg1,
  arg2,
  arg3,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_decode7_relax_matmul_relax_add_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* arg3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  DLTensor* ret4 = (DLTensor*)(((TVMValue*)args)[4].v_handle);
  fused_decode7_relax_matmul_relax_add_cutlass_wrapper_(arg0,arg1,arg2,arg3,ret4);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_decode6_relax_matmul1_cutlass_(DLTensor* model_layers_0_self_attn_query_key_value_proj_weight_int8_01, DLTensor* model_layers_0_self_attn_query_key_value_proj_weight_float16_11, DLTensor* lv65, DLTensor* out0){

  
  
  using namespace fastertransformer;
  constexpr auto QuantOp = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;

  int m = 1;
  int n = model_layers_0_self_attn_query_key_value_proj_weight_int8_01->shape[1] * 2;
  int k = model_layers_0_self_attn_query_key_value_proj_weight_int8_01->shape[0];

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
    
  gemm_fp16_int_bias_act<cutlass::uint4b_t, QuantOp>(static_cast<cutlass::half_t*>(lv65->data),
                static_cast<cutlass::uint4b_t*>(model_layers_0_self_attn_query_key_value_proj_weight_int8_01->data),
                static_cast<cutlass::half_t*>(model_layers_0_self_attn_query_key_value_proj_weight_float16_11->data),
                nullptr,
                static_cast<cutlass::half_t*>(out0->data),
                "identity",
                m, n, k, k, 0, nullptr, 0, stream);

}

int fused_decode6_relax_matmul1_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* out0) {
  fused_decode6_relax_matmul1_cutlass_(arg0,
  arg1,
  arg2,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_decode6_relax_matmul1_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* ret3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  fused_decode6_relax_matmul1_cutlass_wrapper_(arg0,arg1,arg2,ret3);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_decode8_relax_matmul_cutlass_(DLTensor* model_layers_0_mlp_gate_up_proj_weight_int8_0, DLTensor* model_layers_0_mlp_gate_up_proj_weight_float16_1, DLTensor* lv1, DLTensor* out0){

  
  
  using namespace fastertransformer;
  constexpr auto QuantOp = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;

  int m = lv1->shape[0] * lv1->shape[1];
  int n = model_layers_0_mlp_gate_up_proj_weight_int8_0->shape[1] * 2;
  int k = model_layers_0_mlp_gate_up_proj_weight_int8_0->shape[0];

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
    
  gemm_fp16_int_bias_act<cutlass::uint4b_t, QuantOp>(static_cast<cutlass::half_t*>(lv1->data),
                static_cast<cutlass::uint4b_t*>(model_layers_0_mlp_gate_up_proj_weight_int8_0->data),
                static_cast<cutlass::half_t*>(model_layers_0_mlp_gate_up_proj_weight_float16_1->data),
                nullptr,
                static_cast<cutlass::half_t*>(out0->data),
                "identity",
                m, n, k, k, 0, nullptr, 0, stream);

}

int fused_decode8_relax_matmul_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* out0) {
  fused_decode8_relax_matmul_cutlass_(arg0,
  arg1,
  arg2,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_decode8_relax_matmul_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* ret3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  fused_decode8_relax_matmul_cutlass_wrapper_(arg0,arg1,arg2,ret3);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_relax_nn_attention1_cutlass1_(DLTensor* lv1625, DLTensor* lv1635, DLTensor* lv1636, DLTensor* workspace_2, DLTensor* out0){

  
  using T = cutlass::half_t;

  using Attention =
      AttentionKernel<T,
                      /*ArchTag=*/cutlass::arch::Sm80,
                      /*is_aligned=*/1,
                      /*queries_per_block=*/32,
                      /*keys_per_block=*/128,
                      /*kMaxK=*/128,
                      /*supports_dropout=*/0,
                      /*supports_bias=*/0
      >;

  typename Attention::Params p;
  p.logsumexp_ptr = nullptr;
  p.output_ptr = reinterpret_cast<T *>(out0->data);

  p.output_accum_ptr = nullptr;
  uint64_t accumulator_buf_size = 1 * 1 * 32 * 128 * sizeof(Attention::output_accum_t);
  bool accumulator_buf_allocated = false;
  if (Attention::kNeedsOutputAccumulatorBuffer) {
    if (accumulator_buf_size <= workspace_2->shape[0]) {
        p.output_accum_ptr = static_cast<float*>(workspace_2->data);
    } else {
        accumulator_buf_size = true;
        cudaMalloc(
          &p.output_accum_ptr,
          accumulator_buf_size
        );
    }
  }

  p.num_heads = 32; // N
  p.num_batches = 1; // B
  p.head_dim = 128; // H
  p.head_dim_value = 128; // H'
  p.num_queries = 1; // S
  p.num_keys = lv1635->shape[1]; // S'
  p.scale = 0.08838834764831843;
  p.custom_mask_type = 2;


  p.o_strideM = p.head_dim_value * p.num_heads; // H' * N
  CHECK(out0->ndim == 4); // B, S, N, H'

  
  p.query_ptr = reinterpret_cast<T *>(lv1625->data);
  p.key_ptr = reinterpret_cast<T *>(lv1635->data);
  p.value_ptr = reinterpret_cast<T *>(lv1636->data);
  CHECK(lv1625->ndim == 4); // B, S, N, H
  CHECK(lv1635->ndim == 4); // B, S', N, H
  CHECK(lv1636->ndim == 4); // B, S', N, H'

  // stride for N
  p.q_strideH = p.head_dim; // H
  p.k_strideH = p.head_dim; // H
  p.v_strideH = p.head_dim_value; // H'

  // stride for S
  p.q_strideM = p.q_strideH * p.num_heads; // H * N
  p.k_strideM = p.k_strideH * p.num_heads; // H * N
  p.v_strideM = p.v_strideH * p.num_heads; // H' * N

  // stride for B
  p.q_strideB = p.q_strideM * p.num_queries; // H * N * S
  p.k_strideB = p.k_strideM * p.num_keys; // H * N * S'
  p.v_strideB = p.v_strideM * p.num_keys; // H'* N * S'

  
  

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    static bool once = [&]() {
      cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      return true;
    }();
  }

  CHECK(Attention::check_supported(p));
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);

  if (accumulator_buf_allocated) {
    cudaFree(p.output_accum_ptr);
  }

}

int fused_relax_nn_attention1_cutlass1_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* arg3,
	DLTensor* out0) {
  fused_relax_nn_attention1_cutlass1_(arg0,
  arg1,
  arg2,
  arg3,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_relax_nn_attention1_cutlass1(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* arg3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  DLTensor* ret4 = (DLTensor*)(((TVMValue*)args)[4].v_handle);
  fused_relax_nn_attention1_cutlass1_wrapper_(arg0,arg1,arg2,arg3,ret4);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_decode6_relax_matmul_cutlass_(DLTensor* model_layers_0_self_attn_query_key_value_proj_weight_int8_0, DLTensor* model_layers_0_self_attn_query_key_value_proj_weight_float16_1, DLTensor* lv, DLTensor* out0){

  
  
  using namespace fastertransformer;
  constexpr auto QuantOp = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;

  int m = lv->shape[0] * lv->shape[1];
  int n = model_layers_0_self_attn_query_key_value_proj_weight_int8_0->shape[1] * 2;
  int k = model_layers_0_self_attn_query_key_value_proj_weight_int8_0->shape[0];

  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());
    
  gemm_fp16_int_bias_act<cutlass::uint4b_t, QuantOp>(static_cast<cutlass::half_t*>(lv->data),
                static_cast<cutlass::uint4b_t*>(model_layers_0_self_attn_query_key_value_proj_weight_int8_0->data),
                static_cast<cutlass::half_t*>(model_layers_0_self_attn_query_key_value_proj_weight_float16_1->data),
                nullptr,
                static_cast<cutlass::half_t*>(out0->data),
                "identity",
                m, n, k, k, 0, nullptr, 0, stream);

}

int fused_decode6_relax_matmul_cutlass_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* out0) {
  fused_decode6_relax_matmul_cutlass_(arg0,
  arg1,
  arg2,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_decode6_relax_matmul_cutlass(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* ret3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  fused_decode6_relax_matmul_cutlass_wrapper_(arg0,arg1,arg2,ret3);
  return 0;
}
#ifdef __cplusplus
}
#endif

void fused_relax_nn_attention_cutlass1_(DLTensor* lv16, DLTensor* lv26, DLTensor* lv27, DLTensor* workspace, DLTensor* out0){

  
  using T = cutlass::half_t;

  using Attention =
      AttentionKernel<T,
                      /*ArchTag=*/cutlass::arch::Sm80,
                      /*is_aligned=*/1,
                      /*queries_per_block=*/32,
                      /*keys_per_block=*/128,
                      /*kMaxK=*/128,
                      /*supports_dropout=*/0,
                      /*supports_bias=*/0
      >;

  typename Attention::Params p;
  p.logsumexp_ptr = nullptr;
  p.output_ptr = reinterpret_cast<T *>(out0->data);

  p.output_accum_ptr = nullptr;
  uint64_t accumulator_buf_size = 1 * lv16->shape[1] * 32 * 128 * sizeof(Attention::output_accum_t);
  bool accumulator_buf_allocated = false;
  if (Attention::kNeedsOutputAccumulatorBuffer) {
    if (accumulator_buf_size <= workspace->shape[0]) {
        p.output_accum_ptr = static_cast<float*>(workspace->data);
    } else {
        accumulator_buf_size = true;
        cudaMalloc(
          &p.output_accum_ptr,
          accumulator_buf_size
        );
    }
  }

  p.num_heads = 32; // N
  p.num_batches = 1; // B
  p.head_dim = 128; // H
  p.head_dim_value = 128; // H'
  p.num_queries = lv16->shape[1]; // S
  p.num_keys = lv26->shape[1]; // S'
  p.scale = 0.08838834764831843;
  p.custom_mask_type = 2;


  p.o_strideM = p.head_dim_value * p.num_heads; // H' * N
  CHECK(out0->ndim == 4); // B, S, N, H'

  
  p.query_ptr = reinterpret_cast<T *>(lv16->data);
  p.key_ptr = reinterpret_cast<T *>(lv26->data);
  p.value_ptr = reinterpret_cast<T *>(lv27->data);
  CHECK(lv16->ndim == 4); // B, S, N, H
  CHECK(lv26->ndim == 4); // B, S', N, H
  CHECK(lv27->ndim == 4); // B, S', N, H'

  // stride for N
  p.q_strideH = p.head_dim; // H
  p.k_strideH = p.head_dim; // H
  p.v_strideH = p.head_dim_value; // H'

  // stride for S
  p.q_strideM = p.q_strideH * p.num_heads; // H * N
  p.k_strideM = p.k_strideH * p.num_heads; // H * N
  p.v_strideM = p.v_strideH * p.num_heads; // H' * N

  // stride for B
  p.q_strideB = p.q_strideM * p.num_queries; // H * N * S
  p.k_strideB = p.k_strideM * p.num_keys; // H * N * S'
  p.v_strideB = p.v_strideM * p.num_keys; // H'* N * S'

  
  

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    static bool once = [&]() {
      cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      return true;
    }();
  }

  CHECK(Attention::check_supported(p));
  auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
  ICHECK(func != nullptr);
  cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);

  if (accumulator_buf_allocated) {
    cudaFree(p.output_accum_ptr);
  }

}

int fused_relax_nn_attention_cutlass1_wrapper_(DLTensor* arg0,
	DLTensor* arg1,
	DLTensor* arg2,
	DLTensor* arg3,
	DLTensor* out0) {
  fused_relax_nn_attention_cutlass1_(arg0,
  arg1,
  arg2,
  arg3,
  out0);
  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
TVM_DLL int32_t fused_relax_nn_attention_cutlass1(TVMValue* args, int* type_code, int num_args, TVMValue* out_value, int* out_type_code) {
  DLTensor* arg0 = (DLTensor*)(((TVMValue*)args)[0].v_handle);
  DLTensor* arg1 = (DLTensor*)(((TVMValue*)args)[1].v_handle);
  DLTensor* arg2 = (DLTensor*)(((TVMValue*)args)[2].v_handle);
  DLTensor* arg3 = (DLTensor*)(((TVMValue*)args)[3].v_handle);
  DLTensor* ret4 = (DLTensor*)(((TVMValue*)args)[4].v_handle);
  fused_relax_nn_attention_cutlass1_wrapper_(arg0,arg1,arg2,arg3,ret4);
  return 0;
}
#ifdef __cplusplus
}
#endif
