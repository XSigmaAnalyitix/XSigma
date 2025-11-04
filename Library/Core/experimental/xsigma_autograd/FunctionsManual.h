#pragma once

// NB: Must be xsigma the top of file to avoid including the deprecated "math.h".
// https://stackoverflow.com/questions/6563810/m-pi-works-with-math-h-but-not-with-cmath-in-visual-studio
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#endif

#include <ATen/ATen.h>
#include <torch/csrc/autograd/generated/Functions.h>

namespace torch::autograd::generated::details
{

extern const char* kCudnnDoubleBackwardMsg;

// A simple way to imperatively compute index ranges for slots
// that have been flattened
struct TORCH_API IndexRangeGenerator
{
    IndexRange range(size_t range_size)
    {
        i += range_size;
        return {i - range_size, i};
    }
    size_t size() { return i; }

private:
    size_t i = 0;
};

TORCH_API Tensor toNonOptFwGrad(const std::optional<Tensor>& t);
TORCH_API Tensor toNonOptPrimal(const std::optional<Tensor>& t);
TORCH_API Tensor toNonOptTensor(const std::optional<Tensor>& t);

inline std::optional<Tensor> wrap_opt_if(const Tensor& t, const bool cond)
{
    using OptTensor = std::optional<Tensor>;
    return cond ? OptTensor(t) : static_cast<OptTensor>(std::nullopt);
}

TORCH_API Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction);
TORCH_API bool   any_variable_defined(const variable_list& variables);
TORCH_API void   copy_range(variable_list& out, IndexRange range, const xsigma::Tensor& t);
TORCH_API void copy_range(variable_list& out, IndexRange range, xsigma::ArrayRef<xsigma::Tensor> t);
TORCH_API xsigma::Tensor copysign_tensor_self_backward(
    const Tensor& grad, const Tensor& self, const Tensor& result);
TORCH_API xsigma::Tensor not_implemented(const char* name, const char* reason = "");
TORCH_API std::vector<Tensor> not_implemented_list(const char* name, const char* reason = "");
xsigma::Tensor                handle_r_to_c(ScalarType self_st, Tensor gradient_result);
xsigma::Tensor                maybe_multiply(const xsigma::Tensor& t, const xsigma::Scalar& s);
int64_t                       _safe_size(IntArrayRef sizes, IntArrayRef dim);
Tensor         restore_reduced_dims(const Tensor& output, IntArrayRef dims, bool keepdim);
Tensor         scale_grad_by_count(const Tensor& grad, const Tensor& mask, IntArrayRef dims);
xsigma::Tensor norm_backward(
    const xsigma::Tensor&                grad,
    const xsigma::Tensor&                self,
    const std::optional<xsigma::Scalar>& p_,
    const xsigma::Tensor&                norm);
xsigma::Tensor norm_backward(
    xsigma::Tensor                       grad,
    const xsigma::Tensor&                self,
    const std::optional<xsigma::Scalar>& p_,
    xsigma::Tensor                       norm,
    xsigma::IntArrayRef                  dim,
    bool                                 keepdim);
Tensor norm_jvp(
    const Tensor&                self_p,
    const Tensor&                self_t,
    const std::optional<Scalar>& p_,
    Tensor                       norm,
    IntArrayRef                  dim,
    bool                         keepdim);
Tensor norm_jvp(
    const Tensor& grad, const Tensor& self, const std::optional<Scalar>& p_, Tensor norm);
Tensor _nested_from_padded_backward(
    const Tensor& grad, const Tensor& input, const bool do_transform_0213);
std::tuple<Tensor, Tensor, Tensor> linear_double_backward(
    const variable_list& grads,
    const Tensor&        self,
    const Tensor&        grad_output,
    const Tensor&        weight);
Tensor linalg_vector_norm_jvp(
    const Tensor&                      self_p,
    const Tensor&                      self_t,
    const Scalar&                      scalar_ord,
    Tensor                             norm,
    const xsigma::OptionalIntArrayRef& opt_dim,
    bool                               keepdim);
xsigma::Tensor linalg_vector_norm_backward(
    xsigma::Tensor                     grad,
    const xsigma::Tensor&              self,
    const xsigma::Scalar&              ord,
    xsigma::Tensor                     norm,
    const xsigma::OptionalIntArrayRef& opt_dim,
    bool                               keepdim);
xsigma::Tensor pow_backward(
    xsigma::Tensor grad, const xsigma::Tensor& self, const xsigma::Scalar& exponent_);
xsigma::Tensor pow_backward_self(
    const xsigma::Tensor& grad, const xsigma::Tensor& self, const xsigma::Tensor& exponent);
xsigma::Tensor pow_backward_exponent(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& self,
    const xsigma::Tensor& exponent,
    const xsigma::Tensor& result);
xsigma::Tensor pow_backward_exponent(
    const xsigma::Tensor& grad,
    const xsigma::Scalar& base,
    const xsigma::Tensor& exponent,
    const xsigma::Tensor& result);
xsigma::Tensor angle_backward(const xsigma::Tensor& grad, const xsigma::Tensor& self);
template <typename T>
xsigma::Tensor mul_tensor_backward(const Tensor& grad, T other, ScalarType self_st);
template <typename T>
xsigma::Tensor div_tensor_self_backward(
    const Tensor&                          grad,
    T                                      other,
    ScalarType                             self_st,
    const std::optional<std::string_view>& rounding_mode = std::nullopt);
xsigma::Tensor div_tensor_other_backward(
    const Tensor&                          grad,
    const Tensor&                          self,
    const Tensor&                          other,
    const std::optional<std::string_view>& rounding_mode = std::nullopt);
xsigma::Tensor mvlgamma_backward(const xsigma::Tensor& grad, const xsigma::Tensor& self, int64_t p);
xsigma::Tensor permute_backwards(const xsigma::Tensor& grad, xsigma::IntArrayRef fwd_dims);
xsigma::Tensor rad2deg_backward(const xsigma::Tensor& grad);
xsigma::Tensor deg2rad_backward(const xsigma::Tensor& grad);
xsigma::Tensor unsqueeze_multiple(
    const xsigma::Tensor& t, xsigma::OptionalIntArrayRef opt_dim, size_t n_dims);
xsigma::Tensor sum_backward(
    const xsigma::Tensor&       grad,
    xsigma::SymIntArrayRef      sizes,
    xsigma::OptionalIntArrayRef opt_dims,
    bool                        keepdim);
xsigma::Tensor sum_backward(
    const xsigma::Tensor&  grad,
    xsigma::SymIntArrayRef sizes,
    xsigma::IntArrayRef    dims,
    bool                   keepdim);
xsigma::Tensor nansum_backward(
    const xsigma::Tensor&       grad,
    const xsigma::Tensor&       self,
    xsigma::OptionalIntArrayRef dims,
    bool                        keepdim);
std::vector<int64_t>        reverse_list(const xsigma::IntArrayRef list);
std::vector<xsigma::SymInt> reverse_list_symint(const xsigma::SymIntArrayRef list);
xsigma::Tensor              reverse_dim(const xsigma::Tensor& t, int64_t dim);
xsigma::Tensor              prod_safe_zeros_backward(
                 const xsigma::Tensor& grad, const xsigma::Tensor& inp, int64_t dim);
xsigma::Tensor prod_backward(
    const xsigma::Tensor& grad, const xsigma::Tensor& input, const xsigma::Tensor& result);
xsigma::Tensor prod_backward(
    xsigma::Tensor        grad,
    const xsigma::Tensor& input,
    xsigma::Tensor        result,
    int64_t               dim,
    bool                  keepdim);
xsigma::Tensor solve_jvp(const Tensor& X, const Tensor& A, const Tensor& dA, const Tensor& dB);
xsigma::Tensor solve_backward_self(
    const xsigma::Tensor& grad, const xsigma::Tensor& self, const xsigma::Tensor& A);
xsigma::Tensor solve_backward_A(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& self,
    const xsigma::Tensor& A,
    const xsigma::Tensor& solution);
xsigma::Tensor cumsum_backward(const xsigma::Tensor& grad, int64_t dim);
xsigma::Tensor logsumexp_backward(
    xsigma::Tensor        grad,
    const xsigma::Tensor& self,
    xsigma::Tensor        result,
    xsigma::IntArrayRef   dim,
    bool                  keepdim);
xsigma::Tensor logsumexp_jvp(
    const xsigma::Tensor& self_p, const xsigma::Tensor& self_t, IntArrayRef dim, bool keepdim);
xsigma::Tensor safe_logsumexp_jvp(
    const xsigma::Tensor& self_p, const xsigma::Tensor& self_t, IntArrayRef dim, bool keepdim);
xsigma::Tensor logcumsumexp_backward(
    xsigma::Tensor grad, const xsigma::Tensor& self, const xsigma::Tensor& result, int64_t dim);
xsigma::Tensor logcumsumexp_jvp(
    const xsigma::Tensor& self_p, const xsigma::Tensor& self_t, int64_t dim);
xsigma::Tensor unbind_backward(const variable_list& grads, int64_t dim);
xsigma::Tensor unbind_backward_nested(
    const variable_list&         grads,
    const Tensor&                nt_sizes,
    int64_t                      dim,
    const xsigma::TensorOptions& options);
xsigma::Tensor unbind_backward_nested_jagged(
    const variable_list& grads, const Tensor& self, int64_t dim);
xsigma::Tensor unsqueeze_to(const xsigma::Tensor& self, xsigma::SymIntArrayRef sym_sizes);
xsigma::Tensor unsqueeze_to(
    const xsigma::Tensor& self, int64_t dim, xsigma::SymIntArrayRef sym_sizes);
xsigma::Tensor unsqueeze_to(
    const xsigma::Tensor& self, IntArrayRef dim, xsigma::SymIntArrayRef sym_sizes);
std::vector<xsigma::Tensor> cat_tensors_backward(
    const xsigma::Tensor&                           grad,
    const std::vector<std::vector<xsigma::SymInt>>& sizes,
    const std::vector<ScalarType>&                  dtypes,
    int64_t                                         dim);
std::vector<xsigma::Tensor> stack_tensors_backward(
    const xsigma::Tensor& grad, int64_t dim, const std::vector<ScalarType>& dtypes);
std::vector<xsigma::Tensor> block_diag_backward(
    const xsigma::Tensor&                    grad,
    const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<ScalarType>&           dtypes);
xsigma::Tensor clamp_backward(
    const xsigma::Tensor&                grad,
    const xsigma::Tensor&                self,
    const std::optional<xsigma::Scalar>& min,
    const std::optional<xsigma::Scalar>& max);
xsigma::Tensor clamp_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& self,
    const xsigma::Tensor& min,
    const xsigma::Tensor& max);
std::tuple<xsigma::Tensor, xsigma::Tensor> clamp_backward_min_max(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& self,
    const xsigma::Tensor& min,
    const xsigma::Tensor& max,
    const std::array<bool, 2>& /*grad_input_mask*/);
xsigma::Tensor clamp_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const Tensor& min_p,
    const Tensor& min_t,
    const Tensor& max_p,
    const Tensor& max_t);
xsigma::SymIntArrayRef strides_or_error(const Tensor& input, std::string_view const& input_name);
xsigma::Tensor         mm_mat1_backward(
            const Tensor&          grad,
            const Tensor&          mat2,
            xsigma::SymIntArrayRef mat1_sizes,
            xsigma::SymIntArrayRef mat1_strides,
            xsigma::Layout         mat1_layout,
            const Scalar&          alpha);
xsigma::Tensor mm_mat2_backward(
    const xsigma::Tensor&  grad,
    const xsigma::Tensor&  mat1,
    xsigma::SymIntArrayRef sizes,
    xsigma::SymIntArrayRef strides,
    xsigma::Layout         layout,
    const xsigma::Scalar&  alpha);
xsigma::Tensor _grouped_mm_mat1_backward(
    const Tensor&          grad,
    const Tensor&          mat2,
    xsigma::SymIntArrayRef mat1_sizes,
    xsigma::SymIntArrayRef mat1_strides,
    xsigma::Layout         mat1_layout,
    std::optional<Tensor>  offs,
    const Scalar&          alpha);
xsigma::Tensor _grouped_mm_mat2_backward(
    const xsigma::Tensor&  grad,
    const xsigma::Tensor&  mat1,
    xsigma::SymIntArrayRef sizes,
    xsigma::SymIntArrayRef strides,
    xsigma::Layout         layout,
    std::optional<Tensor>  offs,
    const xsigma::Scalar&  alpha);
xsigma::Tensor mm_mat1_sparse_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& mat1,
    const xsigma::Tensor& mat2,
    const xsigma::Scalar& alpha);
std::tuple<Tensor, Tensor, Tensor> sparse_sampled_addmm_backward(
    const Tensor&                grad,
    const Tensor&                self,
    const std::optional<Tensor>& mat1,
    const std::optional<Tensor>& mat2,
    const Scalar&                alpha,
    const Scalar&                beta,
    const std::array<bool, 3>&   grad_input_mask);
xsigma::Tensor sparse_mask_backward(
    const xsigma::Tensor& grad, const xsigma::Tensor& mask, xsigma::Layout self_layout);
xsigma::Tensor sparse_sparse_matmul_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& mat1,
    const xsigma::Tensor& mat2,
    int64_t               grad_order);
xsigma::Tensor renorm_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& self,
    const xsigma::Scalar& p,
    int64_t               dim,
    const xsigma::Scalar& maxnorm);
xsigma::Tensor renorm_jvp(
    const xsigma::Tensor& self_p,
    const xsigma::Tensor& self_t,
    const xsigma::Scalar& p,
    int64_t               dim,
    const xsigma::Scalar& maxnorm);
xsigma::Tensor repeat_backward(
    xsigma::Tensor grad, xsigma::SymIntArrayRef repeats, xsigma::SymIntArrayRef input_shape);
xsigma::Tensor _fused_dropout_backward(
    const xsigma::Tensor& grad, const xsigma::Tensor& mask, double p1m);
xsigma::Tensor infinitely_differentiable_native_dropout_backward(
    const xsigma::Tensor& grad, const xsigma::Tensor& mask, double scale);
xsigma::Tensor native_dropout_double_backward(
    const xsigma::Tensor& ggI,
    const xsigma::Tensor& grad,
    const xsigma::Tensor& mask,
    double                scale);
xsigma::Tensor evenly_distribute_backward(
    const xsigma::Tensor& grad, const xsigma::Tensor& input, const xsigma::Tensor& value);
Tensor         sgn_backward(const Tensor& x, const Tensor& gx, const Tensor& sgn);
Tensor         masked_fill_backward(const Tensor& grad, const Tensor& mask);
xsigma::Tensor var_backward(
    xsigma::Tensor                       grad,
    const xsigma::Tensor&                self,
    xsigma::OptionalIntArrayRef          dim,
    const std::optional<xsigma::Scalar>& correction,
    bool                                 keepdim);
xsigma::Tensor var_jvp(
    const xsigma::Tensor&                self_t,
    const xsigma::Tensor&                self_p,
    const xsigma::Tensor&                result,
    xsigma::OptionalIntArrayRef          dim_opt,
    const std::optional<xsigma::Scalar>& correction,
    bool                                 keepdim);
xsigma::Tensor std_backward(
    const xsigma::Tensor&                result,
    const xsigma::Tensor&                grad,
    const xsigma::Tensor&                self,
    xsigma::OptionalIntArrayRef          dim,
    const std::optional<xsigma::Scalar>& correction,
    bool                                 keepdim);
Tensor mean_backward(
    const Tensor&               grad,
    xsigma::SymIntArrayRef      shape,
    xsigma::OptionalIntArrayRef opt_dim,
    xsigma::SymInt              numel,
    bool                        keepdim);
Tensor var_mean_backward(
    const Tensor&                        gvar,
    const Tensor&                        gmean,
    const Tensor&                        self,
    xsigma::OptionalIntArrayRef          dim_opt,
    const std::optional<xsigma::Scalar>& correction,
    bool                                 keepdim);
Tensor std_mean_backward(
    const Tensor&                        gstd,
    const Tensor&                        gmean,
    const Tensor&                        self,
    const Tensor&                        std,
    xsigma::OptionalIntArrayRef          dim_opt,
    const std::optional<xsigma::Scalar>& correction,
    bool                                 keepdim);
xsigma::Tensor cholesky_backward(const xsigma::Tensor& grad, bool upper, const xsigma::Tensor& L);
xsigma::Tensor cholesky_jvp(
    const xsigma::Tensor& input_tangent, const xsigma::Tensor& L, bool upper);
xsigma::Tensor cholesky_inverse_backward(
    const xsigma::Tensor& grad, const xsigma::Tensor& L, bool upper, const xsigma::Tensor& inverse);
xsigma::Tensor cholesky_inverse_jvp(
    const xsigma::Tensor& F, const xsigma::Tensor& dF, const xsigma::Tensor& X, bool upper);
Tensor pinv_jvp(const Tensor& A, const Tensor& pinvA, const Tensor& dA);
Tensor pinv_backward(const Tensor& grad, const Tensor& pinvA, const Tensor& A);
Tensor chunk_backward_nested(
    const std::vector<torch::autograd::Variable>& grads,
    const Tensor&                                 self,
    int64_t                                       chunks,
    int64_t                                       dim);
xsigma::Tensor split_with_sizes_backward(
    const std::vector<torch::autograd::Variable>& grads,
    xsigma::SymIntArrayRef                        split_sizes,
    int64_t                                       dim,
    xsigma::SymIntArrayRef                        sizes,
    const xsigma::TensorOptions&                  options);
xsigma::Tensor _nested_split_with_sizes_backward(
    const std::vector<torch::autograd::Variable>& grads,
    xsigma::SymIntArrayRef                        split_sizes,
    int64_t                                       dim,
    const Tensor&                                 nt_sizes,
    const xsigma::TensorOptions&                  options);
xsigma::Tensor split_backward(
    const std::vector<torch::autograd::Variable>& grads,
    const xsigma::SymInt&                         split_size,
    int64_t                                       dim,
    xsigma::SymIntArrayRef                        sizes,
    const xsigma::TensorOptions&                  options);
xsigma::Tensor max_pool_double_backward(
    const xsigma::Tensor& grad, const xsigma::Tensor& indices, int dim);
xsigma::Tensor error_for_max_pool2d_double_backward();
xsigma::Tensor glu_double_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& grad_output,
    const xsigma::Tensor& input,
    int64_t               dim);
xsigma::Tensor glu_double_backward_grad_output(
    const xsigma::Tensor& grad, const xsigma::Tensor& input, int64_t dim);
xsigma::Tensor infinitely_differentiable_silu_backward(
    const xsigma::Tensor& grad_output, const xsigma::Tensor& input);
xsigma::Tensor infinitely_differentiable_mish_backward(
    const xsigma::Tensor& grad_output, const xsigma::Tensor& input);
Tensor infinitely_differentiable_logit_backward(
    const Tensor& grad, const Tensor& self, std::optional<double> eps);
Tensor binary_cross_entropy_target_backward(
    const Tensor&                grad,
    const Tensor&                self,
    const Tensor&                target,
    const std::optional<Tensor>& weight,
    int64_t                      reduction);
Tensor binary_cross_entropy_double_backward_target(
    const Tensor&                grad,
    const Tensor&                grad_output,
    const Tensor&                self,
    const Tensor&                target,
    const std::optional<Tensor>& weight,
    int64_t                      reduction);
Tensor binary_cross_entropy_with_logits_backward(
    const Tensor&                grad,
    const Tensor&                input,
    const Tensor&                target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt,
    int64_t                      reduction);
xsigma::Tensor binary_cross_entropy_with_logits_target_backward(
    const xsigma::Tensor&                grad_output,
    const xsigma::Tensor&                self,
    const xsigma::Tensor&                target,
    const std::optional<xsigma::Tensor>& weight,
    const std::optional<xsigma::Tensor>& pos_weight,
    int64_t                              reduction);
xsigma::Tensor log_sigmoid_double_backward(const xsigma::Tensor& grad, const xsigma::Tensor& input);
xsigma::Tensor softmax_double_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& grad_output,
    int                   dim,
    const xsigma::Tensor& output);
xsigma::Tensor binary_cross_entropy_double_backward(
    const xsigma::Tensor&                grad_output,
    const xsigma::Tensor&                grad,
    const xsigma::Tensor&                input,
    const xsigma::Tensor&                target,
    const std::optional<xsigma::Tensor>& weight,
    int64_t                              reduction);
xsigma::Tensor binary_cross_entropy_double_backward_grad_output(
    const xsigma::Tensor&                grad,
    const xsigma::Tensor&                input,
    const xsigma::Tensor&                target,
    const std::optional<xsigma::Tensor>& weight,
    int64_t                              reduction);
xsigma::Tensor smooth_l1_loss_double_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& input,
    const xsigma::Tensor& target,
    int64_t               reduction,
    double                beta);
xsigma::Tensor huber_loss_double_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& input,
    const xsigma::Tensor& target,
    int64_t               reduction,
    double                delta);
xsigma::Tensor huber_loss_double_backward_grad_output(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& grad_output,
    const xsigma::Tensor& input,
    const xsigma::Tensor& target,
    int64_t               reduction,
    double                delta);
xsigma::Tensor mse_loss_double_backward(
    const xsigma::Tensor& grad, const xsigma::Tensor& input, int64_t reduction);
xsigma::Tensor soft_margin_loss_double_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& input,
    const xsigma::Tensor& target,
    int64_t               reduction);
xsigma::Tensor soft_margin_loss_double_backward_grad_output(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& grad_output,
    const xsigma::Tensor& input,
    const xsigma::Tensor& target,
    int64_t               reduction);
xsigma::Tensor softplus_double_backward(
    const xsigma::Tensor& grad,
    const xsigma::Tensor& input,
    const xsigma::Scalar& beta,
    const xsigma::Scalar& threshold);
std::tuple<xsigma::Tensor, xsigma::Tensor> slogdet_jvp(
    const xsigma::Tensor& LU,
    const xsigma::Tensor& pivots,
    const xsigma::Tensor& dA,
    const xsigma::Tensor& sign,
    const bool            use_A_T);
xsigma::Tensor slogdet_backward(
    const xsigma::Tensor& grad_sign,
    const xsigma::Tensor& grad_logabsdet,
    const xsigma::Tensor& A,
    const xsigma::Tensor& signdet,
    const xsigma::Tensor& LU,
    const xsigma::Tensor& pivots);
xsigma::Tensor log1p_backward(const xsigma::Tensor& grad, const xsigma::Tensor& self);
xsigma::Tensor sinc_backward(const xsigma::Tensor& grad, const xsigma::Tensor& self);
xsigma::Tensor sparse_constructor_values_backward(
    const xsigma::Tensor& sparse_grad_out, const xsigma::Tensor& indices);
xsigma::Tensor embedding_dense_double_backward_symint(
    const xsigma::Tensor& grad, const xsigma::Tensor& indices, const xsigma::SymInt& padding_idx);
xsigma::Tensor index_backward(
    xsigma::Tensor                            zeros_like_self,
    const torch::List<std::optional<Tensor>>& indices,
    const xsigma::Tensor&                     grad);
xsigma::Tensor _cudnn_ctc_loss_backward(
    const xsigma::Tensor& grad_out,
    const xsigma::Tensor& loss,
    const xsigma::Tensor& raw_grad,
    bool                  zero_infinity);
xsigma::Tensor elu_double_backward(
    const Tensor& grad,
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool          is_result,
    const Tensor& self_or_result);

Tensor svd_backward(
    const Tensor& gU,
    const Tensor& gS,
    const Tensor& gVh,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh);

std::tuple<Tensor, Tensor, Tensor> linalg_svd_jvp(
    const Tensor& dA, const Tensor& U, const Tensor& S, const Tensor& Vh, const bool full_matrices);
Tensor slice_backward_wrapper(
    const xsigma::Tensor&         grad,
    const xsigma::SymIntArrayRef& input_sizes,
    int64_t                       dim,
    std::optional<xsigma::SymInt> start,
    std::optional<xsigma::SymInt> end,
    xsigma::SymInt                step);
std::tuple<Tensor, Tensor> linalg_eig_jvp(
    const Tensor& dA, const Tensor& L, const Tensor& V, const bool is_hermitian);
Tensor linalg_eig_backward(
    const Tensor& gL,
    const Tensor& gV,
    const Tensor& L,
    const Tensor& V,
    const bool    is_hermitian,
    const bool    symeig_eigenvectors = true);
Tensor linalg_lstsq_solution_jvp(
    const Tensor& A, const Tensor& B_, const Tensor& dA, const Tensor& dB_);
Tensor linalg_lstsq_residuals_jvp(
    const Tensor& A,
    const Tensor& B_,
    const Tensor& dA,
    const Tensor& dB_,
    const Tensor& X_,
    const Tensor& L);
std::tuple<Tensor, Tensor> triangular_solve_backward(
    const Tensor&       grad_x,
    const Tensor&       grad_m,
    const Tensor&       b,
    const Tensor&       a,
    const Tensor&       x,
    const bool          upper,
    const bool          transpose,
    const bool          unitriangular,
    std::array<bool, 2> output_mask);
Tensor triangular_solve_jvp(
    const Tensor& X,
    const Tensor& A,
    const Tensor& dA,
    const Tensor& dB,
    const bool    upper,
    const bool    transpose,
    const bool    unitriangular);
Tensor linalg_solve_triangular_forward_AD(
    const Tensor& A_t,
    const Tensor& B_t,
    const Tensor& A,
    const Tensor& X,
    const bool    upper,
    const bool    left,
    const bool    unitriangular);
std::tuple<Tensor, Tensor> linalg_solve_triangular_backward(
    const Tensor&       grad,
    const Tensor&       A,
    const Tensor&       X,
    const bool          upper,
    const bool          left,
    const bool          unitriangular,
    std::array<bool, 2> output_mask);
std::tuple<Tensor, Tensor, Tensor> _trilinear_backward(
    const Tensor&                grad_out,
    const std::optional<Tensor>& i1,
    const std::optional<Tensor>& i2,
    const std::optional<Tensor>& i3,
    IntArrayRef                  expand1,
    IntArrayRef                  expand2,
    IntArrayRef                  expand3,
    IntArrayRef                  sumdim,
    std::array<bool, 3>          grad_mask);
std::tuple<Tensor, Tensor> linalg_qr_jvp(
    const Tensor& dA, const Tensor& Q, const Tensor& R, const std::string_view mode);
Tensor linalg_qr_backward(
    const Tensor&          gQ,
    const Tensor&          gR,
    const Tensor&          Q,
    const Tensor&          R,
    const std::string_view mode);
Tensor linalg_matrix_exp_differential(const Tensor& self, const Tensor& grad, bool adjoint);
std::tuple<Tensor, Tensor, Tensor> batchnorm_double_backward(
    const Tensor&                input,
    const std::optional<Tensor>& gamma,
    const Tensor&                ggI,
    const Tensor&                ggG,
    const Tensor&                ggB,
    const Tensor&                gO,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    bool                         training,
    double                       eps,
    const std::optional<Tensor>& save_mean,
    const std::optional<Tensor>& save_invstd,
    std::array<bool, 3>          output_mask);
std::tuple<Tensor, Tensor> _euclidean_dist_backward(
    const Tensor& grad, const Tensor& x1, const Tensor& x2, const Tensor& res);
Tensor fft_backward(
    const Tensor& self,
    const Tensor& grad,
    int64_t       signal_ndim,
    bool          complex_input,
    bool          complex_output,
    bool          inverse,
    IntArrayRef   checked_signal_sizes,
    int64_t       normalization,
    bool          onesided,
    IntArrayRef   output_sizes);
Tensor fft_r2c_backward(
    const Tensor&         grad,
    xsigma::IntArrayRef   dim,
    int64_t               normalization,
    bool                  onesided,
    const xsigma::SymInt& last_dim_size);
Tensor fft_c2r_backward(const Tensor& grad, IntArrayRef dim, int64_t normalization);
Tensor constant_pad_nd_backward(const Tensor& grad, xsigma::SymIntArrayRef pad);
std::tuple<Tensor, Tensor> cholesky_solve_backward(
    const Tensor&       grad_x,
    const Tensor&       self,
    const Tensor&       input2,
    const Tensor&       result,
    const bool          upper,
    std::array<bool, 2> output_mask);
Tensor cholesky_solve_jvp(
    const Tensor& X, const Tensor& U, const Tensor& dU, const Tensor& dB, const bool upper);
std::tuple<Tensor, Tensor, Tensor> infinitely_differentiable_native_group_norm_backward(
    const Tensor&                dY,
    const Tensor&                dmean,
    const Tensor&                drstd,
    const Tensor&                X,
    const Tensor&                mean,
    const Tensor&                rstd,
    const std::optional<Tensor>& gamma,
    xsigma::SymInt               N,
    const xsigma::SymInt&        C,
    xsigma::SymInt               HxW,
    int64_t                      group,
    double                       eps,
    std::array<bool, 3>          grad_input_mask);
Tensor gelu_double_backward(
    const Tensor& ggI, const Tensor& gO, const Tensor& input, std::string_view approximate);
Tensor as_strided_backward(
    Tensor                               grad,
    const TensorGeometry&                input_geometry,
    xsigma::SymIntArrayRef               sizes,
    xsigma::SymIntArrayRef               strides,
    const std::optional<xsigma::SymInt>& storage_offset_);
Tensor as_strided_scatter_backward(
    const Tensor&                 grad,
    const TensorGeometry&         input_geometry,
    const TensorGeometry&         src_geometry,
    xsigma::SymIntArrayRef        sizes,
    xsigma::SymIntArrayRef        strides,
    std::optional<xsigma::SymInt> storage_offset);
std::tuple<Tensor, Tensor> atan2_backward(
    const Tensor& grad, const Tensor& self, const Tensor& other, std::array<bool, 2> output_mask);
Tensor amaxamin_jvp(
    const Tensor& x, const Tensor& dx, const Tensor& result, IntArrayRef dim, bool keepdim);
std::tuple<Tensor, Tensor, Tensor> layer_norm_double_backward(
    const Tensor&                input,
    const std::optional<Tensor>& gamma,
    const Tensor&                ggI,
    const Tensor&                ggG,
    const Tensor&                ggB,
    const Tensor&                gO,
    const Tensor&                save_mean,
    const Tensor&                save_invstd,
    xsigma::SymIntArrayRef       normalized_shape,
    std::array<bool, 3>          output_mask);

std::tuple<Tensor, Tensor> infinitely_differentiable_native_rms_norm_backward(
    const Tensor&                dY,
    const Tensor&                drstd,
    const Tensor&                input,
    IntArrayRef                  normalized_shape,
    const Tensor&                rstd,
    const std::optional<Tensor>& weight_opt,
    std::array<bool, 2>          grad_input_mask);

std::tuple<Tensor, Tensor> householder_product_backward(
    const Tensor& grad,
    const Tensor& result,
    const Tensor& input,
    const Tensor& tau,
    const bool    flip_order = false);
Tensor householder_product_jvp(
    const Tensor& dV, const Tensor& dtau, const Tensor& prod, const Tensor& V, const Tensor& tau);
std::tuple<Tensor, Tensor, Tensor> ormqr_backward(
    const Tensor&       grad,
    const Tensor&       result,
    const Tensor&       self,
    const Tensor&       tau,
    const Tensor&       other,
    bool                left,
    bool                transpose,
    std::array<bool, 3> grad_output_mask);
std::tuple<Tensor, Tensor> polar_backward(const Tensor& grad, const Tensor& result);
Tensor i1_backward(const Tensor& grad, const Tensor& self, const Tensor& result);
Tensor i1e_backward(const Tensor& grad, const Tensor& self, const Tensor& result);
Tensor linalg_lu_solve_LU(
    const Tensor& grad,
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& X,
    const bool    left,
    const bool    adjoint);
Tensor linalg_lu_solve_jvp(
    const Tensor& X,
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& dLU,
    const Tensor& dB,
    const bool    left,
    const bool    adjoint);
std::tuple<Tensor, Tensor> linalg_solve_backward(
    const Tensor& gX,
    const Tensor& X,
    const Tensor& A,
    const Tensor& LU,
    const Tensor& pivots,
    const bool    left,
    const bool    B_requires_grad);
Tensor linalg_solve_jvp(
    const Tensor& dA,
    const Tensor& dB,
    const Tensor& X,
    const Tensor& LU,
    const Tensor& pivots,
    const bool    left,
    const bool    use_A_T);
Tensor lu_unpack_backward(
    const Tensor& L_grad, const Tensor& U_grad, const xsigma::SymInt& m, const xsigma::SymInt& n);

Tensor linalg_det_backward(
    const Tensor& grad, const Tensor& det, const Tensor& A, const Tensor& LU, const Tensor& pivots);
Tensor linalg_det_jvp(
    const Tensor& dA,
    const Tensor& det,
    const Tensor& LU,
    const Tensor& pivots,
    const bool    use_A_T);
std::tuple<Tensor, Tensor> linalg_lstsq_backward(
    const Tensor&              gX_,
    const Tensor&              gL,
    const Tensor&              A,
    const Tensor&              B_,
    const Tensor&              X_,
    const std::array<bool, 2>& grad_input_mask);
Tensor linalg_lu_backward(
    const Tensor& L_grad,
    const Tensor& U_grad,
    const Tensor& P,
    const Tensor& L,
    const Tensor& U,
    const bool    pivot);

std::tuple<Tensor, Tensor> linalg_lu_jvp(
    const Tensor& dA, const Tensor& P, const Tensor& L, const Tensor& U, const bool pivot);

Tensor lu_factor_ex_backward(
    const Tensor& grad, const Tensor& LU, const Tensor& pivs, const bool pivot);
Tensor lu_factor_ex_jvp(const Tensor& dX, const Tensor& LU, const Tensor& pivs, const bool pivot);

Tensor batch_norm_jvp(
    const Tensor&                input_p,
    const Tensor&                input_t,
    const Tensor&                weight_p,
    const Tensor&                weight_t,
    const Tensor&                bias_p,
    const Tensor&                bias_t,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    const Tensor&                saved_mean,
    const Tensor&                saved_invstd,
    bool                         train,
    double                       eps);

Tensor layer_norm_jvp(
    const Tensor&          input_p,
    const Tensor&          input_t,
    const Tensor&          weight_p,
    const Tensor&          weight_t,
    const Tensor&          bias_p,
    const Tensor&          bias_t,
    const Tensor&          saved_mean,
    const Tensor&          saved_invstd,
    xsigma::SymIntArrayRef normalized_shape);

Tensor rms_norm_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& saved_rstd,
    IntArrayRef   normalized_shape);

Tensor rms_norm_rstd_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& saved_rstd,
    IntArrayRef   normalized_shape);

Tensor group_norm_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t,
    const Tensor& saved_mean,
    const Tensor& saved_invstd,
    int64_t       groups);
Tensor group_norm_mean_jvp(const Tensor& input_t, const Tensor& mean_p, int64_t groups);
Tensor group_norm_invstd_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& mean_p,
    const Tensor& invstd_p,
    int64_t       groups);

Tensor convolution_jvp(
    const Tensor&          input_p,
    const Tensor&          input_t,
    const Tensor&          weight_p,
    const Tensor&          weight_t,
    const Tensor&          bias_p,
    const Tensor&          bias_t,
    xsigma::SymIntArrayRef stride,
    xsigma::SymIntArrayRef padding,
    xsigma::SymIntArrayRef dilation,
    bool                   transposed,
    xsigma::SymIntArrayRef output_padding,
    const xsigma::SymInt&  groups);

Tensor _convolution_jvp(
    const Tensor&          input_p,
    const Tensor&          input_t,
    const Tensor&          weight_p,
    const Tensor&          weight_t,
    const Tensor&          bias_p,
    const Tensor&          bias_t,
    xsigma::SymIntArrayRef stride,
    xsigma::SymIntArrayRef padding,
    xsigma::SymIntArrayRef dilation,
    bool                   transposed,
    xsigma::SymIntArrayRef output_padding,
    const xsigma::SymInt&  groups,
    bool                   benchmark,
    bool                   deterministic,
    bool                   cudnn_enabled,
    bool                   allow_tf32);

Tensor convolution_backward_jvp_grad_bias(const Tensor& grad_out_t, const Tensor& grad_bias);

Tensor cat_jvp(const xsigma::ITensorListRef& tensors, int64_t dim);
Tensor block_diag_jvp(xsigma::TensorList tensors);
Tensor stack_jvp(xsigma::TensorList tensors, int64_t dim);
Tensor cumprod_jvp(const Tensor& self_t, const Tensor& self_p, const Tensor& result, int dim);
Tensor gather_with_keepdimed_indices(
    const Tensor& input, int64_t dim, const Tensor& indices, bool keepdim);
Tensor evenly_read_jvp(const Tensor& fw_grad, const Tensor& input, const Tensor& value);
Tensor warn_backwards(const Tensor& grad_output);

std::tuple<Tensor, Tensor> _cudnn_convolution_backward(
    const xsigma::Tensor&  self,
    const xsigma::Tensor&  grad_output,
    const xsigma::Tensor&  weight,
    xsigma::SymIntArrayRef padding,
    xsigma::SymIntArrayRef output_padding,
    xsigma::SymIntArrayRef stride,
    xsigma::SymIntArrayRef dilation,
    bool                   transposed,
    xsigma::SymInt         groups,
    ::std::array<bool, 2>  output_mask);

Tensor scatter_reduce_jvp(
    const Tensor&    self_p,
    const Tensor&    self_t,
    int              dim,
    const Tensor&    index,
    const Tensor&    src_p,
    const Tensor&    src_t,
    std::string_view reduce,
    bool             include_self,
    const Tensor&    result);

std::tuple<Tensor, Tensor> scatter_reduce_backward(
    const Tensor&    grad,
    const Tensor&    self,
    int              dim,
    const Tensor&    index,
    const Tensor&    src,
    std::string_view reduce,
    bool             include_self,
    const Tensor&    result);

Tensor _to_copy_backward(const Tensor& grad, const xsigma::TensorOptions& self_options);

std::tuple<Tensor, Tensor> index_reduce_backward(
    const Tensor&    grad,
    const Tensor&    self,
    int              dim,
    const Tensor&    index,
    const Tensor&    source,
    std::string_view reduce,
    bool             include_self,
    const Tensor&    result);

Tensor take_backward(const Tensor& grad, const Tensor& self, const Tensor& indices);

Tensor to_sparse_backward(
    const Tensor&                                   grad,
    const xsigma::Layout                            self_layout,
    const xsigma::OptionalArrayRef<xsigma::SymInt>& self_blocksize);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
mkldnn_rnn_layer_differentiable_backward(
    const Tensor&                input,
    const Tensor&                weight0,
    const Tensor&                weight1,
    const Tensor&                weight2,
    const Tensor&                weight3,
    const Tensor&                hx_,
    const Tensor&                cx_tmp,
    const Tensor&                output,
    const Tensor&                hy_,
    const Tensor&                cy_,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    bool                         reverse,
    int64_t                      mode,
    int64_t                      hidden_size,
    int64_t                      num_layers,
    bool                         has_biases,
    bool                         train,
    bool                         bidirectional,
    xsigma::IntArrayRef          batch_sizes,
    bool                         batch_first,
    const xsigma::Tensor&        workspace);

Tensor values_backward(const Tensor& grad, const Tensor& self);

}  // namespace torch::autograd::generated::details
