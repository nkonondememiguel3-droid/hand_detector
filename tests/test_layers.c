// tests/test_layers.c
#include "ds_arena.h"
#include "layers.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <math.h>

static _ds_arena_t_ pa; // param arena
static _ds_arena_t_ ba; // batch arena

void layers_setup( void )
{
  pa = ds_arena_new( 0 );
  ba = ds_arena_new( 0 );
}

void layers_teardown( void )
{
  ds_arena_destroy( &ba );
  ds_arena_destroy( &pa );
}

TestSuite( layers, .init = layers_setup, .fini = layers_teardown );

// Conv2D — Shape
Test( layers, conv2d_output_shape_stride1_nopad )
{
  // 8x8 -> kernel 3, stride 1, pad 0 -> 6x6
  int in_shape[] = { 1, 3, 8, 8 };
  _tensor_t *input = tensor_zeros( &ba, 4, in_shape );

  _layer_t *conv = conv2d_create( &pa, 3, 16, 3, 1, 0 );
  _tensor_t *out = conv->forward( conv, &ba, input );

  cr_assert_eq( out->shape[0], 1 );
  cr_assert_eq( out->shape[1], 16 );
  cr_assert_eq( out->shape[2], 6 );
  cr_assert_eq( out->shape[3], 6 );
}

Test( layers, conv2d_output_shape_stride2_pad1 )
{
  // H_out = (256 + 2 - 3)/2 + 1 = 128
  int in_shape[] = { 1, 3, 256, 256 };
  _tensor_t *input = tensor_zeros( &ba, 4, in_shape );

  _layer_t *conv = conv2d_create( &pa, 3, 16, 3, 2, 1 );
  _tensor_t *out = conv->forward( conv, &ba, input );

  cr_assert_eq( out->shape[2], 128 );
  cr_assert_eq( out->shape[3], 128 );
}

// Conv2D — Known values
Test( layers, conv2d_all_ones_kernel_known_output )
{
  // 1x1x3x3 input of ones, kernel all 1's, bias 0 -> single output = 9
  int in_shape[] = { 1, 1, 3, 3 };
  _tensor_t *input = tensor_ones( &ba, 4, in_shape );

  _layer_t *conv = conv2d_create( &pa, 1, 1, 3, 1, 0 );
  for ( int i = 0; i < conv->weights->size; i++ ) conv->weights->data[i] = 1.0f;
  conv->bias->data[0] = 0.0f;

  _tensor_t *out = conv->forward( conv, &ba, input );

  cr_assert_eq( out->shape[2], 1 );
  cr_assert_eq( out->shape[3], 1 );
  cr_assert_float_eq( out->data[0], 9.0f, 1e-5f );
}

Test( layers, conv2d_1x1_identity_kernel_passes_through )
{
  int in_shape[] = { 1, 1, 4, 4 };
  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

  _layer_t *conv = conv2d_create( &pa, 1, 1, 1, 1, 0 );
  conv->weights->data[0] = 1.0f;
  conv->bias->data[0] = 0.0f;

  _tensor_t *out = conv->forward( conv, &ba, input );

  for ( int i = 0; i < input->size; i++ ) cr_assert_float_eq( out->data[i], input->data[i], 1e-5f );
}

// Conv2D — Gradient checks
static float sum_squares( _tensor_t *t )
{
  float s = 0.0f;
  for ( int i = 0; i < t->size; i++ ) s += t->data[i] * t->data[i];
  return s;
}

Test( layers, conv2d_weight_gradient_matches_numerical )
{
  int in_shape[] = { 1, 1, 4, 4 };
  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

  _layer_t *conv = conv2d_create( &pa, 1, 1, 3, 1, 0 );

  _tensor_t *out = conv->forward( conv, &ba, input );
  _tensor_t *grad_out = tensor_zeros( &ba, out->ndim, out->shape );
  for ( int i = 0; i < out->size; i++ ) grad_out->data[i] = 2.0f * out->data[i];

  tensor_zero_gradient( conv->weights );
  conv->backward( conv, &ba, grad_out );

  float eps = 1e-3f;
  float w0 = conv->weights->data[0];

  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ba );

  conv->weights->data[0] = w0 + eps;
  _tensor_t *out_plus = conv->forward( conv, &ba, input );
  float loss_plus = sum_squares( out_plus );
  ds_arena_reset_to( &ba, cp );

  conv->weights->data[0] = w0 - eps;
  _tensor_t *out_minus = conv->forward( conv, &ba, input );
  float loss_minus = sum_squares( out_minus );
  ds_arena_reset_to( &ba, cp );

  conv->weights->data[0] = w0; // restore

  float numerical = ( loss_plus - loss_minus ) / ( 2 * eps );
  float analytical = conv->weights->gradients[0];

  cr_assert( fabsf( numerical - analytical ) < 1e-2f, "Gradient mismatch: numerical=%.5f analytical=%.5f", numerical, analytical );
}

Test( layers, conv2d_bias_gradient_is_sum_of_output_grad )
{
  int in_shape[] = { 1, 1, 4, 4 };
  _tensor_t *input = tensor_ones( &ba, 4, in_shape );

  _layer_t *conv = conv2d_create( &pa, 1, 1, 3, 1, 0 ); // output 2x2 = 4 elems

  _tensor_t *out = conv->forward( conv, &ba, input );
  _tensor_t *grad_out = tensor_ones( &ba, out->ndim, out->shape );

  tensor_zero_gradient( conv->bias );
  conv->backward( conv, &ba, grad_out );

  cr_assert_float_eq( conv->bias->gradients[0], 4.0f, 1e-5f );
}

// ReLU
Test( layers, relu_zeroes_negative_values )
{
  int shape[] = { 1, 5 };
  _tensor_t *input = tensor_zeros( &ba, 2, shape );
  float vals[] = { -2.0f, -0.5f, 0.0f, 0.5f, 2.0f };
  for ( int i = 0; i < 5; i++ ) input->data[i] = vals[i];

  _layer_t *relu = relu_create( &pa );
  _tensor_t *out = relu->forward( relu, &ba, input );

  float expected[] = { 0.0f, 0.0f, 0.0f, 0.5f, 2.0f };
  for ( int i = 0; i < 5; i++ ) cr_assert_float_eq( out->data[i], expected[i], 1e-6f );
}

Test( layers, relu_gradient_blocks_negative_inputs )
{
  int shape[] = { 1, 4 };
  _tensor_t *input = tensor_zeros( &ba, 2, shape );
  input->data[0] = -1.0f;
  input->data[1] = 1.0f;
  input->data[2] = -3.0f;
  input->data[3] = 2.0f;

  _layer_t *relu = relu_create( &pa );
  relu->forward( relu, &ba, input );

  _tensor_t *grad_out = tensor_ones( &ba, 2, shape );
  _tensor_t *grad_in = relu->backward( relu, &ba, grad_out );

  cr_assert_float_eq( grad_in->data[0], 0.0f, 1e-6f );
  cr_assert_float_eq( grad_in->data[1], 1.0f, 1e-6f );
  cr_assert_float_eq( grad_in->data[2], 0.0f, 1e-6f );
  cr_assert_float_eq( grad_in->data[3], 1.0f, 1e-6f );
}

// BatchNorm
Test( layers, batchnorm_normalizes_to_zero_mean_unit_var )
{
  int shape[] = { 4, 2, 3, 3 }; // N=4, C=2, H=3, W=3
  _tensor_t *input = tensor_random_normal( &ba, 4, shape, 5.0f, 2.0f );

  _layer_t *bn = batchnorm_create( &pa, 2 );
  _tensor_t *out = bn->forward( bn, &ba, input );

  int N = 4, C = 2, H = 3, W = 3, M = N * H * W;

  float mean = 0.0f;
  for ( int n = 0; n < N; n++ )
    for ( int h = 0; h < H; h++ )
      for ( int w = 0; w < W; w++ ) mean += out->data[( ( n * C + 0 ) * H + h ) * W + w];
  mean /= (float)M;

  cr_assert( fabsf( mean ) < 1e-4f, "Mean %.6f should be ~0", mean );

  float var = 0.0f;
  for ( int n = 0; n < N; n++ )
    for ( int h = 0; h < H; h++ )
      for ( int w = 0; w < W; w++ )
      {
        float d = out->data[( ( n * C + 0 ) * H + h ) * W + w] - mean;
        var += d * d;
      }
  var /= (float)M;

  cr_assert( fabsf( var - 1.0f ) < 1e-3f, "Var %.6f should be ~1", var );
}

Test( layers, batchnorm_gamma_beta_scale_and_shift )
{
  int shape[] = { 4, 1, 2, 2 };
  _tensor_t *input = tensor_random_normal( &ba, 4, shape, 0.0f, 1.0f );

  _layer_t *bn = batchnorm_create( &pa, 1 );
  bn->gamma->data[0] = 2.0f;
  bn->beta->data[0] = 3.0f;

  _tensor_t *out = bn->forward( bn, &ba, input );

  float mean = 0.0f;
  for ( int i = 0; i < out->size; i++ ) mean += out->data[i];
  mean /= (float)out->size;

  cr_assert( fabsf( mean - 3.0f ) < 1e-3f, "Output mean %.4f should be ~beta(3)", mean );
}

Test( layers, batchnorm_running_stats_update_with_momentum )
{
  int shape[] = { 2, 1, 2, 2 };
  _tensor_t *input = tensor_ones( &ba, 4, shape ); // mean=1, var=0

  _layer_t *bn = batchnorm_create( &pa, 1 );
  bn->bn_momentum = 0.1f;
  // running_mean starts 0, running_var starts 1

  bn->forward( bn, &ba, input );

  cr_assert_float_eq( bn->running_mean->data[0], 0.1f, 1e-5f ); // 0.9*0+0.1*1
  cr_assert_float_eq( bn->running_var->data[0], 0.9f, 1e-5f );  // 0.9*1+0.1*0
}

Test( layers, batchnorm_gamma_gradient_matches_numerical )
{
  int shape[] = { 4, 1, 2, 2 };
  _tensor_t *input = tensor_random_normal( &ba, 4, shape, 0.0f, 1.0f );

  _layer_t *bn = batchnorm_create( &pa, 1 );
  bn->gamma->data[0] = 1.5f;

  _tensor_t *out = bn->forward( bn, &ba, input );
  _tensor_t *grad_out = tensor_zeros( &ba, out->ndim, out->shape );
  for ( int i = 0; i < out->size; i++ ) grad_out->data[i] = 2.0f * out->data[i];

  tensor_zero_gradient( bn->gamma );
  bn->backward( bn, &ba, grad_out );

  float eps = 1e-3f;
  float g0 = bn->gamma->data[0];

  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ba );

  bn->gamma->data[0] = g0 + eps;
  _tensor_t *out_plus = bn->forward( bn, &ba, input );
  float loss_plus = sum_squares( out_plus );
  ds_arena_reset_to( &ba, cp );

  bn->gamma->data[0] = g0 - eps;
  _tensor_t *out_minus = bn->forward( bn, &ba, input );
  float loss_minus = sum_squares( out_minus );
  ds_arena_reset_to( &ba, cp );

  bn->gamma->data[0] = g0;

  float numerical = ( loss_plus - loss_minus ) / ( 2 * eps );
  float analytical = bn->gamma->gradients[0];

  cr_assert( fabsf( numerical - analytical ) < 1e-1f, "Gamma grad mismatch: numerical=%.4f analytical=%.4f", numerical, analytical );
}

// Dense
Test( layers, dense_output_shape )
{
  int in_shape[] = { 4, 128 };
  _tensor_t *input = tensor_zeros( &ba, 2, in_shape );

  _layer_t *dense = dense_create( &pa, 128, 64 );
  _tensor_t *out = dense->forward( dense, &ba, input );

  cr_assert_eq( out->shape[0], 4 );
  cr_assert_eq( out->shape[1], 64 );
}

Test( layers, dense_known_value_matmul )
{
  // in=2, out=1; w=[2,3], b=1; input=[1,1] -> out = 2+3+1 = 6
  int in_shape[] = { 1, 2 };
  _tensor_t *input = tensor_ones( &ba, 2, in_shape );

  _layer_t *dense = dense_create( &pa, 2, 1 );
  dense->weights->data[0] = 2.0f;
  dense->weights->data[1] = 3.0f;
  dense->bias->data[0] = 1.0f;

  _tensor_t *out = dense->forward( dense, &ba, input );

  cr_assert_float_eq( out->data[0], 6.0f, 1e-5f );
}

Test( layers, dense_weight_gradient_is_outer_product )
{
  // input=[2,3], grad_out=[1] -> grad_weight = [2,3]
  int in_shape[] = { 1, 2 };
  _tensor_t *input = tensor_zeros( &ba, 2, in_shape );
  input->data[0] = 2.0f;
  input->data[1] = 3.0f;

  _layer_t *dense = dense_create( &pa, 2, 1 );
  dense->forward( dense, &ba, input );

  int go_shape[] = { 1, 1 };
  _tensor_t *grad_out = tensor_ones( &ba, 2, go_shape );

  tensor_zero_gradient( dense->weights );
  dense->backward( dense, &ba, grad_out );

  cr_assert_float_eq( dense->weights->gradients[0], 2.0f, 1e-5f );
  cr_assert_float_eq( dense->weights->gradients[1], 3.0f, 1e-5f );
}

// Global Average Pooling
Test( layers, gap_output_shape_and_value )
{
  // 1 sample, 2 channels, 2x2; channel0 all 2's -> avg 2, channel1 all 4's -> avg 4
  int shape[] = { 1, 2, 2, 2 };
  _tensor_t *input = tensor_zeros( &ba, 4, shape );
  for ( int i = 0; i < 4; i++ ) input->data[i] = 2.0f;
  for ( int i = 4; i < 8; i++ ) input->data[i] = 4.0f;

  _layer_t *gap = gap_create( &pa );
  _tensor_t *out = gap->forward( gap, &ba, input );

  cr_assert_eq( out->shape[0], 1 );
  cr_assert_eq( out->shape[1], 2 );
  cr_assert_float_eq( out->data[0], 2.0f, 1e-5f );
  cr_assert_float_eq( out->data[1], 4.0f, 1e-5f );
}

Test( layers, gap_gradient_distributed_equally )
{
  int shape[] = { 1, 1, 2, 2 };
  _tensor_t *input = tensor_zeros( &ba, 4, shape );

  _layer_t *gap = gap_create( &pa );
  gap->forward( gap, &ba, input );

  int go_shape[] = { 1, 1 };
  _tensor_t *grad_out = tensor_ones( &ba, 2, go_shape );

  _tensor_t *grad_in = gap->backward( gap, &ba, grad_out );

  for ( int i = 0; i < 4; i++ ) cr_assert_float_eq( grad_in->data[i], 0.25f, 1e-5f );
}

// End-to-end pipeline
Test( layers, pipeline_conv_relu_gap_dense_shapes )
{
  int in_shape[] = { 2, 3, 8, 8 };
  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

  _layer_t *conv = conv2d_create( &pa, 3, 8, 3, 1, 1 ); // -> (2,8,8,8)
  _layer_t *relu = relu_create( &pa );
  _layer_t *gap = gap_create( &pa );           // -> (2,8)
  _layer_t *dense = dense_create( &pa, 8, 5 ); // -> (2,5)

  _tensor_t *t1 = conv->forward( conv, &ba, input );
  cr_assert_eq( t1->shape[1], 8 );
  cr_assert_eq( t1->shape[2], 8 );

  _tensor_t *t2 = relu->forward( relu, &ba, t1 );
  _tensor_t *t3 = gap->forward( gap, &ba, t2 );
  cr_assert_eq( t3->shape[0], 2 );
  cr_assert_eq( t3->shape[1], 8 );

  _tensor_t *t4 = dense->forward( dense, &ba, t3 );
  cr_assert_eq( t4->shape[0], 2 );
  cr_assert_eq( t4->shape[1], 5 );
}

// Arena reset after backward — full step pattern
Test( layers, batch_arena_reset_after_full_step )
{
  _layer_t *conv = conv2d_create( &pa, 1, 1, 3, 1, 1 ); // params in pa

  int in_shape[] = { 1, 1, 4, 4 };

  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ba );

  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );
  _tensor_t *out = conv->forward( conv, &ba, input );

  _tensor_t *grad_out = tensor_ones( &ba, out->ndim, out->shape );
  tensor_zero_gradient( conv->weights );
  conv->backward( conv, &ba, grad_out );

  // Gradients landed in pa-allocated weight tensor — survive reset
  float w_grad_sample = conv->weights->gradients[0];

  ds_arena_reset_to( &ba, cp );

  // batch_arena reclaimed...
  size_t used_after = ba.head ? ba.head->chunk_size_used : 0;
  cr_assert_eq( used_after, cp.checkpoint_size_used );

  // ...but param_arena data (gradients) untouched
  cr_assert_float_eq( conv->weights->gradients[0], w_grad_sample, 1e-7f );
}
