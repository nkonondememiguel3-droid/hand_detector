#include "bn_fold.h"
#include "ds_arena.h"
#include "layers.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <math.h>

static _ds_arena_t_ pa;
static _ds_arena_t_ ba;

void bnfold_setup( void )
{
  pa = ds_arena_new( 0 );
  ba = ds_arena_new( 0 );
}

void bnfold_teardown( void )
{
  ds_arena_destroy( &ba );
  ds_arena_destroy( &pa );
}

TestSuite( bn_fold, .init = bnfold_setup, .fini = bnfold_teardown );

static float sum_abs( _tensor_t *t )
{
  float s = 0.0f;
  for ( int i = 0; i < t->size; i++ ) s += fabsf( t->data[i] );
  return s;
}

// Folding produces identical output to conv->bn chain
Test( bn_fold, folded_conv_matches_conv_then_bn_inference )
{
  // Build conv -> bn, run forward (training mode, populates running stats
  // after one batch), then fold and compare folded-conv-only output
  // against the original conv+bn output using the SAME running stats.

  int in_shape[] = { 2, 1, 4, 4 };
  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

  _layer_t *conv = conv2d_create( &pa, 1, 2, 3, 1, 1 ); // -> (2,2,4,4)
  _layer_t *bn = batchnorm_create( &pa, 2 );

  // Give BN non-trivial running stats and affine params, as if trained
  bn->running_mean->data[0] = 0.3f;
  bn->running_mean->data[1] = -0.2f;
  bn->running_var->data[0] = 1.5f;
  bn->running_var->data[1] = 0.8f;
  bn->gamma->data[0] = 1.2f;
  bn->gamma->data[1] = 0.9f;
  bn->beta->data[0] = 0.1f;
  bn->beta->data[1] = -0.05f;

  // Reference: conv -> bn, using running stats directly (simulate
  //    inference mode by overwriting batch_mean/batch_var with running
  //    stats before calling bn_forward, since our bn_forward always
  //    computes batch stats — see note below)
  _tensor_t *conv_out = conv->forward( conv, &ba, input );

  // Manually compute "inference-mode" BN output using running stats,
  // since bn->forward (training mode) recomputes batch statistics.
  int N = conv_out->shape[0], C = conv_out->shape[1];
  int H = conv_out->shape[2], W = conv_out->shape[3];

  int ref_shape[] = { N, C, H, W };
  _tensor_t *reference = tensor_zeros( &ba, 4, ref_shape );

  for ( int c = 0; c < C; c++ )
  {
    float mean = bn->running_mean->data[c];
    float var = bn->running_var->data[c];
    float gamma = bn->gamma->data[c];
    float beta = bn->beta->data[c];
    float inv_std = 1.0f / sqrtf( var + bn->bn_eps );

    for ( int n = 0; n < N; n++ )
      for ( int h = 0; h < H; h++ )
        for ( int w = 0; w < W; w++ )
        {
          int idx = ( ( n * C + c ) * H + h ) * W + w;
          float xhat = ( conv_out->data[idx] - mean ) * inv_std;
          reference->data[idx] = gamma * xhat + beta;
        }
  }

  // Fold BN into conv, then run conv ALONE on the same input
  fold_batchnorm_into_conv( conv, bn );

  _tensor_t *folded_out = conv->forward( conv, &ba, input );

  // Compare
  for ( int i = 0; i < reference->size; i++ )
    cr_assert_float_eq( folded_out->data[i], reference->data[i], 1e-4f, "Mismatch at element %d: folded=%.6f reference=%.6f", i, folded_out->data[i],
                        reference->data[i] );
}

// Folding scales weights correctly per output channel
Test( bn_fold, weight_scale_applied_per_output_channel )
{
  int in_shape[] = { 1, 1, 3, 3 };
  _tensor_t *input = tensor_ones( &ba, 4, in_shape );

  _layer_t *conv = conv2d_create( &pa, 1, 2, 3, 1, 0 ); // out_channels=2, k=3 -> 1*3*3=9 weights/channel
  for ( int i = 0; i < conv->weights->size; i++ ) conv->weights->data[i] = 1.0f;
  conv->bias->data[0] = 0.0f;
  conv->bias->data[1] = 0.0f;

  _layer_t *bn = batchnorm_create( &pa, 2 );
  bn->running_mean->data[0] = 0.0f;
  bn->running_mean->data[1] = 0.0f;
  bn->running_var->data[0] = 1.0f;
  bn->running_var->data[1] = 1.0f;
  bn->gamma->data[0] = 2.0f; // scale[0] = 2/sqrt(1+eps) ~= 2.0
  bn->gamma->data[1] = 0.5f; // scale[1] = 0.5/sqrt(1+eps) ~= 0.5
  bn->beta->data[0] = 0.0f;
  bn->beta->data[1] = 0.0f;

  fold_batchnorm_into_conv( conv, bn );

  int kernel_size = 1 * 3 * 3; // C_in*k*k

  // Channel 0 weights should all be ~2.0 (1.0 * scale0)
  for ( int i = 0; i < kernel_size; i++ ) cr_assert_float_eq( conv->weights->data[0 * kernel_size + i], 2.0f, 1e-3f );

  // Channel 1 weights should all be ~0.5 (1.0 * scale1)
  for ( int i = 0; i < kernel_size; i++ ) cr_assert_float_eq( conv->weights->data[1 * kernel_size + i], 0.5f, 1e-3f );
}

// Folding adjusts bias correctly: b' = scale*(b-mean) + beta
Test( bn_fold, bias_fold_formula )
{
  int in_shape[] = { 1, 1, 1, 1 };
  _tensor_t *input = tensor_zeros( &ba, 4, in_shape );

  _layer_t *conv = conv2d_create( &pa, 1, 1, 1, 1, 0 );
  conv->weights->data[0] = 1.0f;
  conv->bias->data[0] = 3.0f; // b = 3

  _layer_t *bn = batchnorm_create( &pa, 1 );
  bn->running_mean->data[0] = 1.0f; // mean = 1
  bn->running_var->data[0] = 3.0f;  // var = 3
  bn->gamma->data[0] = 2.0f;        // gamma = 2
  bn->beta->data[0] = 0.5f;         // beta = 0.5
  bn->bn_eps = 0.0f;                // simplify: eps=0

  // scale = gamma/sqrt(var) = 2/sqrt(3)
  // b' = scale*(b-mean) + beta = (2/sqrt(3))*(3-1) + 0.5
  float scale = 2.0f / sqrtf( 3.0f );
  float expected_bias = scale * ( 3.0f - 1.0f ) + 0.5f;

  fold_batchnorm_into_conv( conv, bn );

  cr_assert_float_eq( conv->bias->data[0], expected_bias, 1e-5f );
}

// Zero gamma collapses channel to constant (beta)
Test( bn_fold, zero_gamma_collapses_channel_to_beta )
{
  int in_shape[] = { 1, 1, 2, 2 };
  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

  _layer_t *conv = conv2d_create( &pa, 1, 1, 1, 1, 0 );
  conv->weights->data[0] = 5.0f;
  conv->bias->data[0] = 2.0f;

  _layer_t *bn = batchnorm_create( &pa, 1 );
  bn->running_mean->data[0] = 1.0f;
  bn->running_var->data[0] = 4.0f;
  bn->gamma->data[0] = 0.0f; // gamma=0 -> scale=0
  bn->beta->data[0] = 7.0f;

  fold_batchnorm_into_conv( conv, bn );

  // scale=0 -> all weights become 0, bias becomes beta=7
  for ( int i = 0; i < conv->weights->size; i++ ) cr_assert_float_eq( conv->weights->data[i], 0.0f, 1e-6f );

  cr_assert_float_eq( conv->bias->data[0], 7.0f, 1e-5f );

  // Output should be constant 7.0 everywhere
  _tensor_t *out = conv->forward( conv, &ba, input );
  for ( int i = 0; i < out->size; i++ ) cr_assert_float_eq( out->data[i], 7.0f, 1e-5f );
}
