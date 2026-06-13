// tests/test_network.c
#include "ds_arena.h"
#include "loss.h"
#include "network.h"
#include "optimizer.h"
#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <math.h>

static _ds_arena_t_ pa; // param arena
static _ds_arena_t_ ba; // batch arena

void net_setup( void )
{
  pa = ds_arena_new( 0 );
  ba = ds_arena_new( 0 );
}

void net_teardown( void )
{
  ds_arena_destroy( &ba );
  ds_arena_destroy( &pa );
}

TestSuite( network, .init = net_setup, .fini = net_teardown );

// Forward chaining
Test( network, forward_chains_shapes_correctly )
{
  _network_t *net = network_create( &pa );
  network_add( net, conv2d_create( &pa, 3, 8, 3, 1, 1 ) ); // -> (N,8,H,W)
  network_add( net, relu_create( &pa ) );
  network_add( net, gap_create( &pa ) );         // -> (N,8)
  network_add( net, dense_create( &pa, 8, 5 ) ); // -> (N,5)

  int in_shape[] = { 2, 3, 8, 8 };
  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

  _tensor_t *out = network_forward( net, &ba, input );

  cr_assert_eq( out->shape[0], 2 );
  cr_assert_eq( out->shape[1], 5 );
}

// Backward populates gradients
Test( network, backward_populates_all_gradients )
{
  _network_t *net = network_create( &pa );
  network_add( net, conv2d_create( &pa, 1, 2, 3, 1, 1 ) ); // -> (N,2,H,W)
  network_add( net, relu_create( &pa ) );
  network_add( net, gap_create( &pa ) );         // -> (N,2)
  network_add( net, dense_create( &pa, 2, 1 ) ); // -> (N,1)

  int in_shape[] = { 1, 1, 4, 4 };
  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

  network_zero_gradient( net );
  network_forward( net, &ba, input );

  int go_shape[] = { 1, 1 };
  _tensor_t *grad_out = tensor_ones( &ba, 2, go_shape );

  network_backward( net, &ba, grad_out );

  // Conv weights (layer 0) should have non-zero gradients
  _layer_t *conv = net->layers[0];
  int any_nonzero = 0;
  for ( int i = 0; i < conv->weights->size; i++ )
    if ( fabsf( conv->weights->gradients[i] ) > 1e-8f ) any_nonzero = 1;
  cr_assert( any_nonzero, "Conv weight gradients should be non-zero" );

  // Dense weights (last layer) should also have gradients
  _layer_t *dense = net->layers[3];
  any_nonzero = 0;
  for ( int i = 0; i < dense->weights->size; i++ )
    if ( fabsf( dense->weights->gradients[i] ) > 1e-8f ) any_nonzero = 1;
  cr_assert( any_nonzero, "Dense weight gradients should be non-zero" );
}

Test( network, zero_grad_clears_all_layers )
{
  _network_t *net = network_create( &pa );
  network_add( net, conv2d_create( &pa, 1, 2, 3, 1, 1 ) );
  network_add( net, dense_create( &pa, 2, 1 ) );

  _layer_t *conv = net->layers[0];
  conv->weights->gradients[0] = 5.0f;
  conv->bias->gradients[0] = 3.0f;

  network_zero_gradient( net );

  cr_assert_float_eq( conv->weights->gradients[0], 0.0f, 1e-7f );
  cr_assert_float_eq( conv->bias->gradients[0], 0.0f, 1e-7f );
}

// Full forward+backward+reset cycle
Test( network, full_step_then_arena_reset )
{
  _network_t *net = network_create( &pa );
  network_add( net, conv2d_create( &pa, 1, 2, 3, 1, 1 ) );
  network_add( net, relu_create( &pa ) );
  network_add( net, gap_create( &pa ) );
  network_add( net, dense_create( &pa, 2, 5 ) ); // mimic detection head

  int in_shape[] = { 1, 1, 8, 8 };

  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ba );

  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

  network_zero_gradient( net );
  _tensor_t *out = network_forward( net, &ba, input );
  cr_assert_eq( out->shape[1], 5 );

  _tensor_t *grad_out = tensor_ones( &ba, out->ndim, out->shape );
  network_backward( net, &ba, grad_out );

  // Sanity: gradients landed in param_arena tensors
  _layer_t *dense = net->layers[3];
  cr_assert( fabsf( dense->weights->gradients[0] ) > 0.0f || fabsf( dense->bias->gradients[0] ) > 0.0f,
             "Dense layer should have received gradients" );

  ds_arena_reset_to( &ba, cp );

  size_t used_after = ba.head ? ba.head->chunk_size_used : 0;
  cr_assert_eq( used_after, cp.checkpoint_size_used );

  // Gradients survive the batch_arena reset (they live in param_arena)
  cr_assert( fabsf( dense->weights->gradients[0] ) > 0.0f || fabsf( dense->bias->gradients[0] ) > 0.0f,
             "Param gradients should survive batch_arena reset" );
}

// set_training toggles BatchNorm only
Test( network, set_training_toggles_batchnorm_only )
{
  _network_t *net = network_create( &pa );
  network_add( net, conv2d_create( &pa, 1, 1, 3, 1, 1 ) );
  network_add( net, batchnorm_create( &pa, 1 ) );
  network_add( net, relu_create( &pa ) );

  _layer_t *bn = net->layers[1];
  cr_assert_eq( bn->training, 1 ); // default

  network_set_training( net, 0 );
  cr_assert_eq( bn->training, 0 );

  network_set_training( net, 1 );
  cr_assert_eq( bn->training, 1 );
}

// combine_head_gradients layout
Test( network, combine_head_gradients_layout )
{
  int batch = 2;

  int conf_shape[] = { batch, 1 };
  _tensor_t *conf_grad = tensor_zeros( &ba, 2, conf_shape );
  conf_grad->data[0] = 0.1f;
  conf_grad->data[1] = 0.2f;

  int box_shape[] = { batch, 4 };
  _tensor_t *box_grad = tensor_zeros( &ba, 2, box_shape );
  for ( int i = 0; i < 8; i++ ) box_grad->data[i] = 1.0f;

  _tensor_t *combined = combiine_head_gradients( &ba, conf_grad, box_grad, batch, 5.0f );

  cr_assert_eq( combined->shape[0], 2 );
  cr_assert_eq( combined->shape[1], 5 );

  // Sample 0: [conf=0.1, box*5 = 5,5,5,5]
  cr_assert_float_eq( combined->data[0], 0.1f, 1e-6f );
  for ( int k = 1; k < 5; k++ ) cr_assert_float_eq( combined->data[k], 5.0f, 1e-6f );

  // Sample 1: [conf=0.2, box*5 = 5,5,5,5]
  cr_assert_float_eq( combined->data[5], 0.2f, 1e-6f );
  for ( int k = 1; k < 5; k++ ) cr_assert_float_eq( combined->data[5 + k], 5.0f, 1e-6f );
}

// Optimizer — Adam
Test( network, adam_step_moves_param_toward_negative_gradient )
{
  _network_t *net = network_create( &pa );
  network_add( net, dense_create( &pa, 2, 1 ) );

  _layer_t *dense = net->layers[0];
  dense->weights->data[0] = 1.0f;
  dense->weights->data[1] = 1.0f;
  dense->weights->gradients[0] = 1.0f;  // positive gradient
  dense->weights->gradients[1] = -1.0f; // negative gradient

  _adam_optimizer_t *opt = adam_create( &pa, net, 0.1f );
  adam_step( opt );

  cr_assert( dense->weights->data[0] < 1.0f, "Weight with positive grad should decrease, got %.5f", dense->weights->data[0] );

  cr_assert( dense->weights->data[1] > 1.0f, "Weight with negative grad should increase, got %.5f", dense->weights->data[1] );
}

Test( network, adam_converges_on_simple_quadratic )
{
  // Minimize f(w) = (w-3)^2, grad = 2(w-3)
  _network_t *net = network_create( &pa );
  network_add( net, dense_create( &pa, 1, 1 ) );

  _layer_t *dense = net->layers[0];
  dense->weights->data[0] = 0.0f;
  dense->bias->data[0] = 0.0f;

  _adam_optimizer_t *opt = adam_create( &pa, net, 0.1f );

  for ( int step = 0; step < 200; step++ )
  {
    float w = dense->weights->data[0];
    float grad = 2.0f * ( w - 3.0f );

    tensor_zero_gradient( dense->weights );
    dense->weights->gradients[0] = grad;

    adam_step( opt );
  }

  cr_assert( fabsf( dense->weights->data[0] - 3.0f ) < 0.05f, "Weight should converge near 3.0, got %.5f", dense->weights->data[0] );
}

Test( network, adam_registers_bn_params_too )
{
  _network_t *net = network_create( &pa );
  network_add( net, conv2d_create( &pa, 1, 2, 3, 1, 1 ) );
  network_add( net, batchnorm_create( &pa, 2 ) );

  _adam_optimizer_t *opt = adam_create( &pa, net, 0.01f );

  // conv: weights + bias = 2 slots
  // bn: gamma + beta = 2 slots
  cr_assert_eq( opt->num_slots, 4 );
}

Test( network, adam_m_v_persist_across_batch_arena_reset )
{
  _network_t *net = network_create( &pa );
  network_add( net, dense_create( &pa, 1, 1 ) );

  _layer_t *dense = net->layers[0];
  _adam_optimizer_t *opt = adam_create( &pa, net, 0.1f );

  // Step 1: build up momentum
  dense->weights->gradients[0] = 1.0f;
  adam_step( opt );
  float m_after_step1 = opt->slots[0].m->data[0];
  cr_assert( fabsf( m_after_step1 ) > 0.0f );

  // Simulate a batch_arena reset (m/v are in param_arena, unaffected)
  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ba );
  tensor_ones( &ba, 1, ( int[] ){ 100 } ); // some unrelated batch allocation
  ds_arena_reset_to( &ba, cp );

  // Step 2: momentum should still reflect step 1's history
  dense->weights->gradients[0] = 1.0f;
  adam_step( opt );

  // With beta1=0.9: m2 = 0.9*m1 + 0.1*g = 0.9*0.1 + 0.1*1 = 0.19
  cr_assert_float_eq( opt->slots[0].m->data[0], 0.19f, 1e-5f );
}

// End-to-end mini training step (BCE + SmoothL1 combined)
Test( network, end_to_end_detection_head_training_step )
{
  // Tiny "detector": Conv -> ReLU -> GAP -> Dense(5)
  _network_t *net = network_create( &pa );
  network_add( net, conv2d_create( &pa, 1, 4, 3, 1, 1 ) );
  network_add( net, relu_create( &pa ) );
  network_add( net, gap_create( &pa ) );
  network_add( net, dense_create( &pa, 4, 5 ) );

  _adam_optimizer_t *opt = adam_create( &pa, net, 0.01f );

  int in_shape[] = { 1, 1, 8, 8 };

  float initial_w = net->layers[3]->weights->data[0];

  for ( int step = 0; step < 5; step++ )
  {
    _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ba );

    _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

    network_zero_gradient( net );
    _tensor_t *pred = network_forward( net, &ba, input ); // (1,5)

    // Targets: hand present, centered box
    int conf_shape[] = { 1, 1 };
    _tensor_t *conf_pred = tensor_zeros( &ba, 2, conf_shape );
    conf_pred->data[0] = pred->data[0];
    _tensor_t *conf_tgt = tensor_ones( &ba, 2, conf_shape );

    int box_shape[] = { 1, 4 };
    _tensor_t *box_pred = tensor_zeros( &ba, 2, box_shape );
    for ( int k = 0; k < 4; k++ ) box_pred->data[k] = pred->data[1 + k];
    _tensor_t *box_tgt = tensor_zeros( &ba, 2, box_shape );
    for ( int k = 0; k < 4; k++ ) box_tgt->data[k] = 0.5f;

    _loss_t_ conf_loss = bce_with_logits( &ba, conf_pred, conf_tgt );
    _loss_t_ box_loss = smooth_l1( &ba, box_pred, box_tgt, NULL );

    _tensor_t *combined = combiine_head_gradients( &ba, conf_loss.gradients, box_loss.gradients, 1, 5.0f );

    network_backward( net, &ba, combined );
    adam_step( opt );

    ds_arena_reset_to( &ba, cp );
  }

  float final_w = net->layers[3]->weights->data[0];

  cr_assert( fabsf( final_w - initial_w ) > 1e-6f, "Dense weights should have changed after training steps" );
}
