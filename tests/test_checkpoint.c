#include "bn_fold.h"
#include "checkpoint.h"
#include "ds_arena.h"
#include "layers.h"
#include "network.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <stdio.h>
#include <unistd.h>

static _ds_arena_t_ pa;
static char ckpt_path[256];

void ckpt_setup( void )
{
  pa = ds_arena_new( 0 );
  snprintf( ckpt_path, sizeof( ckpt_path ), "/tmp/ds_ckpt_test_%d.bin", getpid() );
}

void ckpt_teardown( void )
{
  remove( ckpt_path );
  ds_arena_destroy( &pa );
}

TestSuite( checkpoint, .init = ckpt_setup, .fini = ckpt_teardown );

static _network_t *build_small_net( _ds_arena_t_ *a )
{
  _network_t *net = network_create( a );
  network_add( net, conv2d_create( a, 1, 2, 3, 1, 1 ) );
  network_add( net, batchnorm_create( a, 2 ) );
  network_add( net, relu_create( a ) );
  network_add( net, gap_create( a ) );
  network_add( net, dense_create( a, 2, 5 ) );
  return net;
}

// Round trip: save then load into a fresh identical network
Test( checkpoint, save_then_load_roundtrip_preserves_weights )
{
  _network_t *net1 = build_small_net( &pa );

  // Assign distinctive values
  for ( int i = 0; i < net1->layers[0]->weights->size; i++ ) net1->layers[0]->weights->data[i] = (float)i * 0.1f;
  for ( int i = 0; i < net1->layers[0]->bias->size; i++ ) net1->layers[0]->bias->data[i] = (float)i + 1.0f;

  _layer_t *bn1 = net1->layers[1];
  bn1->gamma->data[0] = 1.5f;
  bn1->beta->data[1] = -0.3f;
  bn1->running_mean->data[0] = 0.7f;
  bn1->running_var->data[1] = 2.2f;

  for ( int i = 0; i < net1->layers[4]->weights->size; i++ ) net1->layers[4]->weights->data[i] = (float)i * 0.01f - 1.0f;

  cr_assert( checkpoint_save( net1, ckpt_path ) );

  // Build a fresh net (different random init) and load into it
  _ds_arena_t_ pa2 = ds_arena_new( 0 );
  _network_t *net2 = build_small_net( &pa2 );

  cr_assert( checkpoint_load( net2, ckpt_path ) );

  for ( int i = 0; i < net1->layers[0]->weights->size; i++ )
    cr_assert_float_eq( net2->layers[0]->weights->data[i], net1->layers[0]->weights->data[i], 1e-7f );

  for ( int i = 0; i < net1->layers[0]->bias->size; i++ ) cr_assert_float_eq( net2->layers[0]->bias->data[i], net1->layers[0]->bias->data[i], 1e-7f );

  _layer_t *bn2 = net2->layers[1];
  cr_assert_float_eq( bn2->gamma->data[0], 1.5f, 1e-7f );
  cr_assert_float_eq( bn2->beta->data[1], -0.3f, 1e-7f );
  cr_assert_float_eq( bn2->running_mean->data[0], 0.7f, 1e-7f );
  cr_assert_float_eq( bn2->running_var->data[1], 2.2f, 1e-7f );

  for ( int i = 0; i < net1->layers[4]->weights->size; i++ )
    cr_assert_float_eq( net2->layers[4]->weights->data[i], net1->layers[4]->weights->data[i], 1e-7f );

  ds_arena_destroy( &pa2 );
}

// Load into wrong architecture fails cleanly
Test( checkpoint, load_fails_on_layer_count_mismatch )
{
  _network_t *net1 = build_small_net( &pa );
  cr_assert( checkpoint_save( net1, ckpt_path ) );

  _ds_arena_t_ pa2 = ds_arena_new( 0 );
  _network_t *net2 = network_create( &pa2 );
  network_add( net2, conv2d_create( &pa2, 1, 2, 3, 1, 1 ) );
  network_add( net2, batchnorm_create( &pa2, 2 ) );
  // missing relu/gap/dense -> layer count mismatch

  cr_assert_not( checkpoint_load( net2, ckpt_path ) );

  ds_arena_destroy( &pa2 );
}

Test( checkpoint, load_fails_on_shape_mismatch_without_corrupting_net )
{
  _network_t *net1 = build_small_net( &pa );
  cr_assert( checkpoint_save( net1, ckpt_path ) );

  // Build a structurally-similar net but with different conv out_channels
  _ds_arena_t_ pa2 = ds_arena_new( 0 );
  _network_t *net2 = network_create( &pa2 );
  network_add( net2, conv2d_create( &pa2, 1, 4, 3, 1, 1 ) ); // out_channels=4 instead of 2
  network_add( net2, batchnorm_create( &pa2, 4 ) );
  network_add( net2, relu_create( &pa2 ) );
  network_add( net2, gap_create( &pa2 ) );
  network_add( net2, dense_create( &pa2, 4, 5 ) );

  // Capture original (random-init) weights before failed load
  float original_w0 = net2->layers[0]->weights->data[0];

  cr_assert_not( checkpoint_load( net2, ckpt_path ) );

  // net2's weights must be untouched (all-or-nothing contract)
  cr_assert_float_eq( net2->layers[0]->weights->data[0], original_w0, 1e-9f );

  ds_arena_destroy( &pa2 );
}

Test( checkpoint, load_fails_on_layer_type_mismatch )
{
  _network_t *net1 = build_small_net( &pa );
  cr_assert( checkpoint_save( net1, ckpt_path ) );

  // Same layer count, but layer 1 is RELU instead of BATCHNORM
  _ds_arena_t_ pa2 = ds_arena_new( 0 );
  _network_t *net2 = network_create( &pa2 );
  network_add( net2, conv2d_create( &pa2, 1, 2, 3, 1, 1 ) );
  network_add( net2, relu_create( &pa2 ) ); // mismatch: was BATCHNORM
  network_add( net2, relu_create( &pa2 ) );
  network_add( net2, gap_create( &pa2 ) );
  network_add( net2, dense_create( &pa2, 2, 5 ) );

  cr_assert_not( checkpoint_load( net2, ckpt_path ) );

  ds_arena_destroy( &pa2 );
}

Test( checkpoint, load_fails_on_missing_file )
{
  _network_t *net = build_small_net( &pa );
  cr_assert_not( checkpoint_load( net, "/nonexistent/path/ckpt.bin" ) );
}

Test( checkpoint, load_fails_on_bad_magic )
{
  // Write garbage that isn't a valid checkpoint
  FILE *f = fopen( ckpt_path, "wb" );
  const char *garbage = "NOTC";
  fwrite( garbage, 1, 4, f );
  int32_t num_layers = 5;
  fwrite( &num_layers, sizeof( num_layers ), 1, f );
  fclose( f );

  _network_t *net = build_small_net( &pa );
  cr_assert_not( checkpoint_load( net, ckpt_path ) );
}

// Save then fold then save again -- folded checkpoint differs
Test( checkpoint, folded_network_checkpoint_differs_from_unfolded )
{
  _network_t *net = build_small_net( &pa );

  _layer_t *conv = net->layers[0];
  _layer_t *bn = net->layers[1];

  bn->running_mean->data[0] = 0.5f;
  bn->running_var->data[0] = 2.0f;
  bn->gamma->data[0] = 1.5f;
  bn->beta->data[0] = 0.1f;

  float original_w0 = conv->weights->data[0];

  cr_assert( checkpoint_save( net, ckpt_path ) );

  fold_batchnorm_into_conv( conv, bn );

  cr_assert( fabsf( conv->weights->data[0] - original_w0 ) > 1e-6f, "Folding should change conv weights" );

  // Save folded checkpoint to a different path and verify it round-trips
  char folded_path[300];
  snprintf( folded_path, sizeof( folded_path ), "%s.folded", ckpt_path );

  cr_assert( checkpoint_save( net, folded_path ) );

  _ds_arena_t_ pa2 = ds_arena_new( 0 );
  _network_t *net2 = build_small_net( &pa2 );
  cr_assert( checkpoint_load( net2, folded_path ) );

  cr_assert_float_eq( net2->layers[0]->weights->data[0], conv->weights->data[0], 1e-7f );

  remove( folded_path );
  ds_arena_destroy( &pa2 );
}
