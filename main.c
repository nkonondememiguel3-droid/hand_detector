#include "augmentation.h"
#include "checkpoint.h"
#include "dataloader.h"
#include "ds_arena.h"
#include "layers.h"
#include "loss.h"
#include "network.h"
#include "optimizer.h"
#include "preprocess.h"
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 256
#define BATCH_SIZE 16
#define NUM_EPOCHS 30
#define LEARNING_RATE 1e-3f
#define BOX_LOSS_WEIGHT 5.0f
#define GRAD_CLIP_NORM 5.0f
#define CHECKPOINT_PATH "checkpoints/hand_detector.bin"
#define CHECKPOINT_EVERY_N_EPOCHS 1

// Network architecture
static _network_t *build_network( _ds_arena_t_ *param_arena )
{
  _network_t *net = network_create( param_arena );

  // Stem: 256x256x3 -> 128x128x16
  network_add( net, conv2d_create( param_arena, 3, 16, 3, 2, 1 ) );
  network_add( net, batchnorm_create( param_arena, 16 ) );
  network_add( net, relu_create( param_arena ) );

  // 128x128x16 -> 64x64x32
  network_add( net, conv2d_create( param_arena, 16, 32, 3, 2, 1 ) );
  network_add( net, batchnorm_create( param_arena, 32 ) );
  network_add( net, relu_create( param_arena ) );

  // 64x64x32 -> 32x32x64
  network_add( net, conv2d_create( param_arena, 32, 64, 3, 2, 1 ) );
  network_add( net, batchnorm_create( param_arena, 64 ) );
  network_add( net, relu_create( param_arena ) );

  // 32x32x64 -> 16x16x128
  network_add( net, conv2d_create( param_arena, 64, 128, 3, 2, 1 ) );
  network_add( net, batchnorm_create( param_arena, 128 ) );
  network_add( net, relu_create( param_arena ) );

  // 16x16x128 -> 8x8x128
  network_add( net, conv2d_create( param_arena, 128, 128, 3, 2, 1 ) );
  network_add( net, batchnorm_create( param_arena, 128 ) );
  network_add( net, relu_create( param_arena ) );

  // Global average pool -> (1,128)
  network_add( net, gap_create( param_arena ) );

  // Head: 128 -> 5  [conf, cx, cy, w, h]
  network_add( net, dense_create( param_arena, 128, 5 ) );

  return net;
}

// Per-sample forward + loss + backward
// Returns the combined scalar loss for this sample, and accumulates
// gradients into the network's parameter gradient tensors via backward().
static float train_one_sample( _network_t *net, _ds_arena_t_ *batch_arena, _tensor_t *input, _sample_t *label )
{
  _tensor_t *output = network_forward( net, batch_arena, input ); // (1,5)

  // Split output into conf (1 elem) and box (4 elems)
  int conf_shape[] = { 1, 1 };
  _tensor_t *conf_pred = tensor_zeros( batch_arena, 2, conf_shape );
  conf_pred->data[0] = output->data[0];

  int box_shape[] = { 1, 4 };
  _tensor_t *box_pred = tensor_zeros( batch_arena, 2, box_shape );
  for ( int i = 0; i < 4; i++ ) box_pred->data[i] = output->data[1 + i];

  // Targets
  _tensor_t *conf_target = tensor_zeros( batch_arena, 2, conf_shape );
  conf_target->data[0] = label->has_hand ? 1.0f : 0.0f;

  _tensor_t *box_target = tensor_zeros( batch_arena, 2, box_shape );
  box_target->data[0] = label->cx;
  box_target->data[1] = label->cy;
  box_target->data[2] = label->w;
  box_target->data[3] = label->h;

  // Mask: box loss only contributes for positive samples
  _tensor_t *mask = tensor_zeros( batch_arena, 2, box_shape );
  for ( int i = 0; i < 4; i++ ) mask->data[i] = label->has_hand ? 1.0f : 0.0f;

  _loss_t_ conf_loss = bce_with_logits( batch_arena, conf_pred, conf_target );
  _loss_t_ box_loss = smooth_l1( batch_arena, box_pred, box_target, mask );

  float total_loss = conf_loss.value + BOX_LOSS_WEIGHT * box_loss.value;

  // Combine gradients back into a (1,5) tensor matching output's shape
  _tensor_t *grad_output = combiine_head_gradients( batch_arena, conf_loss.gradients, box_loss.gradients, BATCH_SIZE, BOX_LOSS_WEIGHT );

  network_backward( net, batch_arena, grad_output );

  return total_loss;
}

// Epoch loop
// Loads, augments, and preprocesses sample `idx` from `dataset`.
// Returns NULL if the image fails to load (caller should skip).
static _tensor_t *prepare_sample( _ds_arena_t_ *batch_arena, _sample_t *sample )
{
  int w, h, c;
  unsigned char *raw = load_image_raw( batch_arena, sample->image_path, &w, &h, &c );
  if ( !raw ) return NULL;

  // Augmentation (applied to the raw HWC buffer, mutates `sample`)
  // 50% horizontal flip
  if ( rand() % 2 == 0 ) augmentation_horizontal_flip( raw, w, h, c, sample );

  // Photometric jitter — labels unaffected
  float brightness_factor = 0.8f + ( (float)rand() / (float)RAND_MAX ) * 0.4f; // [0.8,1.2]
  augmentation_birghtness( raw, w * h * c, brightness_factor );

  float contrast_factor = 0.8f + ( (float)rand() / (float)RAND_MAX ) * 0.4f; // [0.8,1.2]
  augmentation_contrast( raw, w * h * c, contrast_factor );

  augmenttion_gaussian_noise( raw, w * h * c, 0.02f );

  // ── Resize + tensor conversion ──
  unsigned char *resized = resize_image( batch_arena, raw, w, h, c, INPUT_SIZE, INPUT_SIZE );
  return hwc_uint8_to_chw_tensor( batch_arena, resized, INPUT_SIZE, INPUT_SIZE, c );
}

static float run_epoch( _network_t *net, _adam_optimizer_t *opt, _ds_arena_t_ *batch_arena, _dataset_t *dataset, bool training )
{
  dataset_shuffle( dataset );

  float epoch_loss = 0.0f;
  int loss_count = 0;
  int accumulated = 0;

  for ( int i = 0; i < dataset->count; i++ )
  {
    _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( batch_arena );

    // Copy the sample so augmentation's in-place label mutation doesn't
    // corrupt the dataset for the next epoch's shuffle/use.
    _sample_t sample = dataset->samples[i];

    _tensor_t *input = prepare_sample( batch_arena, &sample );
    if ( !input )
    {
      fprintf( stderr, "Warning: failed to load %s, skipping\n", sample.image_path );
      ds_arena_reset_to( batch_arena, cp );
      continue;
    }

    float loss = train_one_sample( net, batch_arena, input, &sample );
    epoch_loss += loss;
    loss_count++;

    if ( training )
    {
      accumulated++;
      if ( accumulated >= BATCH_SIZE || i == dataset->count - 1 )
      {
        /* tensor_clip_gradient( opt, GRAD_CLIP_NORM ); */
        /* adam_step( opt, LEARNING_RATE ); */
        adam_step( opt );
        network_zero_gradient( net );
        accumulated = 0;
      }
    }

    // CRITICAL: reset AFTER backward (and after any optimizer step that
    // reads gradients) — batch_arena held last_input/last_output caches
    // that backward needed.
    ds_arena_reset_to( batch_arena, cp );

    if ( ( i + 1 ) % 100 == 0 ) printf( "  sample %d/%d  loss=%.4f\n", i + 1, dataset->count, loss );
  }

  return loss_count > 0 ? epoch_loss / (float)loss_count : 0.0f;
}

// main
int main( int argc, char **argv )
{
  if ( argc < 2 )
  {
    fprintf( stderr, "Usage: %s <datasets_root>\n", argv[0] );
    fprintf( stderr, "  expects datasets_root/oxford/.../unified and datasets_root/egohands/unified\n" );
    return 1;
  }

  srand( (unsigned int)time( NULL ) );

  // Two-arena setup
  _ds_arena_t_ param_arena = ds_arena_new( 0 );
  _ds_arena_t_ batch_arena = ds_arena_new( 0 );

  // Build network and optimizer
  _network_t *net = build_network( &param_arena );
  _adam_optimizer_t *opt = adam_create( &param_arena, net, LEARNING_RATE );

  // Load datasets
  char oxford_path_labels[1024], egohands_path_labels[1024];
  char oxford_path_images[1024], egohands_path_images[1024];
  snprintf( oxford_path_labels, sizeof( oxford_path_labels ), "%s/labelds", argv[1] );
  snprintf( egohands_path_labels, sizeof( egohands_path_labels ), "%s/images", argv[1] );
  snprintf( oxford_path_images, sizeof( oxford_path_images ), "%s/labelds", argv[1] );
  snprintf( egohands_path_images, sizeof( egohands_path_images ), "%s/images", argv[1] );

  _dataset_t *oxford = dataset_load( &param_arena, oxford_path_labels, oxford_path_images );
  _dataset_t *egohands = dataset_load( &param_arena, egohands_path_labels, egohands_path_images );

  printf( "Oxford: %d samples\n", oxford->count );
  printf( "EgoHands: %d samples\n", egohands->count );

  _dataset_t *dss[] = { oxford, egohands };
  _dataset_t *combined = dataset_concat( &param_arena, dss, 2 );
  printf( "Combined: %d samples\n", combined->count );

  int positives = 0, negatives = 0;
  dataset_stats( combined , &positives, &negatives);

  _dataset_t *train_set, *val_set;
  dataset_split( &param_arena, combined, &train_set, &val_set, 0.9f );
  printf( "Train: %d  Val: %d\n", train_set->count, val_set->count );

  // Optionally resume from checkpoint
  if ( argc >= 3 && checkpoint_load( net, argv[2] ) ) printf( "Resumed from checkpoint: %s\n", argv[2] );

  // Training loop
  for ( int epoch = 0; epoch < NUM_EPOCHS; epoch++ )
  {
    printf( "Epoch %d/%d\n", epoch + 1, NUM_EPOCHS );

    float train_loss = run_epoch( net, opt, &batch_arena, train_set, true );
    float val_loss = run_epoch( net, opt, &batch_arena, val_set, false );

    printf( "  train_loss=%.4f  val_loss=%.4f\n", train_loss, val_loss );

    if ( ( epoch + 1 ) % CHECKPOINT_EVERY_N_EPOCHS == 0 )
    {
      if ( checkpoint_save( net, CHECKPOINT_PATH ) ) printf( "  checkpoint saved -> %s\n", CHECKPOINT_PATH );
      else fprintf( stderr, "  WARNING: checkpoint save failed\n" );
    }
  }

  ds_arena_destroy( &batch_arena );
  ds_arena_destroy( &param_arena );

  return 0;
}
