#include "augmentation.h"
#include "dataloader.h"
#include "ds_arena.h"
#include <criterion/assert.h>
#include <criterion/criterion.h>
#include <criterion/internal/assert.h>
#include <string.h>

static _ds_arena_t_ ba;

void aug_setup( void )
{
  ba = ds_arena_new( 0 );
}

void aug_teardown( void )
{
  ds_arena_destroy( &ba );
}

TestSuite( augmentation, .init = aug_setup, .fini = aug_teardown );

// Horizontal flip
Test( augmentation, flip_mirrors_pixels_1channel )
{
  unsigned char data[4] = { 0, 1, 2, 3 };
  _sample_t s = { 0 };

  augmentation_horizontal_flip( data, 4, 1, 1, &s );

  cr_assert_eq( data[0], 3 );
  cr_assert_eq( data[1], 2 );
  cr_assert_eq( data[2], 1 );
  cr_assert_eq( data[3], 0 );
}

Test( augmentation, flip_mirrors_pixels_3channels )
{
  // 2*1 RGB. pixel0=(1, 2, 3), pixel1=(4, 5, 6)
  unsigned char data[6] = { 1, 2, 3, 4, 5, 6 };
  _sample_t s = { 0 };

  augmentation_horizontal_flip( data, 2, 1, 3, &s );

  cr_assert_eq( data[0], 4 );
  cr_assert_eq( data[1], 5 );
  cr_assert_eq( data[2], 6 );

  cr_assert_eq( data[3], 1 );
  cr_assert_eq( data[4], 2 );
  cr_assert_eq( data[5], 3 );
}

Test( augmentation, flip_odd_width_center_column_unchanged )
{
  unsigned char data[3] = { 1, 2, 3 };
  _sample_t s = { 0 };

  augmentation_horizontal_flip( data, 3, 1, 1, &s );

  cr_assert_eq( data[0], 3 );
  cr_assert_eq( data[1], 2 ); // center untouched
  cr_assert_eq( data[2], 1 );
}

Test( augmentation, flip_updates_cx_only )
{
  _sample_t s = {
    .cx = 0.3f,
    .cy = 0.5f,
    .w = 0.2f,
    .h = 0.3f,
    .has_hand = 1,
  };

  unsigned char data[16];
  for ( int i = 0; i < 16; i++ ) data[i] = (unsigned char)( i * 16 );

  augmentation_horizontal_flip( data, 4, 4, 1, &s );

  cr_assert_float_eq( s.cx, 0.7f, 1e-6f );
  cr_assert_float_eq( s.cy, 0.5f, 1e-6f );
  cr_assert_float_eq( s.w, 0.2f, 1e-6f );
  cr_assert_float_eq( s.h, 0.3f, 1e-6f );
}

Test( augmentation, flip_negative_sample_label_untouched )
{
  _sample_t s = {
    .cx = 0.0f,
    .has_hand = 0,
  };

  unsigned char data[4] = { 1, 2, 3, 4 };
  augmentation_horizontal_flip( data, 4, 1, 1, &s );

  cr_assert_float_eq( s.cx, 0.0f, 1e-6f );
  cr_assert_eq( s.has_hand, false );
}

Test( augmentation, flip_twice_is_identity )
{
  unsigned char data[8] = {
    10, 20, 30, 40, 50, 60, 70, 80,
  };
  unsigned char original[8];
  memcpy( original, data, 8 );

  _sample_t s = {
    .cx = 0.25f,
    .has_hand = true,
  };

  augmentation_horizontal_flip( data, 8, 1, 1, &s );
  augmentation_horizontal_flip( data, 8, 1, 1, &s );

  for ( int i = 0; i < 8; i++ ) cr_assert_eq( data[i], original[i] );

  cr_assert_float_eq( s.cx, 0.25f, 1e-6f );
}

Test( augmentation, brightness_clamps_high )
{
  unsigned char data[3] = { 200, 100, 50 };
  augmentation_birghtness( data, 3, 2.0f ); // 200*2=400 -> clamp 255

  cr_assert_eq( data[0], 255 );
  cr_assert_eq( data[1], 200 );
  cr_assert_eq( data[2], 100 );
}

Test( augmentation, brightness_clamps_low )
{
  unsigned char data[2] = { 10, 5 };
  augmentation_birghtness( data, 2, 0.0f );

  cr_assert_eq( data[0], 0 );
  cr_assert_eq( data[1], 0 );
}

Test( augmentation, brightness_identity_at_factor_one )
{
  unsigned char data[3] = { 10, 128, 250 };
  unsigned char original[3] = { 10, 128, 250 };

  augmentation_birghtness( data, 3, 1.0f );

  for ( int i = 0; i < 3; i++ ) cr_assert_eq( data[i], original[i] );
}

// Contrast
Test( augmentation, contrast_identity_at_factor_one )
{
  unsigned char data[3] = { 50, 127, 200 };
  unsigned char original[3] = { 50, 127, 200 };

  augmentation_contrast( data, 3, 1.0f );

  for ( int i = 0; i < 3; i++ ) cr_assert_eq( data[i], original[i] );
}

Test( augmentation, contrast_zero_factor_collapses_to_midgray )
{
  // factor=0 -> v = (x - 127.5)*0 + 127.5 = 127.5 -> truncates to 127
  unsigned char data[3] = { 0, 100, 255 };
  augmentation_contrast( data, 3, 0.0f );

  for ( int i = 0; i < 3; i++ ) cr_assert( data[i] == 127 || data[i] == 128, "Expected ~127.5, got %d", data[i] );
}

Test( augmentation, contrast_increases_spread )
{
  unsigned char data[2] = { 100, 150 }; // below / above 127.5
  augmentation_contrast( data, 2, 2.0f );

  cr_assert( data[0] < 100 ); // pushed further below midgray
  cr_assert( data[1] > 150 ); // pushed further above midgray
}

// Gaussian noise
Test( augmentation, gaussian_noise_stays_in_bounds )
{
  unsigned char data[1000];
  for ( int i = 0; i < 1000; i++ ) data[i] = 128;

  augmenttion_gaussian_noise( data, 1000, 0.1f );

  for ( int i = 0; i < 1000; i++ )
  {
    cr_assert( data[i] >= 0 );
    cr_assert( data[i] <= 255 );
  }
}

Test( augmentation, gaussian_noise_changes_values )
{
  unsigned char data[1000];
  unsigned char original[1000];
  for ( int i = 0; i < 1000; i++ )
  {
    data[i] = 128;
    original[i] = 128;
  }

  srand( 42 );
  augmenttion_gaussian_noise( data, 1000, 0.1f );

  int changed = 0;
  for ( int i = 0; i < 1000; i++ )
    if ( data[i] != original[i] ) changed++;

  cr_assert( changed > 100, "Expected most values to change, got %d/1000", changed );
}

Test( augmentation, gaussian_noise_zero_sigma_is_identity )
{
  unsigned char data[10];
  unsigned char original[10];
  for ( int i = 0; i < 10; i++ )
  {
    data[i] = 100;
    original[i] = 100;
  }

  augmenttion_gaussian_noise( data, 10, 0.0f );

  for ( int i = 0; i < 10; i++ ) cr_assert_eq( data[i], original[i] );
}

// Random Crop -- pixel extraction
Test( augmentation, crop_extracts_correct_pixel_region )
{
  // 4x4 single-channel, values = row*4+col (0..15)
  unsigned char data[16];
  for ( int i = 0; i < 16; i++ ) data[i] = (unsigned char)i;

  int w = 4, h = 4;
  _sample_t s = { .has_hand = 0 };

  // Crop top-left 2x2: x0=0,y0=0 -> values 0,1,4,5
  unsigned char *cropped = augmenttion_random_crop( &ba, data, &w, &h, 1, &s, 2, 2, 0, 0 );

  cr_assert_eq( w, 2 );
  cr_assert_eq( h, 2 );
  cr_assert_eq( cropped[0], 0 );
  cr_assert_eq( cropped[1], 1 );
  cr_assert_eq( cropped[2], 4 );
  cr_assert_eq( cropped[3], 5 );
}

Test( augmentation, crop_extracts_offset_region )
{
  unsigned char data[16];
  for ( int i = 0; i < 16; i++ ) data[i] = (unsigned char)i;

  int w = 4, h = 4;
  _sample_t s = { .has_hand = 0 };

  // Crop bottom-right 2x2: x0=2,y0=2 -> values 10,11,14,15
  unsigned char *cropped = augmenttion_random_crop( &ba, data, &w, &h, 1, &s, 2, 2, 2, 2 );

  cr_assert_eq( cropped[0], 10 );
  cr_assert_eq( cropped[1], 11 );
  cr_assert_eq( cropped[2], 14 );
  cr_assert_eq( cropped[3], 15 );
}

Test( augmentation, crop_3channel_preserves_pixel_layout )
{
  // 2x2 RGB: Pixel(0,0)=(1,2,3) (0,1)=(4,5,6) (1,0)=(7,8,9) (1,1)=(10,11,12)
  unsigned char data[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  int w = 2, h = 2;
  _sample_t s = { .has_hand = 0 };

  // Crop 1x1 at (1,1) -> pixel (10,11,12)
  unsigned char *cropped = augmenttion_random_crop( &ba, data, &w, &h, 3, &s, 1, 1, 1, 1 );

  cr_assert_eq( cropped[0], 10 );
  cr_assert_eq( cropped[1], 11 );
  cr_assert_eq( cropped[2], 12 );
}

// Random Crop -- label transform
Test( augmentation, crop_box_fully_inside_updates_label_correctly )
{
  // 100x100 image. Box cx=0.5,cy=0.5,w=0.2,h=0.2 -> pixel box [40,40]-[60,60]
  _sample_t s = { .cx = 0.5f, .cy = 0.5f, .w = 0.2f, .h = 0.2f, .has_hand = 1 };

  int w = 100, h = 100;
  unsigned char *data = ARENA_ARRAY( &ba, unsigned char, 100 * 100 );

  // Crop [20,20]-[80,80] (60x60), fully contains the box
  augmenttion_random_crop( &ba, data, &w, &h, 1, &s, 60, 60, 20, 20 );

  cr_assert_eq( w, 60 );
  cr_assert_eq( h, 60 );
  cr_assert_eq( s.has_hand, true );

  // Box in crop frame: [20,20]-[40,40] of 60x60
  // cx=(20+40)/2/60=0.5, w=(40-20)/60=0.3333
  cr_assert_float_eq( s.cx, 0.5f, 1e-4f );
  cr_assert_float_eq( s.cy, 0.5f, 1e-4f );
  cr_assert_float_eq( s.w, 20.0f / 60.0f, 1e-4f );
  cr_assert_float_eq( s.h, 20.0f / 60.0f, 1e-4f );
}

Test( augmentation, crop_box_partially_outside_clips_correctly )
{
  // 100x100 image. Box [40,40]-[60,60] (cx=0.5,cy=0.5,w=0.2,h=0.2)
  _sample_t s = { .cx = 0.5f, .cy = 0.5f, .w = 0.2f, .h = 0.2f, .has_hand = 1 };

  int w = 100, h = 100;
  unsigned char *data = ARENA_ARRAY( &ba, unsigned char, 100 * 100 );

  // Crop [50,50]-[100,100] (50x50) — box overlaps [50,50]-[60,60]
  augmenttion_random_crop( &ba, data, &w, &h, 1, &s, 50, 50, 50, 50 );

  cr_assert_eq( s.has_hand, 1 );

  // Clipped box [50,50]-[60,60] in crop frame (origin 50,50, size 50x50): [0,0]-[10,10]
  // cx = (0+10)/2/50 = 0.1, w = 10/50 = 0.2
  cr_assert_float_eq( s.cx, 0.1f, 1e-4f );
  cr_assert_float_eq( s.cy, 0.1f, 1e-4f );
  cr_assert_float_eq( s.w, 0.2f, 1e-4f );
  cr_assert_float_eq( s.h, 0.2f, 1e-4f );
}

Test( augmentation, crop_box_entirely_outside_sets_has_hand_zero )
{
  // Box top-left [5,5]-[15,15] (cx=0.1,cy=0.1,w=0.1,h=0.1 of 100x100)
  _sample_t s = { .cx = 0.1f, .cy = 0.1f, .w = 0.1f, .h = 0.1f, .has_hand = 1 };

  int w = 100, h = 100;
  unsigned char *data = ARENA_ARRAY( &ba, unsigned char, 100 * 100 );

  // Crop bottom-right [60,60]-[100,100] — no overlap
  augmenttion_random_crop( &ba, data, &w, &h, 1, &s, 40, 40, 60, 60 );

  cr_assert_eq( s.has_hand, 0 );
  cr_assert_float_eq( s.cx, 0.0f, 1e-6f );
  cr_assert_float_eq( s.cy, 0.0f, 1e-6f );
  cr_assert_float_eq( s.w, 0.0f, 1e-6f );
  cr_assert_float_eq( s.h, 0.0f, 1e-6f );
}

Test( augmentation, crop_box_touching_edge_exactly_sets_has_hand_zero )
{
  // Box [0,0]-[20,20] (cx=0.1,cy=0.1,w=0.2,h=0.2 of 100x100)
  _sample_t s = { .cx = 0.1f, .cy = 0.1f, .w = 0.2f, .h = 0.2f, .has_hand = 1 };

  int w = 100, h = 100;
  unsigned char *data = ARENA_ARRAY( &ba, unsigned char, 100 * 100 );

  // Crop starting exactly at x=20,y=20 -> box's max edge == crop's min edge
  // -> zero-area intersection -> has_hand=0
  augmenttion_random_crop( &ba, data, &w, &h, 1, &s, 30, 30, 20, 20 );

  cr_assert_eq( s.has_hand, 0 );
}

Test( augmentation, crop_negative_sample_label_unchanged )
{
  _sample_t s = { .has_hand = 0 };

  int w = 50, h = 50;
  unsigned char *data = ARENA_ARRAY( &ba, unsigned char, 50 * 50 );

  augmenttion_random_crop( &ba, data, &w, &h, 1, &s, 20, 20, 10, 10 );

  cr_assert_eq( s.has_hand, 0 );
  cr_assert_eq( w, 20 );
  cr_assert_eq( h, 20 );
}

//  Arena usage pattern
Test( augmentation, crop_allocates_in_provided_arena_and_resets_cleanly )
{
  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ba );

  int w = 100, h = 100;
  unsigned char *data = ARENA_ARRAY( &ba, unsigned char, 100 * 100 );
  _sample_t s = { .has_hand = 0 };

  unsigned char *cropped = augmenttion_random_crop( &ba, data, &w, &h, 1, &s, 50, 50, 0, 0 );
  cr_assert_not_null( cropped );
  cr_assert_eq( w, 50 );

  ds_arena_reset_to( &ba, cp );

  size_t used_after = ba.head ? ba.head->chunk_size_used : 0;
  cr_assert_eq( used_after, cp.checkpoint_size_used );
}
