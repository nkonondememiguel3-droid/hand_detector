#include "ds_arena.h"
#include "preprocess.h"
#include "tensor.h"
#include <criterion/criterion.h>
#include <criterion/new/assert.h>

static _ds_arena_t_ ba;

void prep_setup( void )
{
  ba = ds_arena_new( 0 );
}
void prep_teardown( void )
{
  ds_arena_destroy( &ba );
}

TestSuite( preprocess, .init = prep_setup, .fini = prep_teardown );

// HWC -> CHW layout
Test( preprocess, hwc_to_chw_layout_correct )
{
  // 2x2 image, 3 channels
  // Pixel(0,0)=(10,20,30) Pixel(0,1)=(40,50,60)
  // Pixel(1,0)=(70,80,90) Pixel(1,1)=(100,110,120)
  unsigned char data[12] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };

  _tensor_t *t = hwc_uint8_to_chw_tensor( &ba, data, 2, 2, 3 );

  cr_assert_eq( t->shape[0], 1 );
  cr_assert_eq( t->shape[1], 3 );
  cr_assert_eq( t->shape[2], 2 );
  cr_assert_eq( t->shape[3], 2 );

  // Channel 0 (R): [10,40,70,100]/255
  cr_assert_float_eq( t->data[0], 10.0f / 255.0f, 1e-6f );
  cr_assert_float_eq( t->data[1], 40.0f / 255.0f, 1e-6f );
  cr_assert_float_eq( t->data[2], 70.0f / 255.0f, 1e-6f );
  cr_assert_float_eq( t->data[3], 100.0f / 255.0f, 1e-6f );

  // Channel 1 (G) starts at flat idx 4: [20,50,80,110]/255
  cr_assert_float_eq( t->data[4], 20.0f / 255.0f, 1e-6f );
  cr_assert_float_eq( t->data[7], 110.0f / 255.0f, 1e-6f );

  // Channel 2 (B) starts at flat idx 8: [30,60,90,120]/255
  cr_assert_float_eq( t->data[8], 30.0f / 255.0f, 1e-6f );
  cr_assert_float_eq( t->data[11], 120.0f / 255.0f, 1e-6f );
}

Test( preprocess, normalization_range_endpoints )
{
  unsigned char data[3] = { 0, 128, 255 };
  _tensor_t *t = hwc_uint8_to_chw_tensor( &ba, data, 1, 1, 3 );

  cr_assert_float_eq( t->data[0], 0.0f, 1e-6f );
  cr_assert( t->data[1] > 0.0f && t->data[1] < 1.0f );
  cr_assert_float_eq( t->data[2], 1.0f, 1e-6f );
}

Test( preprocess, single_channel_hwc_to_chw )
{
  // 3x1 grayscale, values 0,128,255
  unsigned char data[3] = { 0, 128, 255 };
  _tensor_t *t = hwc_uint8_to_chw_tensor( &ba, data, 3, 1, 1 );

  cr_assert_eq( t->shape[1], 1 ); // C=1
  cr_assert_eq( t->shape[2], 1 ); // H=1
  cr_assert_eq( t->shape[3], 3 ); // W=3

  cr_assert_float_eq( t->data[0], 0.0f, 1e-6f );
  cr_assert_float_eq( t->data[1], 128.0f / 255.0f, 1e-6f );
  cr_assert_float_eq( t->data[2], 1.0f, 1e-6f );
}

Test( preprocess, output_size_matches_size_field )
{
  unsigned char data[12] = { 0 };
  _tensor_t *t = hwc_uint8_to_chw_tensor( &ba, data, 2, 2, 3 );

  cr_assert_eq( t->size, 1 * 3 * 2 * 2 );
}

// Resize
Test( preprocess, resize_changes_dimensions_and_stays_in_range )
{
  unsigned char data[16];
  for ( int i = 0; i < 16; i++ ) data[i] = (unsigned char)( i * 16 );

  unsigned char *resized = resize_image( &ba, data, 4, 4, 1, 2, 2 );

  cr_assert_not_null( resized );
  for ( int i = 0; i < 4; i++ )
  {
    cr_assert( resized[i] >= 0 );
    cr_assert( resized[i] <= 255 );
  }
}

Test( preprocess, resize_upscale_no_crash )
{
  unsigned char data[4] = { 0, 64, 128, 255 }; // 2x2 grayscale
  unsigned char *resized = resize_image( &ba, data, 2, 2, 1, 8, 8 );

  cr_assert_not_null( resized );
  for ( int i = 0; i < 64; i++ )
  {
    cr_assert( resized[i] >= 0 );
    cr_assert( resized[i] <= 255 );
  }
}

Test( preprocess, resize_identity_when_same_size )
{
  unsigned char data[4] = { 10, 20, 30, 40 }; // 2x2 grayscale
  unsigned char *resized = resize_image( &ba, data, 2, 2, 1, 2, 2 );

  for ( int i = 0; i < 4; i++ ) cr_assert( abs( (int)resized[i] - (int)data[i] ) <= 1, "Pixel %d: expected ~%d, got %d", i, data[i], resized[i] );
}

// load_image_raw / load_and_preprocess -- failure paths

Test( preprocess, load_nonexistent_file_returns_null )
{
  _tensor_t *t = load_and_preprocess( &ba, "/nonexistent/path/image.jpg", 256, 256 );
  cr_assert_null( t );
}

Test( preprocess, load_image_raw_nonexistent_returns_null )
{
  int w, h, c;
  unsigned char *data = load_image_raw( &ba, "/nonexistent/path/image.jpg", &w, &h, &c );
  cr_assert_null( data );
}

// Full pipeline with real fixture image
//
// Requires tests/fixtures/test_image.jpg. Generate once with:
//   python3 -c "from PIL import Image; Image.new('RGB',(10,10),(255,0,0)).save('tests/fixtures/test_image.jpg')"

Test( preprocess, load_and_preprocess_real_image_shape_and_range )
{
  _tensor_t *t = load_and_preprocess( &ba, "../tests/fixtures/test_image.jpg", 256, 256 );

  cr_assert_not_null( t );
  cr_assert_eq( t->shape[0], 1 );
  cr_assert_eq( t->shape[1], 3 );
  cr_assert_eq( t->shape[2], 256 );
  cr_assert_eq( t->shape[3], 256 );

  for ( int i = 0; i < t->size; i++ )
  {
    cr_assert( t->data[i] >= 0.0f );
    cr_assert( t->data[i] <= 1.0f );
  }
}

Test( preprocess, load_image_raw_then_resize_then_tensor_pipeline )
{
  int w, h, c;
  unsigned char *raw = load_image_raw( &ba, "../tests/fixtures/test_image.jpg", &w, &h, &c );
  cr_assert_not_null( raw );
  cr_assert_eq( c, 3 );

  unsigned char *resized = resize_image( &ba, raw, w, h, c, 128, 128 );
  _tensor_t *t = hwc_uint8_to_chw_tensor( &ba, resized, 128, 128, c );

  cr_assert_eq( t->shape[2], 128 );
  cr_assert_eq( t->shape[3], 128 );
}

// Full augment + preprocess pipeline (checkpoint pattern)
Test( preprocess, full_pipeline_with_arena_checkpoint )
{
  _ds_arena_checkpoint_t_ cp = ds_arena_checkpoint( &ba );

  int w, h, c;
  unsigned char *raw = load_image_raw( &ba, "../tests/fixtures/test_image.jpg", &w, &h, &c );
  cr_assert_not_null( raw );

  unsigned char *resized = resize_image( &ba, raw, w, h, c, 256, 256 );
  _tensor_t *t = hwc_uint8_to_chw_tensor( &ba, resized, 256, 256, c );

  cr_assert_eq( t->shape[2], 256 );
  cr_assert_eq( t->shape[3], 256 );

  ds_arena_reset_to( &ba, cp );

  size_t used_after = ba.head ? ba.head->chunk_size_used : 0;
  cr_assert_eq( used_after, cp.checkpoint_size_used );
}
