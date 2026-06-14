#include "inference.h"
#include <criterion/criterion.h>
#include <criterion/new/assert.h>
/* #include <math.h> */

TestSuite( inference );

// Confidence decoding
Test( inference, confidence_decodes_via_sigmoid )
{
  float raw[5] = { 0.0f, 0.5f, 0.5f, 0.1f, 0.1f }; // logit=0 -> sigmoid=0.5
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert_float_eq( det.confidence, 0.5f, 1e-5f );
}

Test( inference, high_confidence_logit_near_one )
{
  float raw[5] = { 10.0f, 0.5f, 0.5f, 0.1f, 0.1f }; // sigmoid(10) ~= 0.99995
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert( det.confidence > 0.99f );
  cr_assert( det.confidence < 1.0f );
}

Test( inference, low_confidence_logit_near_zero )
{
  float raw[5] = { -10.0f, 0.5f, 0.5f, 0.1f, 0.1f }; // sigmoid(-10) ~= 0.00005
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert( det.confidence < 0.01f );
  cr_assert( det.confidence > 0.0f );
}

// Threshold gating
Test( inference, detected_true_above_threshold )
{
  float raw[5] = { 2.0f, 0.5f, 0.5f, 0.1f, 0.1f }; // sigmoid(2) ~= 0.88
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert( det.detected );
}

Test( inference, detected_false_below_threshold )
{
  float raw[5] = { -2.0f, 0.5f, 0.5f, 0.1f, 0.1f }; // sigmoid(-2) ~= 0.12
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert_not( det.detected );
}

Test( inference, detected_exactly_at_threshold_is_true )
{
  float raw[5] = { 0.0f, 0.5f, 0.5f, 0.1f, 0.1f }; // sigmoid(0)=0.5
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert( det.detected ); // >= threshold
}

// Pixel coordinate conversion
Test( inference, box_converts_to_pixel_coordinates )
{
  // cx=0.5, cy=0.5, w=0.2, h=0.4, frame 1000x500
  // x0=(0.5-0.1)*1000=400, x1=(0.5+0.1)*1000=600 -> width=200
  // y0=(0.5-0.2)*500=150,  y1=(0.5+0.2)*500=350  -> height=200
  float raw[5] = { 5.0f, 0.5f, 0.5f, 0.2f, 0.4f };
  _detection_t det = decode_detection( raw, 1000, 500, 0.5f );

  cr_assert_float_eq( det.x, 400.0f, 1e-3f );
  cr_assert_float_eq( det.y, 150.0f, 1e-3f );
  cr_assert_float_eq( det.width, 200.0f, 1e-3f );
  cr_assert_float_eq( det.height, 200.0f, 1e-3f );
}

Test( inference, centered_full_frame_box )
{
  // cx=0.5,cy=0.5,w=1.0,h=1.0 -> covers entire frame
  float raw[5] = { 5.0f, 0.5f, 0.5f, 1.0f, 1.0f };
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert_float_eq( det.x, 0.0f, 1e-3f );
  cr_assert_float_eq( det.y, 0.0f, 1e-3f );
  cr_assert_float_eq( det.width, 640.0f, 1e-3f );
  cr_assert_float_eq( det.height, 480.0f, 1e-3f );
}

// Clamping out-of-range raw values
Test( inference, negative_box_coords_are_clamped )
{
  // cx=0.05, w=0.5 -> x0=(0.05-0.25)*640 = -128 -> should clamp to 0
  float raw[5] = { 5.0f, 0.05f, 0.5f, 0.5f, 0.1f };
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert( det.x >= 0.0f );
  cr_assert( det.x + det.width <= 640.0f );
}

Test( inference, box_extending_past_frame_edge_is_clamped )
{
  // cx=0.95, w=0.5 -> x1=(0.95+0.25)*640=768 > 640 -> should clamp width
  float raw[5] = { 5.0f, 0.95f, 0.5f, 0.5f, 0.1f };
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert( det.x + det.width <= 640.0f );
  cr_assert( det.x >= 0.0f );
}

Test( inference, raw_values_outside_unit_range_are_clamped_before_pixel_conversion )
{
  // cx=1.5 (invalid, > 1) should clamp to 1.0 before pixel math
  float raw[5] = { 5.0f, 1.5f, 0.5f, 0.2f, 0.2f };
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  // cx clamped to 1.0, w=0.2 -> x0=(1.0-0.1)*640=576, x1=(1.0+0.1)*640=704->clamp 640
  cr_assert( det.x <= 640.0f );
  cr_assert( det.x + det.width <= 640.0f );
  cr_assert( det.x >= 0.0f );
}

Test( inference, negative_raw_width_clamps_to_zero )
{
  float raw[5] = { 5.0f, 0.5f, 0.5f, -0.3f, 0.1f }; // w=-0.3 -> clamp to 0
  _detection_t det = decode_detection( raw, 640, 480, 0.5f );

  cr_assert_float_eq( det.width, 0.0f, 1e-3f );
}

// Full pipeline: run_inference with a tiny network
Test( inference, run_inference_produces_valid_detection )
{
  _ds_arena_t_ pa = ds_arena_new( 0 );
  _ds_arena_t_ ba = ds_arena_new( 0 );

  _network_t *net = network_create( &pa );
  network_add( net, conv2d_create( &pa, 3, 4, 3, 2, 1 ) ); // 8x8 -> 4x4
  network_add( net, relu_create( &pa ) );
  network_add( net, gap_create( &pa ) );
  network_add( net, dense_create( &pa, 4, 5 ) );

  int in_shape[] = { 1, 3, 8, 8 };
  _tensor_t *input = tensor_random_normal( &ba, 4, in_shape, 0.0f, 1.0f );

  _detection_t det = run_inference( net, &ba, input, 640, 480, 0.5f );

  cr_assert( det.confidence >= 0.0f && det.confidence <= 1.0f );
  cr_assert( det.x >= 0.0f && det.x <= 640.0f );
  cr_assert( det.y >= 0.0f && det.y <= 480.0f );
  cr_assert( det.width >= 0.0f && det.x + det.width <= 640.0f );
  cr_assert( det.height >= 0.0f && det.y + det.height <= 480.0f );

  ds_arena_destroy( &ba );
  ds_arena_destroy( &pa );
}
