#include "inference.h"
#include <math.h>

static float sigmoidf( float x )
{
  return 1.0f / ( 1.0f + expf( -x ) );
}

static float clampf( float v, float lo, float hi )
{
  if ( v < lo ) return lo;
  if ( v > hi ) return hi;
  return v;
}

_detection_t decode_detection( const float raw[5], int frame_w, int frame_h, float threshold )
{
  _detection_t det;

  det.confidence = sigmoidf( raw[0] );

  float cx = clampf( raw[1], 0.0f, 1.0f );
  float cy = clampf( raw[2], 0.0f, 1.0f );
  float w = clampf( raw[3], 0.0f, 1.0f );
  float h = clampf( raw[4], 0.0f, 1.0f );

  float x0 = ( cx - w / 2.0f ) * (float)frame_w;
  float y0 = ( cy - h / 2.0f ) * (float)frame_h;
  float x1 = ( cx + w / 2.0f ) * (float)frame_w;
  float y1 = ( cy + h / 2.0f ) * (float)frame_h;

  // Clamp the final rectangle to frame bounds, not the normalized inputs
  x0 = clampf( x0, 0.0f, (float)frame_w );
  y0 = clampf( y0, 0.0f, (float)frame_h );
  x1 = clampf( x1, 0.0f, (float)frame_w );
  y1 = clampf( y1, 0.0f, (float)frame_h );

  det.x = x0;
  det.y = y0;
  det.width = x1 - x0;
  det.height = y1 - y0;
  det.detected = det.confidence >= threshold;

  return det;
}

_detection_t run_inference( _network_t *net, _ds_arena_t_ *batch_arena, _tensor_t *input, int frame_w, int frame_h, float threshold )
{
  _tensor_t *output = network_forward( net, batch_arena, input );

  // output shape is (1,5): [conf_logit, cx, cy, w, h]
  float raw[5];
  for ( int i = 0; i < 5; i++ ) raw[i] = output->data[i];

  return decode_detection( raw, frame_w, frame_h, threshold );
}
