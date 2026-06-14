#include "augmentation.h"
#include "ds_arena.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void augmentation_horizontal_flip( unsigned char *data, int w, int h, int numb_channels, _sample_t *sample )
{
  for ( int row = 0; row < h; row++ )
  {
    for ( int col = 0; col < w / 2; col++ )
    {
      int left = ( row * w + col ) * numb_channels;
      int right = ( row * w + ( w - 1 - col ) ) * numb_channels;

      for ( int ch = 0; ch < numb_channels; ch++ )
      {
        unsigned char tmp = data[left + ch];
        data[left + ch] = data[right + ch];
        data[right + ch] = tmp;
      }
    }
  }

  if ( sample->has_hand ) sample->cx = 1.0f - sample->cx;
}

void augmentation_birghtness( unsigned char *data, int size, float factor )
{
  for ( int i = 0; i < size; i++ )
  {
    float v = (float)data[i] * factor;
    if ( v < 0.0f ) v = 0.0f;
    if ( v > 255.0f ) v = 255.0f;
    data[i] = (unsigned char)v;
  }
}

void augmentation_contrast( unsigned char *data, int size, float factor )
{
  for ( int i = 0; i < size; i++ )
  {
    float v = ( (float)data[i] - 127.5f ) * factor + 127.5f;
    if ( v < 0.0f ) v = 0.0f;
    if ( v > 255.0f ) v = 255.0f;
    data[i] = (unsigned char)v;
  }
}

static float gaussian_sample( float sigma )
{
  float u1 = (float)rand() / (float)RAND_MAX;
  float u2 = (float)rand() / (float)RAND_MAX;
  if ( u1 < 1e-7f ) u1 = 1e-7f; // avoid log(0)

  return sigma * sqrtf( -2.0f * logf( u1 ) ) * cosf( 2.0f * (float)M_PI * u2 );
}

void augmenttion_gaussian_noise( unsigned char *data, int size, float sigma )
{
  for ( int i = 0; i < size; i++ )
  {
    float v = (float)data[i] + gaussian_sample( sigma * 255.0f );
    if ( v < 0.0 ) v = 0.0f;
    if ( v > 255.0f ) v = 255.0f;
    data[i] = (unsigned char)v;
  }
}

unsigned char *augmenttion_random_crop( _ds_arena_t_ *a, const unsigned char *data, int *w, int *h, int c, _sample_t *sample, int crop_w, int crop_h,
                                        int x0, int y0 )
{
  int W = *w, H = *h;
  int x1 = x0 + crop_w;
  int y1 = y0 + crop_h;

  // update the label
  if ( sample->has_hand )
  {
    float box_x0 = ( sample->cx - sample->w / 2.0f ) * (float)W;
    float box_y0 = ( sample->cy - sample->h / 2.0f ) * (float)H;
    float box_x1 = ( sample->cx + sample->w / 2.0f ) * (float)W;
    float box_y1 = ( sample->cy + sample->h / 2.0f ) * (float)H;

    float new_x0 = fmaxf( box_x0, (float)x0 );
    float new_x1 = fminf( box_x1, (float)x1 );
    float new_y0 = fmaxf( box_y0, (float)y0 );
    float new_y1 = fminf( box_y1, (float)y1 );

    if ( new_x1 <= new_x0 || new_y1 <= new_y0 )
    {
      sample->has_hand = false;
      sample->cx = sample->cy = sample->w = sample->h = 0.0f;
    }
    else
    {
      sample->cx = ( ( new_x0 + new_x1 ) / 2.0f - (float)x0 ) / (float)crop_w;
      sample->cy = ( ( new_y0 + new_y1 ) / 2.0f - (float)y0 ) / (float)crop_h;
      sample->w = ( new_x1 - new_x0 ) / (float)crop_w;
      sample->h = ( new_y1 - new_y0 ) / (float)crop_h;
    }
  }

  // crop pixels into a fresh arean buffer
  unsigned char *cropped = ARENA_ARRAY( a, unsigned char, (size_t)crop_w *crop_h *c );

  for ( int row = 0; row < crop_h; row++ )
    memcpy( cropped + (size_t)row * crop_w * c, data + ( (size_t)( y0 + row ) * W + x0 ) * c, (size_t)crop_w * c );

  *w = crop_w;
  *h = crop_h;
  return cropped;
}
