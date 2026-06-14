#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "preprocess.h"
#include <string.h>

unsigned char *load_image_raw( _ds_arena_t_ *a, const char *path, int *w, int *h, int *c )
{
  // stbi_load uses malloc internally -- temporary, freed immediately below.
  unsigned char *img = stbi_load( path, w, h, c, 3 ); // force 3 channels
  if ( !img ) return NULL;
  *c = 3;

  size_t size = (size_t)( *w ) * (size_t)( *h ) * (size_t)( *c );
  unsigned char *out = ARENA_ARRAY( a, unsigned char, size );
  memcpy( out, img, size );

  stbi_image_free( img ); // frees stb's malloc'd buffer, NOT arena memory
  return out;
}

unsigned char *resize_image( _ds_arena_t_ *a, const unsigned char *data, int w, int h, int c, int target_w, int target_h )
{
  unsigned char *out = ARENA_ARRAY( a, unsigned char, (size_t)target_w *target_h *c );

  stbir_resize_uint8_linear( data, w, h, 0, out, target_w, target_h, 0, (stbir_pixel_layout)c );
  return out;
}

_tensor_t *hwc_uint8_to_chw_tensor( _ds_arena_t_ *a, const unsigned char *data, int w, int h, int c )
{
  int shape[] = { 1, c, h, w };
  _tensor_t *t = tensor_zeros( a, 4, shape );

  for ( int ch = 0; ch < c; ch++ )
    for ( int row = 0; row < h; row++ )
      for ( int col = 0; col < w; col++ )
      {
        int src_idx = ( row * w + col ) * c + ch; // HWC
        int dst_idx = ( ch * h + row ) * w + col; // CHW (N=1)
        t->data[dst_idx] = (float)data[src_idx] / 255.0f;
      }
  return t;
}

_tensor_t *load_and_preprocess( _ds_arena_t_ *a, const char *path, int target_w, int target_h )
{
  int w, h, c;
  unsigned char *img = load_image_raw( a, path, &w, &h, &c );
  if ( !img ) return NULL;

  unsigned char *resized = resize_image( a, img, w, h, c, target_w, target_h );

  return hwc_uint8_to_chw_tensor( a, resized, target_w, target_h, c );
}
