#include "bn_fold.h"
#include <math.h>

void fold_batchnorm_into_conv( _layer_t *conv, _layer_t *bn )
{
  int C_out = conv->out_channels;
  int C_in = conv->in_channels;
  int k = conv->kernel_size;
  int kernel_size = C_in * k * k; // elements per output channel

  for ( int c = 0; c < C_out; c++ )
  {
    float mean = bn->running_mean->data[c];
    float var = bn->running_var->data[c];
    float gamma = bn->gamma->data[c];
    float beta = bn->beta->data[c];

    float scale = gamma / sqrtf( var + bn->bn_eps );

    // W'_c = scale * W_c  (every weight in this output channel's kernel)
    for ( int i = 0; i < kernel_size; i++ ) conv->weights->data[c * kernel_size + i] *= scale;

    // b'_c = scale * (b_c - mean) + beta
    conv->bias->data[c] = scale * ( conv->bias->data[c] - mean ) + beta;
  }
}
