#include "layers.h"
#include "ds_arena.h"
#include "tensor.h"
#include <math.h>

// convolutional 2D
static _tensor_t *conv2d_forward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *input )
{
  int N = input->shape[0], C_in = input->shape[1];
  int H = input->shape[2], W = input->shape[3];
  int C_out = self->out_channels;
  int k = self->kernel_size, s = self->stride, p = self->padding;

  int H_out = ( H + 2 * p - k ) / s + 1;
  int W_out = ( W + 2 * p - k ) / s + 1;

  int out_shape[] = { N, C_out, H_out, W_out };
  _tensor_t *output = tensor_zeros( batch_arena, 4, out_shape );

  for ( int n = 0; n < N; n++ )
    for ( int co = 0; co < C_out; co++ )
      for ( int oh = 0; oh < H_out; oh++ )
        for ( int ow = 0; ow < W_out; ow++ )
        {
          float sum = self->bias->data[co];

          for ( int ci = 0; ci < C_in; ci++ )
            for ( int kh = 0; kh < k; kh++ )
              for ( int kw = 0; kw < k; kw++ )
              {
                int ih = oh * s - p + kh;
                int iw = ow * s - p + kw;
                if ( ih < 0 || ih >= H || iw < 0 || iw >= W ) continue;

                int w_idx = ( ( co * C_in + ci ) * k + kh ) * k + kw;
                int i_idx = ( ( n * C_in + ci ) * H + ih ) * W + iw;
                sum += input->data[i_idx] * self->weights->data[w_idx];
              }

          int o_idx = ( ( n * C_out + co ) * H_out + oh ) * W_out + ow;
          output->data[o_idx] = sum;
        }

  self->last_input = input;
  self->last_input = output;

  return output;
}

static _tensor_t *conv2d_backward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *gradients_out )
{
  _tensor_t *input = self->last_input;

  int N = input->shape[0], C_in = input->shape[1];
  int H = input->shape[2], W = input->shape[3];
  int C_out = self->out_channels;
  int k = self->kernel_size, s = self->stride, p = self->padding;
  int H_out = gradients_out->shape[2], W_out = gradients_out->shape[3];

  _tensor_t *gradients_input = tensor_zeros( batch_arena, 4, input->shape );

  for ( int n = 0; n < N; n++ )
    for ( int co = 0; co < C_out; co++ )
      for ( int oh = 0; oh < H_out; oh++ )
        for ( int ow = 0; ow < W_out; ow++ )
        {
          int go_idx = ( ( n * C_out + co ) * H_out + oh ) * W_out + ow;
          float g = gradients_out->data[go_idx];

          self->bias->gradients[co] += g;

          for ( int ci = 0; ci < C_in; ci++ )
            for ( int kh = 0; kh < k; kh++ )
              for ( int kw = 0; kw < k; kw++ )
              {
                int ih = oh * s - p + kh;
                int iw = ow * s - p + kw;
                if ( ih < 0 || ih >= H || iw < 0 || iw >= W ) continue;

                int w_idx = ( ( co * C_in + ci ) * k + kh ) * k + kw;
                int i_idx = ( ( n * C_in + ci ) * H + ih ) * W + iw;

                self->weights->gradients[w_idx] += input->data[i_idx] * g;
                gradients_input->data[i_idx] += self->weights->data[w_idx] * g;
              }
        }

  return gradients_input;
}

_layer_t *conv2d_create( _ds_arena_t_ *param_arena, int in_c, int out_c, int kernel_size, int stride, int padding )
{
  _layer_t *l = ARENA_NEW( param_arena, _layer_t );
  l->type = LAYER_CONV2D;
  l->name = "conv2d";

  l->in_channels = in_c;
  l->out_channels = out_c;
  l->kernel_size = kernel_size;
  l->stride = stride;
  l->padding = padding;

  int w_shape[] = { out_c, in_c, kernel_size, kernel_size };
  float he_std = sqrtf( 2.0f / (float)( in_c * kernel_size * kernel_size ) );
  l->weights = tensor_random_normal( param_arena, 4, w_shape, 0.0f, he_std );
  l->weights->requires_gradients = true; // 1
  l->weights->gradients = ARENA_ARRAY( param_arena, float, l->weights->size );

  int b_shape[] = { out_c };
  l->bias = tensor_zeros( param_arena, 1, b_shape );
  l->bias->requires_gradients = true; // 1;
  l->bias->gradients = ARENA_ARRAY( param_arena, float, l->bias->size );

  l->forward = conv2d_forward;
  l->backward = conv2d_backward;

  return l;
}

_layer_t *relu_create( _ds_arena_t_ *param_arena )
{
}

_layer_t *batchnorm_create( _ds_arena_t_ *param_arena, int channels )
{
}

_layer_t *dense_create( _ds_arena_t_ *param_arena, int in_features, int out_features )
{
}

_layer_t *gap_create( _ds_arena_t_ *param_arena )
{
}
