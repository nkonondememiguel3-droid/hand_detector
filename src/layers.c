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

// relu implementation
static _tensor_t *relu_forward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *input )
{
  _tensor_t *output = tensor_zeros( batch_arena, input->ndim, input->shape );
  for ( int i = 0; i < input->size; i++ ) output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;

  self->last_input = input;
  return output;
}

static _tensor_t *relu_backward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *gradients_out )
{
  _tensor_t *gradients_input = tensor_zeros( batch_arena, gradients_out->ndim, gradients_out->shape );
  for ( int i = 0; i < gradients_out->size; i++ ) gradients_input->data[i] = ( self->last_input->data[i] > 0.0f ) ? gradients_out->data[i] : 0.0f;

  return gradients_out;
}

_layer_t *relu_create( _ds_arena_t_ *param_arena )
{
  _layer_t *l = ARENA_NEW( param_arena, _layer_t );
  l->type = LAYER_RELU;
  l->name = "relu";
  l->forward = relu_forward;
  l->backward = relu_backward;

  return l;
}

// batch normalization
static _tensor_t *batchnorm_forward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *input )
{
  int N = input->shape[0], C = input->shape[1];
  int H = input->shape[2], W = input->shape[3];
  int M = N * H * W;

  _tensor_t *output = tensor_zeros( batch_arena, 4, input->shape );
  _tensor_t *normalized = tensor_zeros( batch_arena, 4, input->shape );

  for ( int c = 0; c < C; c++ )
  {
    float mean = 0.0f;
    for ( int n = 0; n < N; n++ )
      for ( int h = 0; h < H; h++ )
        for ( int w = 0; w < W; w++ ) mean += input->data[( ( n * C + c ) * H + h ) * W + w];
    mean /= (float)M;

    float var = 0.0f;
    for ( int n = 0; n < N; n++ )
      for ( int h = 0; h < H; h++ )
        for ( int w = 0; w < W; w++ )
        {
          float d = input->data[( ( n * C + c ) * H + h ) * W + w] - mean;
          var += d * d;
        }
    var /= (float)M;

    self->batch_mean->data[c] = mean;
    self->batch_var->data[c] = var;

    float m = self->bn_momentum;
    self->running_mean->data[c] = ( 1 - m ) * self->running_mean->data[c] + m * mean;
    self->running_var->data[c] = ( 1 - m ) * self->running_var->data[c] + m * var;

    float inv_std = 1.0f / sqrtf( var + self->bn_eps );

    for ( int n = 0; n < N; n++ )
      for ( int h = 0; h < H; h++ )
        for ( int w = 0; w < W; w++ )
        {
          int idx = ( ( n * C + c ) * H + h ) * W + w;
          float xhat = ( input->data[idx] - mean ) * inv_std;
          normalized->data[idx] = xhat;
          output->data[idx] = self->gamma->data[c] * xhat + self->beta->data[c];
        }
  }

  self->last_input = input;
  self->normalized = normalized;

  return output;
}

static _tensor_t *batchnorm_backward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *gradients_out )
{
  _tensor_t *input = self->last_input;
  _tensor_t *xhat = self->normalized;

  int N = input->shape[0], C = input->shape[1];
  int H = input->shape[2], W = input->shape[3];
  int M = N * H * W;

  _tensor_t *gradients_input = tensor_zeros( batch_arena, 4, input->shape );

  for ( int c = 0; c < C; ++c )
  {
    float var = self->batch_var->data[c];
    float inv_std = 1.0f / sqrtf( var + self->bn_eps );
    float gamma_c = self->gamma->data[c];

    float sum_dy = 0.0f, sum_dy_xhat = 0.0f;

    for ( int n = 0; n < N; n++ )
      for ( int h = 0; h < H; h++ )
        for ( int w = 0; w < W; w++ )
        {
          int idx = ( ( n * C + c ) * H + h ) * W + w;
          float dy = gradients_out->data[idx];
          sum_dy += dy;
          sum_dy_xhat += dy * xhat->data[idx];
        }

    self->gamma->gradients[c] += sum_dy_xhat;
    self->beta->gradients[c] += sum_dy;

    for ( int n = 0; n < N; n++ )
      for ( int h = 0; h < H; h++ )
        for ( int w = 0; w < W; w++ )
        {
          int idx = ( ( n * C + c ) * H + h ) * W + w;
          float dy = gradients_out->data[idx];
          float dx = gamma_c * inv_std / (float)M * ( (float)M * dy - sum_dy - xhat->data[idx] * sum_dy_xhat );

          gradients_input->data[idx] = dx;
        }
  }

  return gradients_input;
}

_layer_t *batchnorm_create( _ds_arena_t_ *param_arena, int channels )
{
  _layer_t *l = ARENA_NEW( param_arena, _layer_t );
  l->type = LAYER_BATCHNORM;
  l->name = "batchnorm";

  int shape[] = { channels };

  l->gamma = tensor_ones( param_arena, 1, shape );
  l->gamma->requires_gradients = true;
  l->gamma->gradients = ARENA_ARRAY( param_arena, float, channels );

  l->beta = tensor_zeros( param_arena, 1, shape );
  l->beta->requires_gradients = true;
  l->beta->gradients = ARENA_ARRAY( param_arena, float, channels );

  l->running_mean = tensor_zeros( param_arena, 1, shape );
  l->running_var = tensor_ones( param_arena, 1, shape );
  l->batch_mean = tensor_zeros( param_arena, 1, shape );
  l->batch_var = tensor_zeros( param_arena, 1, shape );

  l->bn_momentum = 0.1f;
  l->bn_eps = 1e-5f;
  l->training = true;

  l->forward = batchnorm_forward;
  l->backward = batchnorm_backward;

  return l;
}

// Dense
static _tensor_t *dense_forward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *input )
{
  int N = input->shape[0];
  int in_feat = input->shape[1];
  int out_feat = self->out_channels;

  int out_shape[] = { N, out_feat };
  _tensor_t *output = tensor_zeros( batch_arena, 2, out_shape );

  for ( int n = 0; n < N; n++ )
    for ( int j = 0; j < out_feat; j++ )
    {
      float sum = self->bias->data[j];
      for ( int i = 0; i < in_feat; i++ ) sum += self->weights->data[j * in_feat + i] * input->data[n * in_feat + i];
      output->data[n * out_feat + j] = sum;
    }

  self->last_input = input;
  return output;
}

static _tensor_t *dense_backward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *gradients_out )
{
  _tensor_t *input = self->last_input;
  int N = input->shape[0];
  int in_feat = input->shape[1];
  int out_feat = self->out_channels;

  _tensor_t *gradients_input = tensor_zeros( batch_arena, 2, input->shape );

  for ( int n = 0; n < N; n++ )
    for ( int j = 0; j < out_feat; j++ )
    {
      float g = gradients_out->data[n * out_feat + j];
      self->bias->gradients[j] += g;

      for ( int i = 0; i < in_feat; i++ )
      {
        self->weights->gradients[j * in_feat + i] += g * input->data[n * in_feat + i];
        gradients_input->data[n * in_feat + i] += g * self->weights->data[j * in_feat + i];
      }
    }

  return gradients_input;
}

_layer_t *dense_create( _ds_arena_t_ *param_arena, int in_features, int out_features )
{
  _layer_t *l = ARENA_NEW( param_arena, _layer_t );
  l->type = LAYER_DENSE;
  l->name = "dense";

  l->in_channels = in_features;
  l->out_channels = out_features;

  int w_shape[] = { out_features, in_features };
  float he_std = sqrtf( 2.0f / (float)in_features );
  l->weights = tensor_random_normal( param_arena, 2, w_shape, 0.0f, he_std );
  l->weights->requires_gradients = true;
  l->weights->gradients = ARENA_ARRAY( param_arena, float_t, l->weights->size );

  int b_shape[] = { out_features };
  l->bias = tensor_zeros( param_arena, 1, b_shape );
  l->bias->requires_gradients = true;
  l->bias->gradients = ARENA_ARRAY( param_arena, float, l->bias->size );

  l->forward = dense_forward;
  l->backward = dense_backward;

  return l;
}

// global average pooling
static _tensor_t *gap_forward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *input )
{
  int N = input->shape[0], C = input->shape[1];
  int H = input->shape[2], W = input->shape[3];

  int out_shape[] = { N, C };
  _tensor_t *output = tensor_zeros( batch_arena, 2, out_shape );

  for ( int n = 0; n < N; n++ )
    for ( int c = 0; c < C; c++ )
    {
      float sum = 0.0f;
      for ( int h = 0; h < H; h++ )
        for ( int w = 0; w < W; w++ ) sum += input->data[( ( n * C + c ) * H + h ) * W + W];
      output->data[n * C + c] = sum / (float)( H * W );
    }

  self->last_input = input;
  return output;
}

static _tensor_t *gap_backward( _layer_t *self, _ds_arena_t_ *batch_arena, _tensor_t *gradients_out )
{
  _tensor_t *input = self->last_input;
  int N = input->shape[0], C = input->shape[1];
  int H = input->shape[2], W = input->shape[3];

  _tensor_t *gradients_input = tensor_zeros( batch_arena, 4, input->shape );
  float scale = 1.0f / (float)( H * W );

  for ( int n = 0; n < N; n++ )
    for ( int c = 0; c < C; c++ )
    {
      float g = gradients_out->data[n * C + c] * scale;
      for ( int h = 0; h < H; h++ )
        for ( int w = 0; w < W; w++ ) gradients_input->data[( ( n * C + c ) * H + h ) * W + w] = g;
    }

  return gradients_input;
}

_layer_t *gap_create( _ds_arena_t_ *param_arena )
{
  _layer_t *l = ARENA_NEW( param_arena, _layer_t );
  l->type = LAYER_GAP;
  l->name = "gap";
  l->forward = gap_forward;
  l->backward = gap_backward;

  return l;
}
