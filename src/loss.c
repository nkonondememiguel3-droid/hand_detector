#include "loss.h"
#include "tensor.h"
#include <math.h>

_loss_t_ bce_with_logits( _ds_arena_t_ *a, _tensor_t *predicted, _tensor_t *target )
{
  _loss_t_ loss = { 0 };
  loss.gradients = tensor_zeros( a, predicted->ndim, predicted->shape );

  int n = predicted->size;
  float sum = 0.0f;

  for ( int i = 0; i < n; ++i )
  {
    float x = predicted->data[i];
    float y = target->data[i];

    // numerically stable bce-with-logits:
    // max(x,0) - x*y + log(1 + exp(-|x|))
    float p = x > 0.0f ? x : 0.0f;
    sum += p - x * y + logf( 1.0f + expf( -fabsf( x ) ) );

    // dL/dx = sigmoid(x) - y
    float sig = 1.0f / ( 1.0f + expf( -x ) );
    loss.gradients->data[i] = ( sig - y ) / (float)n;
  }

  loss.value = sum / (float)n;

  return loss;
}

_loss_t_ smooth_l1( _ds_arena_t_ *a, _tensor_t *predicted, _tensor_t *target, _tensor_t *mask )
{
  _loss_t_ loss = { 0 };
  loss.gradients = tensor_zeros( a, predicted->ndim, predicted->shape );

  int n = predicted->size;
  float sum = 0.0f;
  int count = 0;

  for ( int i = 0; i < n; ++i )
  {
    if ( mask && mask->data[i] < 0.5f ) continue;

    float d = predicted->data[i] - target->data[i];
    float abs_d = fabsf( d );

    if ( abs_d < 1.0f )
    {
      sum += 0.5f * d * d;
      loss.gradients->data[i] = d;
    }
    else
    {
      sum += abs_d - 0.5f;
      loss.gradients->data[i] = d > 0.0f ? 1.0f : -1.0f;
    }

    count++;
  }

  if ( count > 0 )
  {
    loss.value = sum / (float)count;
    for ( int i = 0; i < n; i++ ) loss.gradients->data[i] /= (float)count;
  }

  return loss;
}
