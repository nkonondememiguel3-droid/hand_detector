#include "network.h"
#include "tensor.h"

_network_t *network_create( _ds_arena_t_ *pa )
{
  _network_t *net = ARENA_NEW( pa, _network_t );
  return net;
}

void network_add( _network_t *net, _layer_t *layer )
{
  net->layers[net->num_layers++] = layer;
}

_tensor_t *network_forward( _network_t *net, _ds_arena_t_ *ba, _tensor_t *input )
{
  _tensor_t *current = input;
  for ( int i = 0; i < net->num_layers; i++ ) current = net->layers[i]->forward( net->layers[i], ba, current );
  return current;
}

void network_backward( _network_t *net, _ds_arena_t_ *ba, _tensor_t *grad_output )
{
  _tensor_t *grad = grad_output;
  for ( int i = net->num_layers - 1; i >= 0; i-- ) grad = net->layers[i]->backward( net->layers[i], ba, grad );
  // `grad` here is dL/d(network input) — typically unused, lives in
  // batch_arena and is reclaimed on the next ds_arena_reset_to.
}

void network_zero_gradient( _network_t *net )
{
  for ( int i = 0; i < net->num_layers; i++ )
  {
    _layer_t *l = net->layers[i];
    if ( l->weights ) tensor_zero_gradient( l->weights );
    if ( l->bias ) tensor_zero_gradient( l->bias );
    if ( l->gamma ) tensor_zero_gradient( l->gamma );
    if ( l->beta ) tensor_zero_gradient( l->beta );
  }
}

void network_set_training( _network_t *net, int training )
{
  for ( int i = 0; i < net->num_layers; i++ )
    if ( net->layers[i]->type == LAYER_BATCHNORM ) net->layers[i]->training = training;
}

_tensor_t *combiine_head_gradients( _ds_arena_t_ *ba, _tensor_t *conf_grad, _tensor_t *box_grad, int batch_size, float box_weight )
{
  int shape[] = { batch_size, 5 };
  _tensor_t *combined = tensor_zeros( ba, 2, shape );

  for ( int n = 0; n < batch_size; n++ )
  {
    combined->data[n * 5 + 0] = conf_grad->data[n];
    for ( int k = 0; k < 4; k++ ) combined->data[n * 5 + 1 + k] = box_weight * box_grad->data[n * 4 + k];
  }
  return combined;
}
