#include "optimizer.h"
#include "ds_arena.h"
#include <math.h>

static void register_param( _ds_arena_t_ *pa, _adam_optimizer_t *opt, _tensor_t *param )
{
  if ( !param ) return;

  _adam_slot_t *slot = &opt->slots[opt->num_slots++];
  slot->param = param;
  slot->m = tensor_zeros( pa, param->ndim, param->shape );
  slot->v = tensor_zeros( pa, param->ndim, param->shape );
}

_adam_optimizer_t *adam_create( _ds_arena_t_ *pa, _network_t *net, float lr )
{
  _adam_optimizer_t *opt = ARENA_NEW( pa, _adam_optimizer_t );
  opt->lr = lr;
  opt->beta1 = 0.9f;
  opt->beta2 = 0.999f;
  opt->eps = 1e-8f;
  opt->t = 0;

  for ( int i = 0; i < net->num_layers; i++ )
  {
    _layer_t *l = net->layers[i];
    register_param( pa, opt, l->weights );
    register_param( pa, opt, l->bias );
    register_param( pa, opt, l->gamma );
    register_param( pa, opt, l->beta );
  }
  return opt;
}

void adam_step( _adam_optimizer_t *opt )
{
  opt->t++;
  float b1 = opt->beta1, b2 = opt->beta2;

  // Bias-corrected learning rate
  float lr_t = opt->lr * sqrtf( 1.0f - powf( b2, (float)opt->t ) ) / ( 1.0f - powf( b1, (float)opt->t ) );

  for ( int s = 0; s < opt->num_slots; s++ )
  {
    _adam_slot_t *slot = &opt->slots[s];
    _tensor_t *param = slot->param;

    for ( int i = 0; i < param->size; i++ )
    {
      float g = param->gradients[i];

      slot->m->data[i] = b1 * slot->m->data[i] + ( 1.0f - b1 ) * g;
      slot->v->data[i] = b2 * slot->v->data[i] + ( 1.0f - b2 ) * g * g;

      param->data[i] -= lr_t * slot->m->data[i] / ( sqrtf( slot->v->data[i] ) + opt->eps );
    }
  }
}
