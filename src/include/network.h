#ifndef hand_detector_network_h
#define hand_detector_network_h

#include "ds_arena.h"
#include "layers.h"
#include "tensor.h"

#define NETWORK_MAX_LAYERS 64

typedef struct
{
  _layer_t *layers[NETWORK_MAX_LAYERS];
  int num_layers;
} _network_t;

extern _network_t *network_create( _ds_arena_t_ *param_arena );
extern void network_add( _network_t *net, _layer_t *layer );

extern _tensor_t *network_forward( _network_t *net, _ds_arena_t_ *batch_arena, _tensor_t *input );

extern void network_backward( _network_t *net, _ds_arena_t_ *batch_arena, _tensor_t *gradients_output );

extern void network_zero_gradient( _network_t *net );

extern void network_set_training( _network_t *net, int training );

extern _tensor_t *combiine_head_gradients( _ds_arena_t_ *batch_arena, _tensor_t *conf_gradient, _tensor_t *box_gradient, int batch_size,
                                           float box_weight );

#endif // hand_detector_network_h
