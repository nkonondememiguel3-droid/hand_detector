#ifndef hand_detector_optimizer_h
#define hand_detector_optimizer_h

#include "ds_arena.h"
#include "network.h"
#include "tensor.h"

#define OPTIMIZER_MAX_PARAMS 256

typedef struct
{
  _tensor_t *param;
  _tensor_t *m;
  _tensor_t *v;
} _adam_slot_t;

typedef struct
{
  float lr, beta1, beta2, eps;
  int t;

  _adam_slot_t slots[OPTIMIZER_MAX_PARAMS];
  int num_slots;
} _adam_optimizer_t;

extern _adam_optimizer_t *adam_create( _ds_arena_t_ *param_arena, _network_t *net, float lr );
extern void adam_step( _adam_optimizer_t *opt );

#endif // hand_detector_optimizer_h
