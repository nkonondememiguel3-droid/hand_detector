#include "ds_arena.h"
#include "tensor.h"
#include <stdbool.h>
#include <stdio.h>

int main(void)
{
  _ds_arena_t_ arena = { 0 };

  int shape[] = { 10, 3, 64, 64 };
  _tensor_t *t = tensor_create( &arena, 4, shape, false );

  for ( int i = 0; i < t->size; i++ ) printf( "%f ", t->data[i] );

  ds_arena_destroy( &arena );

  return 0;
}
