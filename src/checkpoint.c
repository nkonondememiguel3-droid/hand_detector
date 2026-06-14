#include "checkpoint.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CKPT_MAGIC "HDC1"

#define MAX_PARAMS_PER_LAYER 4

// Fills `out[*count]` with pointers to this layer's learnable tensors,
// in the canonical order used by both save and load.
static void layer_param_tensors( _layer_t *l, _tensor_t *out[MAX_PARAMS_PER_LAYER], int *count )
{
  *count = 0;
  switch ( l->type )
  {
  case LAYER_CONV2D:
  case LAYER_DENSE:
    out[( *count )++] = l->weights;
    out[( *count )++] = l->bias;
    break;
  case LAYER_BATCHNORM:
    out[( *count )++] = l->gamma;
    out[( *count )++] = l->beta;
    out[( *count )++] = l->running_mean;
    out[( *count )++] = l->running_var;
    break;
  case LAYER_RELU:
  case LAYER_GAP:
    break;
  }
}

// save
static bool write_tensor( FILE *f, _tensor_t *t )
{
  int32_t ndim = (int32_t)t->ndim;
  if ( fwrite( &ndim, sizeof( ndim ), 1, f ) != 1 ) return false;

  for ( int i = 0; i < t->ndim; i++ )
  {
    int32_t dim = (int32_t)t->shape[i];
    if ( fwrite( &dim, sizeof( dim ), 1, f ) != 1 ) return false;
  }

  size_t n = (size_t)t->size;
  if ( fwrite( t->data, sizeof( float ), n, f ) != n ) return false;

  return true;
}

bool checkpoint_save( _network_t *net, const char *path )
{
  FILE *f = fopen( path, "wb" );
  if ( !f ) return false;

  bool ok = true;

  ok = ok && fwrite( CKPT_MAGIC, 1, 4, f ) == 4;

  int32_t num_layers = (int32_t)net->num_layers;
  ok = ok && fwrite( &num_layers, sizeof( num_layers ), 1, f ) == 1;

  for ( int i = 0; ok && i < net->num_layers; i++ )
  {
    _layer_t *l = net->layers[i];

    int32_t type = (int32_t)l->type;
    ok = ok && fwrite( &type, sizeof( type ), 1, f ) == 1;

    _tensor_t *params[MAX_PARAMS_PER_LAYER];
    int count;
    layer_param_tensors( l, params, &count );

    int32_t param_count = (int32_t)count;
    ok = ok && fwrite( &param_count, sizeof( param_count ), 1, f ) == 1;

    for ( int p = 0; ok && p < count; p++ ) ok = ok && write_tensor( f, params[p] );
  }

  fclose( f );
  return ok;
}

// Load
// Reads a tensor's header (ndim + shape) and validates it against `t`.
// On success, reads `t->size` floats directly into `t->data`.
// Returns false on read error OR shape mismatch.
static bool read_and_validate_tensor( FILE *f, _tensor_t *t )
{
  int32_t ndim;
  if ( fread( &ndim, sizeof( ndim ), 1, f ) != 1 ) return false;
  if ( ndim != t->ndim ) return false;

  for ( int i = 0; i < ndim; i++ )
  {
    int32_t dim;
    if ( fread( &dim, sizeof( dim ), 1, f ) != 1 ) return false;
    if ( dim != t->shape[i] ) return false;
  }

  size_t n = (size_t)t->size;
  if ( fread( t->data, sizeof( float ), n, f ) != n ) return false;

  return true;
}

// Reads a tensor's header+data into a throwaway heap buffer, discarding
// it. Used during the validation pass so we don't corrupt `net`'s live
// tensors if a LATER layer in the file turns out to be mismatched.
static bool skip_tensor( FILE *f )
{
  int32_t ndim;
  if ( fread( &ndim, sizeof( ndim ), 1, f ) != 1 ) return false;

  size_t size = 1;
  for ( int i = 0; i < ndim; i++ )
  {
    int32_t dim;
    if ( fread( &dim, sizeof( dim ), 1, f ) != 1 ) return false;
    size *= (size_t)dim;
  }

  return fseek( f, (long)( size * sizeof( float ) ), SEEK_CUR ) == 0;
}

bool checkpoint_load( _network_t *net, const char *path )
{
  FILE *f = fopen( path, "rb" );
  if ( !f ) return false;

  bool ok = true;

  char magic[4];
  ok = ok && fread( magic, 1, 4, f ) == 4;
  ok = ok && memcmp( magic, CKPT_MAGIC, 4 ) == 0;

  int32_t num_layers;
  ok = ok && fread( &num_layers, sizeof( num_layers ), 1, f ) == 1;
  ok = ok && num_layers == net->num_layers;

  if ( !ok )
  {
    fclose( f );
    return false;
  }

  // Pass 1: validate everything WITHOUT modifying net
  long data_start = ftell( f );

  for ( int i = 0; ok && i < net->num_layers; i++ )
  {
    _layer_t *l = net->layers[i];

    int32_t type;
    ok = ok && fread( &type, sizeof( type ), 1, f ) == 1;
    ok = ok && type == (int32_t)l->type;

    int32_t param_count;
    ok = ok && fread( &param_count, sizeof( param_count ), 1, f ) == 1;

    _tensor_t *params[MAX_PARAMS_PER_LAYER];
    int count;
    layer_param_tensors( l, params, &count );

    ok = ok && param_count == (int32_t)count;

    for ( int p = 0; ok && p < count; p++ )
    {
      // Validate shape without writing data yet
      int32_t ndim;
      ok = ok && fread( &ndim, sizeof( ndim ), 1, f ) == 1;
      ok = ok && ndim == params[p]->ndim;

      for ( int d = 0; ok && d < ndim; d++ )
      {
        int32_t dim;
        ok = ok && fread( &dim, sizeof( dim ), 1, f ) == 1;
        ok = ok && dim == params[p]->shape[d];
      }

      size_t n = (size_t)params[p]->size;
      ok = ok && fseek( f, (long)( n * sizeof( float ) ), SEEK_CUR ) == 0;
    }
  }

  if ( !ok )
  {
    fclose( f );
    return false;
  }

  // ── Pass 2: re-read from data_start, this time loading data ─
  fseek( f, data_start, SEEK_SET );

  for ( int i = 0; ok && i < net->num_layers; i++ )
  {
    _layer_t *l = net->layers[i];

    int32_t type, param_count;
    ok = ok && fread( &type, sizeof( type ), 1, f ) == 1;
    ok = ok && fread( &param_count, sizeof( param_count ), 1, f ) == 1;

    _tensor_t *params[MAX_PARAMS_PER_LAYER];
    int count;
    layer_param_tensors( l, params, &count );

    for ( int p = 0; ok && p < count; p++ ) ok = ok && read_and_validate_tensor( f, params[p] );
  }

  fclose( f );
  return ok;
}
