#include "dataloader.h"
#include "ds_arena.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define DATASET_INITIAL_CAPACITY 4096

static int file_exists( const char *path )
{
  struct stat st;
  return stat( path, &st ) == 0;
}

_dataset_t *dataset_load( _ds_arena_t_ *a, const char *labels_dir, const char *image_dir )
{
  _dataset_t *ds = ARENA_NEW( a, _dataset_t );

  DIR *d = opendir( labels_dir );
  if ( !d )
  {
    printf("directory not found\n");
    ds->samples = NULL;
    ds->count = 0;
    return ds;
  }

  // count al the .txt files.
  int upper_bound = 0;
  struct dirent *entry;
  while ( ( entry = readdir( d ) ) != NULL )
  {
    size_t len = strlen( entry->d_name );
    if ( len > 4 && strcmp( entry->d_name + len - 4, ".txt" ) == 0 ) upper_bound++;
  }
  closedir( d );

  if ( upper_bound == 0 )
  {
    ds->samples = 0;
    ds->count = 0;
    return ds;
  }

  ds->samples = ARENA_ARRAY( a, _sample_t, upper_bound );
  ds->count = 0;

  // second pass: parse and fill
  d = opendir( labels_dir );
  while ( ( entry = readdir( d ) ) != NULL )
  {
    size_t len = strlen( entry->d_name );
    if ( len <= 4 || strcmp( entry->d_name + len - 4, ".txt" ) != 0 ) continue;

    char label_path[SAMPLE_PATH_MAX];
    snprintf( label_path, sizeof( label_path ), "%s/%s", labels_dir, entry->d_name );

    FILE *f = fopen( label_path, "r" );
    if ( !f ) continue;

    int class_id;
    float cx, cy, w, h;
    int parsed = fscanf( f, "%d %f %f %f %f", &class_id, &cx, &cy, &w, &h );
    fclose( f );

    if ( parsed != 5 ) continue;

    char stem[SAMPLE_PATH_MAX - 4];
    size_t stem_len = len - 4;
    if ( stem_len >= sizeof( stem ) ) continue;
    memcpy( stem, entry->d_name, stem_len );
    stem[stem_len] = '\0';

    char image_path[SAMPLE_PATH_MAX];
    snprintf( image_path, sizeof( image_path ), "%s/%s.jpg", image_dir, stem );

    if ( !file_exists( image_path ) ) continue;

    _sample_t *s = &ds->samples[ds->count];
    strncpy( s->image_path, image_path, SAMPLE_PATH_MAX - 1 );
    s->image_path[SAMPLE_PATH_MAX - 1] = '\0';
    s->cx = cx;
    s->cy = cy;
    s->w = w;
    s->h = h;
    s->has_hand = ( class_id == 1 ) ? true : false;

    ds->count++;
  }
  closedir( d );

  return ds;
}

_dataset_t *dataset_concat( _ds_arena_t_ *a, _dataset_t **datasets, int num_datasets )
{
  int total = 0;
  for ( int i = 0; i < num_datasets; i++ ) total += datasets[i]->count;

  _dataset_t *out = ARENA_NEW( a, _dataset_t );
  out->samples = ARENA_ARRAY( a, _sample_t, total );
  out->count = total;

  int offset = 0;
  for ( int i = 0; i < num_datasets; i++ )
  {
    memcpy( out->samples + offset, datasets[i]->samples, datasets[i]->count * sizeof( _sample_t ) );
    offset += datasets[i]->count;
  }

  return out;
}

void dataset_shuffle( _dataset_t *ds )
{
  for ( int i = ds->count - 1; i > 0; i-- )
  {
    int j = rand() % ( i + 1 );
    _sample_t tmp = ds->samples[i];
    ds->samples[i] = ds->samples[j];
    ds->samples[j] = tmp;
  }
}

void dataset_split( _ds_arena_t_ *a, _dataset_t *full, _dataset_t **train, _dataset_t **val, float val_fraction )
{
  dataset_shuffle( full );

  int val_count   = (int)( full->count * val_fraction );
  int train_count = full->count - val_count;

  *train = ARENA_NEW( a, _dataset_t );
  ( *train )->samples = ARENA_ARRAY( a, _sample_t, train_count );
  ( *train )->count = train_count;
  memcpy( ( *train )->samples, full->samples, train_count * sizeof( _sample_t ) );

  *val = ARENA_NEW( a, _dataset_t );
  ( *val )->samples = ARENA_ARRAY( a, _sample_t, val_count );
  ( *val )->count = val_count;
  memcpy( ( *val )->samples, full->samples + train_count, val_count * sizeof( _sample_t ) );
}

void dataset_stats( _dataset_t *ds, int *positives, int *negatives )
{
  int pos = 0, neg = 0;
  for ( int i = 0; i < ds->count; i++ )
  {
    if ( ds->samples[i].has_hand ) pos++;
    else neg++;
  }
  *positives = pos;
  *negatives = neg;
}
