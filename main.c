#include "dataloader.h"
#include "ds_arena.h"
#include <stdbool.h>
#include <stdio.h>

int main( void )
{
  _ds_arena_t_ arena = { 0 };

  _dataset_t *oxford_ds = dataset_load( &arena, "../datasets/oxford/unified/labels", "../datasets/oxford/unified/images" );
  _dataset_t *egohand_ds = dataset_load( &arena, "../datasets/egohands/unified/labels", "../datasets/egohands/unified/images" );
  _dataset_t *dss[] = { oxford_ds, egohand_ds };
  _dataset_t *full_ds = dataset_concat( &arena, dss, 2 );
  dataset_shuffle( full_ds );
  dataset_shuffle( full_ds );
  _dataset_t *train_ds = NULL, *val_ds = NULL;
  dataset_split( &arena, full_ds, &train_ds, &val_ds, 0.2f );

  printf( "oxford dataset count: %d\n", oxford_ds->count );
  printf( "egohand dataset count: %d\n", egohand_ds->count );
  printf( "full dataset count: %d\n", full_ds->count );
  printf( "train dataset count: %d\n", train_ds->count );
  printf( "value dataset count: %d\n", val_ds->count );
  printf( "---------------------------\n" );

  int positives = 0, negatives = 0;
  dataset_stats( train_ds, &positives, &negatives );

  printf( "Number of positives(hand found) in the training data: %d\n", positives );
  printf( "Number of negatives(hand not found) in the training data: %d\n", negatives );

  dataset_stats( val_ds, &positives, &negatives );

  printf( "Number of positives(hand found) in the test data: %d\n", positives );
  printf( "Number of negatives(hand not found) in the test data: %d\n", negatives );

  ds_arena_destroy( &arena );

  return 0;
}
