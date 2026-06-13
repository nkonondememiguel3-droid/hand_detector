// tests/test_dataloader.c
#include "dataloader.h"
#include "ds_arena.h"
#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

static _ds_arena_t_ a;
static char labels_dir[256];
static char images_dir[256];

static void write_label( const char *stem, const char *content )
{
  char path[512];
  snprintf( path, sizeof( path ), "%s/%s.txt", labels_dir, stem );
  FILE *f = fopen( path, "w" );
  fputs( content, f );
  fclose( f );
}

static void touch_image( const char *stem )
{
  char path[512];
  snprintf( path, sizeof( path ), "%s/%s.jpg", images_dir, stem );
  FILE *f = fopen( path, "w" );
  fputs( "fake-jpg-bytes", f );
  fclose( f );
}

void dl_setup( void )
{
  a = ds_arena_new( 0 );

  snprintf( labels_dir, sizeof( labels_dir ), "/tmp/ds_dataloader_test_labels_%d", getpid() );
  snprintf( images_dir, sizeof( images_dir ), "/tmp/ds_dataloader_test_images_%d", getpid() );

  mkdir( labels_dir, 0755 );
  mkdir( images_dir, 0755 );
}

void dl_teardown( void )
{
  // Clean up temp files
  char cmd[600];
  snprintf( cmd, sizeof( cmd ), "rm -rf %s %s", labels_dir, images_dir );
  system( cmd );

  ds_arena_destroy( &a );
}

TestSuite( dataloader, .init = dl_setup, .fini = dl_teardown );

// Basic loading
Test( dataloader, loads_positive_and_negative_samples )
{
  write_label( "img1", "1 0.5 0.5 0.2 0.3\n" );
  touch_image( "img1" );

  write_label( "img2", "0 0.0 0.0 0.0 0.0\n" );
  touch_image( "img2" );

  _dataset_t *ds = dataset_load( &a, labels_dir, images_dir );

  cr_assert_eq( ds->count, 2 );

  int pos = 0, neg = 0;
  for ( int i = 0; i < ds->count; i++ )
  {
    if ( ds->samples[i].has_hand ) pos++;
    else neg++;
  }
  cr_assert_eq( pos, 1 );
  cr_assert_eq( neg, 1 );
}

Test( dataloader, parses_box_coordinates_correctly )
{
  write_label( "img1", "1 0.512345 0.443210 0.301234 0.398012\n" );
  touch_image( "img1" );

  _dataset_t *ds = dataset_load( &a, labels_dir, images_dir );

  cr_assert_eq( ds->count, 1 );
  cr_assert_float_eq( ds->samples[0].cx, 0.512345f, 1e-5f );
  cr_assert_float_eq( ds->samples[0].cy, 0.443210f, 1e-5f );
  cr_assert_float_eq( ds->samples[0].w, 0.301234f, 1e-5f );
  cr_assert_float_eq( ds->samples[0].h, 0.398012f, 1e-5f );
  cr_assert_eq( ds->samples[0].has_hand, 1 );
}

Test( dataloader, builds_correct_image_path )
{
  write_label( "frame_0042", "1 0.5 0.5 0.1 0.1\n" );
  touch_image( "frame_0042" );

  _dataset_t *ds = dataset_load( &a, labels_dir, images_dir );

  cr_assert_eq( ds->count, 1 );

  char expected[512];
  snprintf( expected, sizeof( expected ), "%s/frame_0042.jpg", images_dir );
  cr_assert_str_eq( ds->samples[0].image_path, expected );
}

// Skipping invalid entries
Test( dataloader, skips_label_without_matching_image )
{
  write_label( "orphan", "1 0.5 0.5 0.1 0.1\n" );
  // no corresponding .jpg

  write_label( "valid", "1 0.5 0.5 0.1 0.1\n" );
  touch_image( "valid" );

  _dataset_t *ds = dataset_load( &a, labels_dir, images_dir );

  cr_assert_eq( ds->count, 1 );

  char expected[512];
  snprintf( expected, sizeof( expected ), "%s/valid.jpg", images_dir );
  cr_assert_str_eq( ds->samples[0].image_path, expected );
}

Test( dataloader, skips_malformed_label_file )
{
  write_label( "broken", "not a valid label\n" );
  touch_image( "broken" );

  write_label( "ok", "1 0.5 0.5 0.1 0.1\n" );
  touch_image( "ok" );

  _dataset_t *ds = dataset_load( &a, labels_dir, images_dir );

  cr_assert_eq( ds->count, 1 );
}

Test( dataloader, ignores_non_txt_files_in_labels_dir )
{
  write_label( "img1", "1 0.5 0.5 0.1 0.1\n" );
  touch_image( "img1" );

  // Drop a stray non-.txt file into labels_dir
  char path[512];
  snprintf( path, sizeof( path ), "%s/readme.md", labels_dir );
  FILE *f = fopen( path, "w" );
  fputs( "not a label", f );
  fclose( f );

  _dataset_t *ds = dataset_load( &a, labels_dir, images_dir );

  cr_assert_eq( ds->count, 1 );
}

Test( dataloader, missing_labels_dir_returns_empty_dataset )
{
  _dataset_t *ds = dataset_load( &a, "/nonexistent/labels", "/nonexistent/images" );

  cr_assert_not_null( ds );
  cr_assert_eq( ds->count, 0 );
}

Test( dataloader, empty_labels_dir_returns_empty_dataset )
{
  _dataset_t *ds = dataset_load( &a, labels_dir, images_dir );

  cr_assert_not_null( ds );
  cr_assert_eq( ds->count, 0 );
}

// Shuffle
Test( dataloader, shuffle_preserves_count_and_changes_order )
{
  _dataset_t *ds = ARENA_NEW( &a, _dataset_t );
  ds->count = 100;
  ds->samples = ARENA_ARRAY( &a, _sample_t, 100 );
  for ( int i = 0; i < 100; i++ ) ds->samples[i].cx = (float)i;

  float before[100];
  for ( int i = 0; i < 100; i++ ) before[i] = ds->samples[i].cx;

  srand( 42 );
  dataset_shuffle( ds );

  cr_assert_eq( ds->count, 100 );

  int changed = 0;
  for ( int i = 0; i < 100; i++ )
    if ( ds->samples[i].cx != before[i] ) changed = 1;

  cr_assert( changed, "Shuffle should change order" );

  // No data lost or duplicated
  int seen[100] = { 0 };
  for ( int i = 0; i < 100; i++ )
  {
    int v = (int)ds->samples[i].cx;
    cr_assert( v >= 0 && v < 100 );
    seen[v]++;
  }
  for ( int i = 0; i < 100; i++ ) cr_assert_eq( seen[i], 1, "Value %d should appear exactly once", i );
}

// Split
Test( dataloader, split_respects_fraction )
{
  _dataset_t *full = ARENA_NEW( &a, _dataset_t );
  full->count = 100;
  full->samples = ARENA_ARRAY( &a, _sample_t, 100 );
  for ( int i = 0; i < 100; i++ ) full->samples[i].cx = (float)i;

  _dataset_t *train, *val;
  dataset_split( &a, full, &train, &val, 0.2f );

  cr_assert_eq( train->count, 80 );
  cr_assert_eq( val->count, 20 );
  cr_assert_eq( train->count + val->count, 100 );
}

Test( dataloader, split_zero_fraction_keeps_all_in_train )
{
  _dataset_t *full = ARENA_NEW( &a, _dataset_t );
  full->count = 10;
  full->samples = ARENA_ARRAY( &a, _sample_t, 10 );

  _dataset_t *train, *val;
  dataset_split( &a, full, &train, &val, 0.0f );

  cr_assert_eq( train->count, 10 );
  cr_assert_eq( val->count, 0 );
}

// Concat
Test( dataloader, concat_combines_multiple_datasets )
{
  _dataset_t *d1 = ARENA_NEW( &a, _dataset_t );
  d1->count = 3;
  d1->samples = ARENA_ARRAY( &a, _sample_t, 3 );
  for ( int i = 0; i < 3; i++ ) d1->samples[i].cx = (float)i; // 0,1,2

  _dataset_t *d2 = ARENA_NEW( &a, _dataset_t );
  d2->count = 2;
  d2->samples = ARENA_ARRAY( &a, _sample_t, 2 );
  for ( int i = 0; i < 2; i++ ) d2->samples[i].cx = (float)( 10 + i ); // 10,11

  _dataset_t *datasets[] = { d1, d2 };
  _dataset_t *combined = dataset_concat( &a, datasets, 2 );

  cr_assert_eq( combined->count, 5 );
  cr_assert_float_eq( combined->samples[0].cx, 0.0f, 1e-6f );
  cr_assert_float_eq( combined->samples[2].cx, 2.0f, 1e-6f );
  cr_assert_float_eq( combined->samples[3].cx, 10.0f, 1e-6f );
  cr_assert_float_eq( combined->samples[4].cx, 11.0f, 1e-6f );
}

// Stats
Test( dataloader, stats_counts_positives_and_negatives )
{
  write_label( "pos1", "1 0.5 0.5 0.1 0.1\n" );
  touch_image( "pos1" );
  write_label( "pos2", "1 0.4 0.4 0.2 0.2\n" );
  touch_image( "pos2" );
  write_label( "neg1", "0 0.0 0.0 0.0 0.0\n" );
  touch_image( "neg1" );

  _dataset_t *ds = dataset_load( &a, labels_dir, images_dir );

  int pos, neg;
  dataset_stats( ds, &pos, &neg );

  cr_assert_eq( pos, 2 );
  cr_assert_eq( neg, 1 );
  cr_assert_eq( pos + neg, ds->count );
}

// Two-pass loading correctness with many files
Test( dataloader, loads_large_directory_correctly )
{
  for ( int i = 0; i < 50; i++ )
  {
    char stem[32];
    snprintf( stem, sizeof( stem ), "img_%03d", i );

    char label[64];
    snprintf( label, sizeof( label ), "%d 0.5 0.5 0.1 0.1\n", i % 2 );
    write_label( stem, label );
    touch_image( stem );
  }

  _dataset_t *ds = dataset_load( &a, labels_dir, images_dir );

  cr_assert_eq( ds->count, 50 );

  int pos, neg;
  dataset_stats( ds, &pos, &neg );
  cr_assert_eq( pos, 25 );
  cr_assert_eq( neg, 25 );
}
