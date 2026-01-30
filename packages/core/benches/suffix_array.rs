//! Benchmarks for suffix array construction.
//!
//! Run with: cargo bench --features parallel

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use small_ltsc_core::suffix_array::SuffixArray;

#[cfg(feature = "parallel")]
use small_ltsc_core::suffix_array_parallel::{build_suffix_array_parallel, ParallelSAConfig};

/// Generate tokens with repeated patterns (realistic for compression).
fn generate_repeated_pattern(size: usize, pattern_len: usize) -> Vec<u32> {
    let pattern: Vec<u32> = (0..pattern_len as u32).collect();
    pattern.into_iter().cycle().take(size).collect()
}

/// Generate random tokens (worst case for suffix array).
fn generate_random_tokens(size: usize, vocab_size: u32) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..vocab_size)).collect()
}

/// Generate tokens with some structure (mix of repeated and unique).
fn generate_structured_tokens(size: usize) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(size);
    let mut rng = rand::thread_rng();
    
    // Mix of patterns and random tokens
    let pattern1: Vec<u32> = vec![100, 200, 300, 400, 500];
    let pattern2: Vec<u32> = vec![1000, 1001, 1002];
    
    let mut i = 0;
    while i < size {
        let choice = rng.gen_range(0..3);
        match choice {
            0 => {
                // Add pattern1
                for &t in &pattern1 {
                    if i < size {
                        tokens.push(t);
                        i += 1;
                    }
                }
            }
            1 => {
                // Add pattern2
                for &t in &pattern2 {
                    if i < size {
                        tokens.push(t);
                        i += 1;
                    }
                }
            }
            _ => {
                // Add random token
                tokens.push(rng.gen_range(0..10000));
                i += 1;
            }
        }
    }
    
    tokens.truncate(size);
    tokens
}

fn bench_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_array_sequential");
    
    for size in [1_000, 10_000, 50_000, 100_000].iter() {
        let tokens = generate_repeated_pattern(*size, 5);
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("repeated_pattern", size),
            &tokens,
            |b, tokens| {
                b.iter(|| SuffixArray::build(black_box(tokens)));
            },
        );
    }
    
    for size in [1_000, 10_000, 50_000].iter() {
        let tokens = generate_random_tokens(*size, 1000);
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("random", size),
            &tokens,
            |b, tokens| {
                b.iter(|| SuffixArray::build(black_box(tokens)));
            },
        );
    }
    
    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_array_parallel");
    
    let config = ParallelSAConfig {
        parallel_threshold: 0, // Always use parallel for benchmarking
        ..Default::default()
    };
    
    for size in [10_000, 50_000, 100_000, 200_000].iter() {
        let tokens = generate_repeated_pattern(*size, 5);
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("repeated_pattern", size),
            &tokens,
            |b, tokens| {
                b.iter(|| build_suffix_array_parallel(black_box(tokens), &config));
            },
        );
    }
    
    for size in [10_000, 50_000, 100_000].iter() {
        let tokens = generate_random_tokens(*size, 1000);
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("random", size),
            &tokens,
            |b, tokens| {
                b.iter(|| build_suffix_array_parallel(black_box(tokens), &config));
            },
        );
    }
    
    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_array_comparison");
    
    let config = ParallelSAConfig {
        parallel_threshold: 0,
        ..Default::default()
    };
    
    for size in [10_000, 50_000, 100_000].iter() {
        let tokens = generate_structured_tokens(*size);
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &tokens,
            |b, tokens| {
                b.iter(|| SuffixArray::build(black_box(tokens)));
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &tokens,
            |b, tokens| {
                b.iter(|| build_suffix_array_parallel(black_box(tokens), &config));
            },
        );
    }
    
    group.finish();
}

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_sequential,
    bench_parallel,
    bench_comparison
);

#[cfg(not(feature = "parallel"))]
criterion_group!(benches, bench_sequential);

criterion_main!(benches);
