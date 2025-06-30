// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

use std::cmp::Ordering;

// EVOLVE-BLOCK-START
// Initial implementation: Simple quicksort
// This can be evolved to:
// - Hybrid algorithms (introsort, timsort-like)
// - Adaptive pivot selection
// - Special handling for nearly sorted data
// - Switching to different algorithms based on data characteristics

pub fn adaptive_sort<T: Ord + Clone>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }
    
    // Use quicksort as the base implementation
    quicksort(arr, 0, arr.len() - 1);
}

fn quicksort<T: Ord + Clone>(arr: &mut [T], low: usize, high: usize) {
    if low < high {
        let pivot_index = partition(arr, low, high);
        
        // Recursively sort elements before and after partition
        if pivot_index > 0 {
            quicksort(arr, low, pivot_index - 1);
        }
        quicksort(arr, pivot_index + 1, high);
    }
}

fn partition<T: Ord + Clone>(arr: &mut [T], low: usize, high: usize) -> usize {
    // Choose the last element as pivot (can be evolved to use better strategies)
    let pivot = arr[high].clone();
    let mut i = low;
    
    for j in low..high {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    
    arr.swap(i, high);
    i
}

// Helper function to detect if array is nearly sorted
fn is_nearly_sorted<T: Ord>(arr: &[T], threshold: f64) -> bool {
    if arr.len() <= 1 {
        return true;
    }
    
    let mut inversions = 0;
    let max_inversions = ((arr.len() * (arr.len() - 1)) / 2) as f64 * threshold;
    
    for i in 0..arr.len() - 1 {
        for j in i + 1..arr.len() {
            if arr[i] > arr[j] {
                inversions += 1;
                if inversions as f64 > max_inversions {
                    return false;
                }
            }
        }
    }
    
    true
}

// Helper function for insertion sort (useful for small arrays)
fn insertion_sort<T: Ord>(arr: &mut [T]) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j - 1] > arr[j] {
            arr.swap(j, j - 1);
            j -= 1;
        }
    }
}
// EVOLVE-BLOCK-END

// Benchmark function to test the sort implementation
pub fn run_benchmark(test_data: Vec<Vec<i32>>) -> BenchmarkResults {
    let mut results = BenchmarkResults {
        times: Vec::new(),
        correctness: Vec::new(),
        adaptability_score: 0.0,
    };
    
    for data in test_data {
        let mut arr = data.clone();
        let start = std::time::Instant::now();
        
        adaptive_sort(&mut arr);
        
        let elapsed = start.elapsed();
        results.times.push(elapsed.as_secs_f64());
        
        // Check if correctly sorted
        let is_sorted = arr.windows(2).all(|w| w[0] <= w[1]);
        results.correctness.push(is_sorted);
    }
    
    // Calculate adaptability score based on performance variance
    if results.times.len() > 1 {
        let mean_time: f64 = results.times.iter().sum::<f64>() / results.times.len() as f64;
        let variance: f64 = results.times.iter()
            .map(|t| (t - mean_time).powi(2))
            .sum::<f64>() / results.times.len() as f64;
        
        // Lower variance means better adaptability
        results.adaptability_score = 1.0 / (1.0 + variance.sqrt());
    }
    
    results
}

#[derive(Debug)]
pub struct BenchmarkResults {
    pub times: Vec<f64>,
    pub correctness: Vec<bool>,
    pub adaptability_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_sort() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6];
        adaptive_sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }
    
    #[test]
    fn test_empty_array() {
        let mut arr: Vec<i32> = vec![];
        adaptive_sort(&mut arr);
        assert_eq!(arr, vec![]);
    }
    
    #[test]
    fn test_single_element() {
        let mut arr = vec![42];
        adaptive_sort(&mut arr);
        assert_eq!(arr, vec![42]);
    }
}