#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stack>
#include <omp.h>
#include <cstring>


struct results {
    bool* solution;
};



void knapsack(
    const std::vector<double>& values,
    const std::vector<double>& weights,
    double capacity,
    bool*  solution,
    double eps = 1e-6) {

    int n = values.size();

    // 1. Sort by value/weight ratio (descending)
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    std::sort(indices.begin(), indices.end(),
        [&](int a, int b) {
            return values[a]/weights[a] > values[b]/weights[b];
        });

    std::vector<double> sorted_values(n);
    std::vector<double> sorted_weights(n);
    for (int i = 0; i < n; ++i) {
        sorted_values[i] = values[indices[i]];
        sorted_weights[i] = weights[indices[i]];
    }

    // 2. Branch and bound with depth-first search
    double best_value = 0.0;
    unsigned long long best_mask = 0;

    // Manual stack for DFS (faster than recursion)
    struct Node {
        int depth;
        double weight;
        double value;
        unsigned long long mask;
    };

    std::vector<Node> stack;
    stack.reserve(n * 2);
    stack.push_back({0, 0.0, 0.0, 0ULL});

    while (!stack.empty()) {
        Node node = stack.back();
        stack.pop_back();

        // Compute upper bound for this node
        double bound = node.value;
        double remaining = capacity - node.weight;

        // Linear relaxation: take fractional items
        for (int i = node.depth; i < n; ++i) {
            if (sorted_weights[i] <= remaining + eps) {
                bound += sorted_values[i];
                remaining -= sorted_weights[i];
            } else {
                bound += sorted_values[i] * (remaining / sorted_weights[i]);
                break;
            }
        }

        // Prune if no better solution possible
        if (bound <= best_value + eps) continue;

        // Leaf node: update best solution
        if (node.depth == n) {
            if (node.value > best_value + eps) {
                best_value = node.value;
                best_mask = node.mask;
            }
            continue;
        }

        // Branch order: TAKE item first (most promising when sorted by ratio)
        double new_weight = node.weight + sorted_weights[node.depth];
        if (new_weight <= capacity + eps) {
            // Take item
            stack.push_back({
                node.depth + 1,
                new_weight,
                node.value + sorted_values[node.depth],
                node.mask | (1ULL << node.depth)
            });
        }

        // Skip item
        stack.push_back({
            node.depth + 1,
            node.weight,
            node.value,
            node.mask
        });
    }

    // Convert mask to solution in original order

    for (int i = 0; i < n; ++i) {
        if (best_mask & (1ULL << i)) {
            solution[indices[i]] = true;
        }
    }
}


/*
        short th;
        #pragma omp parallel for num_threads(n_threads) schedule(static) private(th) shared(population, init_pop, tolerance)
        for(short i=0; i < pop_size; i++){
            th = omp_get_thread_num();
*/


extern "C" {
    results* knapsack_(double* p_, double* w_, double* c, int len, int K, int pop_size, int num_procs) {

        bool* sols = new bool[K * len * pop_size];

        omp_set_num_threads(num_procs);

        #pragma omp parallel for schedule(static)
        for(int i=0; i< pop_size; i++) {
            std::vector<double> p = std::vector<double> (len, 0);
            std::vector<double> w = std::vector<double> (len, 0);
            for(int k=0; k< K; k++) {
                for(int j=0; j< len; j++) {
                    p[j] = p_[i*K*len + k*len + j];
                    w[j] = w_[i*K*len + k*len + j];
                    sols[i*K*len + k*len + j] = false;
                }
                knapsack(p, w, c[i*K + k], &sols[i*K*len + k*len]);
            }
        }

        results* res = new results[1];
        res -> solution = sols;

        return res;
        }


    void free_result(results* res){
        delete[] res -> solution;
    }

    void solve_greedy_(const double* p, const double* w, const double* c, double* vals, int M, int K, int P, int L, int num_procs) {
    // Allocate output array
        omp_set_num_threads(num_procs);
//        double* vals = new double[P];

#pragma omp parallel
    {
        // Thread-local storage
        std::vector<int> indices(M);
        std::vector<uint8_t> selected(M);  // uint8_t instead of bool (vector<bool> is slow)

        #pragma omp for schedule(static)
        for (int i = 0; i < P; ++i) {
            double sum_i = 0.0;
            const double* p_i = p + i * K * M;
            const double* w_i = w + i * K * M;
            const double* c_i = c + i * K;

            for (int j = 0; j < K; ++j) {
                int base_idx = j * M;
                double capacity = c_i[j];

                // Initialize indices - unroll small loops
                for (int k = 0; k < M; ++k) indices[k] = k;

                // Sort by efficiency - use a more cache-friendly approach for small M
                // For small M, insertion sort can be faster
                if (M <= 32) {
                    // Insertion sort (good for small arrays)
                    for (int k = 1; k < M; ++k) {
                        int key_idx = indices[k];
                        double key_eff = p_i[base_idx + key_idx] / (w_i[base_idx + key_idx] + 1e-12);
                        int s = k - 1;
                        while (s >= 0 &&
                               (p_i[base_idx + indices[s]] / (w_i[base_idx + indices[s]] + 1e-12)) < key_eff) {
                            indices[s + 1] = indices[s];
                            s--;
                        }
                        indices[s + 1] = key_idx;
                    }
                } else {
                    // Use std::sort with a more efficient comparator
                    std::sort(indices.begin(), indices.end(),
                             [base_idx, p_i, w_i](int a, int b) {
                                 // Pre-compute to avoid repeating divisions
                                 double eff_a = p_i[base_idx + a];
                                 double w_a = w_i[base_idx + a] + 1e-12;
                                 double eff_b = p_i[base_idx + b];
                                 double w_b = w_i[base_idx + b] + 1e-12;
                                 // Compare a/b > c/d as a*d > c*b to avoid divisions
                                 return eff_a * w_b > eff_b * w_a;
                             });
                }

                // Reset selected using memset for speed
                memset(selected.data(), 0, M * sizeof(uint8_t));

                // Greedy selection with early exit
                double cumulative = 0.0;
                for (int s = 0; s < M; ++s) {
                    int orig_idx = indices[s];
                    double weight = w_i[base_idx + orig_idx];

                    // Early exit if even the smallest weight would exceed capacity
                    // (assuming weights are positive)
                    if (cumulative + weight >= capacity) {
                        break;
                    }

                    selected[orig_idx] = 1;
                    cumulative += weight;
                }

                // Sum first L selected items - unroll if L is small
                double sum_ij = 0.0;
                if (L <= 8) {
                    // Small unrolled loop
                    for (int k = 0; k < L; ++k) {
                        sum_ij += selected[k] ? w_i[base_idx + k] : 0.0;
                    }
                } else {
                    // Regular loop
                    for (int k = 0; k < L; ++k) {
                        if (selected[k]) {
                            sum_ij += w_i[base_idx + k];
                        }
                    }
                }
                sum_i += sum_ij;
            }
            vals[i] = sum_i;
        }
    }
    }
}
