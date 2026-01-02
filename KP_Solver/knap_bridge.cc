#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stack>
#include <omp.h>


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
}
