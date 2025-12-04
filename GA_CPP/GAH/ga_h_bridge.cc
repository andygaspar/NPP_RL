#include "genetic_h.h"


extern "C" {


    GeneticH* GeneticH_(double* upper_bounds_, double* lower_bounds_, double* comm_tax_free, int* n_usr, double* trans_costs, short n_commodities_, short n_paths_, 
                        short pop_size_, short off_size_, double mutation_rate_, short recombination_size_,
                        short heuristic_every_, 
                        short num_threads_, bool verbose_, short seed) {
        return new GeneticH(upper_bounds_, lower_bounds_, comm_tax_free, n_usr, trans_costs, n_commodities_, n_paths_, 
                        pop_size_, off_size_, mutation_rate_, recombination_size_, 
                        heuristic_every_,
                        num_threads_, verbose_, seed);
    }

    void run_genetic_h_(GeneticH* g, double* population, int iterations) {g -> run(population,iterations);}
    double get_gen_best_val_(GeneticH* g) {return g -> get_best_val();}
    double* get_population_ (GeneticH* g) {return g -> get_population();}
    double* get_vals_ (GeneticH* g) {return g-> get_vals();}
    int get_h_iterations (GeneticH* g) {int iter = 0; for(int th=0; th < g->n_threads; th++) iter += g->heuristic_iterations[th]; return iter; }
    void destroy(GeneticH* g) {delete g;}

}