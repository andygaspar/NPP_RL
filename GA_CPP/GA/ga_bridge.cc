#include "genetic.h"


extern "C" {


    Genetic* Genetic_(double* upper_bounds_, double* lower_bounds_, double* comm_tax_free, int* n_usr, double* trans_costs, short n_commodities_, short n_paths_, 
                        short pop_size_, short off_size_, double mutation_rate_, short recombination_size_, 
                        short num_threads_, bool verbose_, short seed) {
        return new Genetic(upper_bounds_, lower_bounds_, comm_tax_free, n_usr, trans_costs, n_commodities_, n_paths_, 
                        pop_size_, off_size_, mutation_rate_, recombination_size_, 
                        num_threads_, verbose_, seed);
    }

    void run_genetic_(Genetic* g, double* population, int iterations) {g -> run(population,iterations);}
    double get_gen_best_val_(Genetic* g) {return g -> get_best_val();}
    double* get_population_ (Genetic* g) {return g -> get_population();}
    double* get_vals_ (Genetic* g) {return g-> get_vals();}
    void destroy(Genetic* g) {delete g;}

}