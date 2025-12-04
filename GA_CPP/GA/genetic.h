#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <cstdlib>
#include "../Utils/utils_.h"


class Genetic {
    public:
    // problem related features
    short pop_size;
    short n_paths;
    short off_size;
    short pop_total_size;
    
    short n_commodities;
    double tolerance;

    std::vector<std::vector<double>> population;
    std::vector<double> vals;
  
    std::vector<int> a_combs;
    std::vector<int> b_combs;
    double mutation_rate;
    short recombination_size;
    std::vector<int> random_order;
    std::vector<std::vector<int>> random_element_order;


    short n_threads;
    std::vector<int> indices;
    short start_index;
    

    double current_run_val;
    double init_commodity_val;

    std::vector<std::vector<double>> commodities_tax_free;
    std::vector<std::vector<std::vector<double>>> transfer_costs;
    std::vector<std::vector<int>> n_users;
    std::vector<std::vector<double>> upper_bounds;
    std::vector<std::vector<double>> lower_bounds;

    bool verbose;
    std::vector<std::default_random_engine> generators;

    double best_val;

    int no_improvement;



    Genetic(double* upper_bounds_, double* lower_bounds_, double* comm_tax_free, int* n_usr, double* trans_costs, short n_commodities_, short n_paths_, 
    short pop_size_, short off_size_, double mutation_rate_, short recombination_size_, 
    short num_threads_, bool verbose_, short seed){

    pop_size = pop_size_;
    n_paths = n_paths_;
    off_size = off_size_;
  

    pop_total_size = pop_size + off_size;

    no_improvement = 0;
    best_val = 0;
    tolerance = pow(10, -9);

    

    mutation_rate = mutation_rate_;
    recombination_size = recombination_size_;
    n_threads = num_threads_;
    start_index = pop_size * n_paths;

    verbose = verbose_;

    std::mt19937 r;
    generators = std::vector<std::default_random_engine> (0);
    for (int i = 0; i <  n_threads; i++) {
        std::default_random_engine engine;
        if(seed>=0) engine.seed(i);
        else engine = std::default_random_engine(r());
        generators.emplace_back(engine);

    }

    population = std::vector<std::vector<double>> (pop_total_size, std::vector<double>(n_paths, 0));
    vals = std::vector<double> (pop_total_size, 0);

    n_commodities = n_commodities_;
    init_commodity_val = init_commodity_val=pow(10, 5);
    
    indices = std::vector<int> (pop_total_size);
    std::iota(indices.begin(), indices.begin() + pop_total_size, 0);
    
    commodities_tax_free = std::vector<std::vector<double>> (n_threads, std::vector<double>(n_commodities, 0));;
    for(int i=0; i< n_threads; i++){
        for(int j=0; j< n_commodities; j++) commodities_tax_free[i][j] = comm_tax_free[j];
        }


    transfer_costs = std::vector<std::vector<std::vector<double>>> 
                    (n_threads, std::vector<std::vector<double>>(n_commodities, std::vector<double> (n_paths, 0)));
    for(int i=0; i< n_threads; i++){
        for(int j =0; j<n_commodities; j++)  {
            for(int k=0; k< n_paths; k++) transfer_costs[i][j][k]=trans_costs[j*n_paths + k];
            }
        }

    n_users = std::vector<std::vector<int>> (n_threads, std::vector<int>(n_commodities, 0));
    for(int i=0; i< n_threads; i++){
        for(int j=0; j< n_commodities; j++) n_users[i][j] = n_usr[j];
    }

    a_combs = std::vector<int> (0);
    b_combs = std::vector<int> (0);
    for(short i=0; i < pop_size - 1; i++){
        for(short j=i + 1; j < pop_size; j++) {
            a_combs.push_back(i);
            b_combs.push_back(j);
        }
    }

    random_order = std::vector<int> (a_combs.size());
    for(size_t i=0; i<a_combs.size(); i++) random_order[i] = i;

    random_element_order = std::vector<std::vector<int>> (n_threads, std::vector<int> (n_paths));
    for(int i=0; i<n_threads; i++){
        for(int j=0; j<n_paths; j++) random_element_order[i][j] = j;
        }

    upper_bounds = std::vector<std::vector<double>> (n_threads, std::vector<double>(n_paths, 0));;
    for(int i=0; i<n_threads; i++){
        for(int j=0; j<n_paths; j++) upper_bounds[i][j] = upper_bounds_[j];
        }

    lower_bounds = std::vector<std::vector<double>> (n_threads, std::vector<double>(n_paths, 0));;
    for(int i=0; i<n_threads; i++){
        for(int j=0; j<n_paths; j++) lower_bounds[i][j] = lower_bounds_[j];
        }
    }

    ~Genetic(){
    }


    double get_best_val() {return best_val;}

    double* get_population () {
        double* out_pop = new double[pop_size*n_paths];
        for(short i=0; i < pop_size; i++) {
            for(short j=0; j< n_paths; j++) out_pop[i*n_paths + j] = population[indices[i]][j];
        }
        return out_pop;
    }

    double* get_vals () {
        double* out_vals = new double[pop_size];
        for(short j=0; j< pop_size; j++) out_vals[j] = vals[indices[j]];
        return out_vals;
    }

    void reshuffle_element_order(std::vector<int> &vect){
        size_t idx, temp;
        std::uniform_int_distribution<size_t> distribution = std::uniform_int_distribution<size_t> (0, vect.size() - 1);
        int th = omp_get_thread_num();
        for(size_t i=0; i<vect.size(); i++) {
            idx = distribution(generators[th]);
            temp = vect[idx];
            vect[idx] = vect[i];
            vect[i] = temp;
        }
    }

    void fill_random_velocity_vect(std::vector<std::vector<double>> & v, double start, double end) {
        std::uniform_real_distribution<double> distribution(start, end);
        size_t j;
        for(size_t i=0; i < v.size(); i++){
            for(j=0; j < v.size(); j++){
                v[i][j] = distribution(generators[0]);
            }
        }
    }       
   

    void init_population(double* init_pop){
        short th;
        #pragma omp parallel for num_threads(n_threads) schedule(static) private(th) shared(population, init_pop, tolerance)
        for(short i=0; i < pop_size; i++){
            th = omp_get_thread_num();
            for(short j=0; j < n_paths; j++) population[i][j] = init_pop[i*n_paths + j];

            vals[i] = eval(population[i], transfer_costs[th], commodities_tax_free[th], n_users[th], n_commodities, n_paths, tolerance);
        }
        double maxval = 0;
        for(short i=0; i < pop_size; i++) if(vals[i] > maxval) {maxval = vals[i];}
    }


    void generate(const std::vector<double> &a_parent, const std::vector<double> &b_parent, std::vector<double> &child, 
                    std::vector<double> & u_bounds, std::vector<double> & l_bounds,
                    std::vector<int> &element_order, double m_rate, short num_paths, short recomb_size, std::default_random_engine gen){
            int i,iter, idx;
            short r_size = recomb_size;

            for(i=0; i<num_paths; i++) child[i] = a_parent[i];
            reshuffle_element_order(element_order);
            // randomly select component from b_parent
            for(i=0; i < r_size; i++) {
                child[element_order[i]] = b_parent[element_order[i]];

            }
            int th = omp_get_thread_num();
            std::uniform_real_distribution<double> distribution_;
            std::uniform_real_distribution<double> distribution_mutation(0., 1.);
            for(i=0; i<num_paths; i++) {
                
                if(distribution_mutation(generators[th]) < m_rate) {
                    std::uniform_real_distribution<double> distribution_(l_bounds[i], u_bounds[i]);
                    child[i] = distribution_(generators[th]);
                    }
            }

            // for(i=0; i<n_paths; i++) {if(get_rand(0., 1.) < mutation_rate) child[i] = get_rand(0., u_bounds[i]);}
            // print_vect(child, n_paths);

        }



    void run(double* init_pop, int iterations){
        init_population(init_pop);

        size_t k;
        int j, p;
        double std;
        
        double m_rate = mutation_rate;

        for(int iter= 0; iter<iterations; iter ++) {
        
            short th;
            reshuffle_element_order(random_order);
            // print_pop();
            #pragma omp parallel for num_threads(n_threads) default(none) private(th) shared(upper_bounds, population, transfer_costs, commodities_tax_free, a_combs, b_combs, random_order, random_element_order, n_commodities, init_commodity_val, n_paths, recombination_size, generators, tolerance)
            for(short i=0; i < off_size; i++) {
                th = omp_get_thread_num();
                generate(population[indices[a_combs[random_order[i]]]], 
                        population[indices[b_combs[random_order[i]]]], 
                        population[indices[pop_size + i]], 
                        upper_bounds[th], lower_bounds[th], random_element_order[th], mutation_rate, n_paths, recombination_size, generators[th]);
                vals[indices[pop_size + i]] = eval(population[indices[pop_size + i]], transfer_costs[th], commodities_tax_free[th], n_users[th], n_commodities, n_paths, tolerance);
                }
            

            argsort(vals, indices, pop_total_size);
                if(vals[indices[0]] > best_val) {
                    best_val = vals[indices[0]];
                    no_improvement = 0;
            }
            else no_improvement ++;

            if(no_improvement >= 500) {
                restart_population();
                //std::cout<<"restarted"<<std::endl;
                
                no_improvement = 0;
            }

            std = get_std(vals, indices, pop_size);
            if(verbose and  iter%100 == 0 and iter > 0) std::cout<<"iteration "<< iter<<"    "<<vals[indices[0]]<<"   mean " <<get_mean(vals, indices, pop_size)<<"   std " <<std<<"   no impr " <<no_improvement<<std::endl;
            if(std < 0.0000001) {restart_population(); }//std::cout<<"restarted "<<std<<std::endl;}
        }
        
        best_val = vals[indices[0]];
        if(verbose) std::cout<<"final iteration "<<best_val<<std::endl;
        
        

        
    }

    double eval(const std::vector<double> &p,const std::vector<std::vector<double>> & trans_costs, const std::vector<double> &comm_tax_free, const std::vector<int> n_usr, const short n_comm, const short num_paths, const double tol){

        /* compute objective value */
        double current_run_val=0;
        double path_price;
        int i,j;
        double minimal_cost;
        double leader_profit;

        for(i=0; i<n_comm; i++) {

            leader_profit = 0;
            minimal_cost = comm_tax_free[i];

            for(j=0; j< num_paths; j++) {

                path_price = p[j] + trans_costs[i][j];

                if(path_price <= minimal_cost + tol) {
                    if (path_price < minimal_cost - tol) {
                        minimal_cost = path_price;
                        leader_profit = p[j];
                    }
                    else {
                        if ( p[j] > leader_profit) {
                            minimal_cost = path_price;
                            leader_profit = p[j];
                        }
                    }
                }
            }

        current_run_val += leader_profit*n_usr[i];

        }


        return current_run_val;
    }



    void argsort(const std::vector<double> &array, std::vector<int> &indeces, int end) {
        std::iota(indices.begin(), indices.begin() + end, 0);
        std::sort(indices.begin(), indices.begin() + end,
                [&array](int left, int right) -> bool {
                    // sort indices according to corresponding array element
                    return array[left] > array[right];
                });
    }


    void restart_population() { 
        short th;
        #pragma omp parallel for num_threads(n_threads) default(none) private(th) shared(upper_bounds, population, transfer_costs, commodities_tax_free, n_commodities, n_paths, generators, tolerance)
        for(int i=1; i < pop_size; i++){
            std::uniform_real_distribution<double> distribution;       
            th = omp_get_thread_num();
            for(int j=0; j < n_paths; j++) {
                distribution = std::uniform_real_distribution<double> (lower_bounds[th][i], upper_bounds[th][j]);
                population[indices[i]][j] = distribution(generators[th]);
            }
            vals[indices[i]] = eval(population[indices[i]], transfer_costs[th], commodities_tax_free[th], n_users[th], n_commodities, n_paths, tolerance);

        }
    }


    void print_pop(){
        for(int p=0; p<pop_total_size; p++){
            for(int j =0; j<n_paths; j++)std::cout<<population[indices[p]][j]<<" ";
            std::cout<<"     "<<vals[indices[p]]<<std::endl;
        }
        std::cout<<std::endl;
    }

};


   









