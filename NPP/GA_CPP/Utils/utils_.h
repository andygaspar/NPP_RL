#include <vector>
#include <functional>
#include <random>
//#include "read_file.cpp"

bool compare(std::array<double,2> a, std::array<double,2> b) { return a[1] > b[1]; }

std::vector<std::array<double,2>> create_list(double* results_l,double* results_h, int num) {
    std::vector<std::array<double,2>> list(num);
    for (int i=0;i<num;++i){
        list[i][0]=i;
        list[i][1]=results_h[i]-results_l[i];
    }
    return list;
}

// double get_normal(double mean, double var) {
//     std::default_random_engine generator(std::rand());
//     std::normal_distribution<double> distribution(mean, var);
//     return distribution(generator);
// }

double compute_distance(std::vector<double> p_1,std::vector<double> p_2) {
    double sum=0;
    for (size_t i=0; i<p_1.size(); i++) {
        sum += std::pow((p_1[i]-p_2[i]),2);
    }
    return std::sqrt(sum);
}
double get_rand(double start, double end) {
        std::default_random_engine generator(std::rand());
        std::uniform_real_distribution<double> distribution(start, end);
        return distribution(generator);
}

size_t get_rand_idx(int start, int end){
    // std::random_device rnd_device;
    std::default_random_engine generator(std::rand());
    std::uniform_int_distribution<size_t> distribution(start, end);
    int i = distribution(generator);
    // std::cout<<i<<"  iiii "<<start<<" "<<end<<std::endl;
    return distribution(generator);
}


template <typename T>
void print_vect(T* vect, int size) {
    for(int i=0; i<size; i++) std::cout<<vect[i]<<" ";
    std::cout<<std::endl;
}


template <typename T>
void print_vector(std::vector<T> &vect) {
    for(size_t i=0; i<vect.size(); i++) std::cout<<vect[i]<<" ";
    std::cout<<std::endl;
}


template <typename T>
void print_vector_and_val(std::vector<T> &vect, double val) {
    for(size_t i=0; i<vect.size(); i++) std::cout<<vect[i]<<" ";
    std::cout<<"   "<<val<<std::endl;
}


double get_mean(std::vector<double> &vect, std::vector<int> &indices,  int size) {
    double mean = 0;
    for(int i=0; i<size; i++) mean += vect[indices[i]];
    return mean/size;
}

double get_std(std::vector<double> &vect, std::vector<int> &indices,  int size) {
    double mean = 0;
    for(int i=0; i<size; i++) mean += vect[indices[i]];
    mean = mean/size;
    double var = 0;
    for(int i=0; i<size; i++) var += pow(mean - vect[indices[i]],2);
    return sqrt(var/size);
}


