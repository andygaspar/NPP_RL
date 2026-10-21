#!/usr/bin/env bash
set -e

g++ -c -Ofast -fopenmp -fPIC pp/ga_cpp/ga_bridge.cc -o pp/ga_cpp/ga_bridge.o
g++ -shared -fopenmp -Wl,-soname,ga_bridge.so -o pp/ga_cpp/ga_bridge.so pp/ga_cpp/ga_bridge.o
rm -f pp/ga_cpp/ga_bridge.o

g++ -c -Ofast -fopenmp -fPIC -std=c++17 -march=native -flto -DEXPORT_SYMBOLS skwp/knapsack_cpp/knap_bridge.cc -o skwp/knapsack_cpp/knap_bridge.o
g++ -shared -fopenmp -Wl,-soname,knap_bridge.so -o skwp/knapsack_cpp/knap_bridge.so skwp/knapsack_cpp/knap_bridge.o
rm -f skwp/knapsack_cpp/knap_bridge.o