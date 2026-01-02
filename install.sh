g++ -c -Ofast -fopenmp -fPIC NPP/GA_CPP/GA/ga_bridge.cc -o NPP/GA_CPP/GA/ga_bridge.o
g++ -shared -fopenmp -Wl,-soname,ga_bridge.so -o NPP/GA_CPP/GA/ga_bridge.so NPP/GA_CPP/GA/ga_bridge.o
rm -f NPP/GA_CPP/GA/ga_bridge.o

g++ -c -Ofast -fopenmp -fPIC -std=c++17 -march=native -flto -DEXPORT_SYMBOLS KP_Solver/knap_bridge.cc -o KP_Solver/knap_bridge.o
g++ -shared -fopenmp -Wl,-soname,knap_bridge.so -o KP_Solver/knap_bridge.so KP_Solver/knap_bridge.o
rm -f KP_Solver/knap_bridge.o