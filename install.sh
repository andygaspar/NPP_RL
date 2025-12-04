g++ -c -Ofast -fopenmp -fPIC GA_CPP/GA/ga_bridge.cc -o GA_CPP/GA/ga_bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,ga_bridge.so -o GA_CPP/GA/ga_bridge.so GA_CPP/GA/ga_bridge.o -ljsoncpp
rm -f GA_CPP/GA/ga_bridge.o

g++ -c -Ofast -fopenmp -fPIC GA_CPP/GAH/ga_h_bridge.cc -o GA_CPP/GAH/ga_h_bridge.o -ljsoncpp
g++ -shared -fopenmp -Wl,-soname,ga_h_bridge.so -o GA_CPP/GAH/ga_h_bridge.so GA_CPP/GAH/ga_h_bridge.o -ljsoncpp
rm -f GA_CPP/GAH/ga_h_bridge.o
