CXX = g++
CXXFLAGS = -Wall -g -ggdb -std=c++11

vtg: main.o Video.o stabilise.o
	$(CXX) $(CXXFLAGS) -o vtg main.o Video.o stabilise.o `pkg-config --cflags --libs opencv`

main.o: main.cpp
	$(CXX) -g -c main.cpp

Video.o: Video.cpp
	$(CXX) -g -c Video.cpp

stabilise.o: stabilise.cpp
	$(CXX) -g -c stabilise.cpp

clean:
	rm main.o Video.o stabilise.o
