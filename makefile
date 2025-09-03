# HIP Port of CUDA Commission by Andrew

# flags
override CFLAGS += -O3
override CXXFLAGS += -O3 -std=c++20 -I asio/asio/include -DOMISSION_LARGE_BIOMES=0
override HIPCC_FLAGS += $(CXXFLAGS) -munsafe-fp-atomics --offload-arch=gfx1201

# compiler
CXX := hipcc
CC := hipcc

# source files 
CXX_SRC := src/client.cpp src/cpu.cpp src/main.cpp src/server.cpp
HIP_SRC := src/gpu.hip.cpp

CXX_OBJ := $(CXX_SRC:.cpp=.obj)
C_OBJ := $(C_SRC:.c=.obj)
HIP_OBJ := $(HIP_SRC:.hip.cpp=.obj)

all: main.exe

main.exe: $(CXX_OBJ) $(C_OBJ) $(HIP_OBJ)
	$(CXX) $(CXX_OBJ) $(C_OBJ) $(HIP_OBJ) -o $@ $(HIPCC_FLAGS) -D_WIN32_WINNT=0x0601

%.obj: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)

%.obj: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

%.obj: %.hip.cpp
	$(CXX) -c $< -o $@ $(HIPCC_FLAGS)

# so you made a mistake while changing the code and now you want to know how to clean it up don't worry i understand i got you
.PHONY: clean
clean:
	rm -f main.exe *.obj src/*.obj cubiomes/*.obj
