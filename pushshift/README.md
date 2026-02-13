# Prerequisites
Need a C++20 compiler and install the following dependencies:
```bash
apt install -y build-essential cmake git libboost-all-dev
```

# Building and Running
```
# Configure with CMake
cmake -B ./build/
# Build
cmake --build ./build/ --config Release --target filterreddit
# Launch
chmod +x ./build/src/Release/filterreddit
./build/src/Release/filterreddit
```
