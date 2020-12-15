# cfd-cuda

## How to build

- You need CUDA and cmake to build this project

From root directory of project:
```console
mkdir build
cd build
cmake ..
cmake --build . --config Release
./cfd-cuda <scale-factor> <no. of iterations>
```

- Vary scale factor in change dimensions of the simulation.
- Increasing the number of iterations will calculate more timesteps.

## How to visualize

- Install gnuplot
- In gnuplot run the script inside the file `cuda-cfd.plt`
