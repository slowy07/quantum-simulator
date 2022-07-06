g++ -O3 -march=native -fopenmp -o clfsim_base.x clfsim_base.cc
g++ -O3 -march=native -fopenmp -o clfsim_von_neumann.x clfsim_von_neumann.cc
g++ -O3 -march=native -fopenmp -o clfsim_amplitudes.x clfsim_amplitudes.cc
g++ -O3 -march=native -fopenmp -o clfsimh_base.x clfsimh_base.cc
g++ -O3 -march=native -fopenmp -o clfsimh_amplitudes.x clfsimh_amplitudes.cc

nvcc -O3 -o clfsim_base_cuda.x clfsim_base_cuda.cu
nvcc -O3 -o clfsim_qtrajectory_cuda.x clfsim_qtrajectory_cuda.cu

CUSTATEVECFLAGS="-I${CUQUANTUM_DIR}/include -L${CUQUANTUM_DIR}/lib -L${CUQUANTUM_DIR}/lib64 -lcustatevec -lcublas"
nvcc -O3 $CUSTATEVECFLAGS -o clfsim_base_custatevec.x clfsim_base_custatevec.cu
