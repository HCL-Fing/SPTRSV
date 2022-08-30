SpTrSv-CUDA

Tesis Project

About

This repository contains a Sparse Matrix solver using three diferent methods:
Solver Simple: Solvers any trinagular system without any analysis
Solver Order: Solves any triangular system based on the analysis from multirow_analysis_base_GPU with mode  = 1
Solver Multirow: Solves any triangular system based on the analysis from multirow_analysis_base_GPU with mode  = 0

Project Features:
	mmmmm....

Compiling and Executing Code

Compile using provided Makefile, it is requiered to have nvcc compiler and a CUDA enabled device to run it on.

Compilation Options:
    make (compiles main.cu)

Execution Options:
    sptrsv_double /path/to/matrix.mtx

Clean Options:

    make clean (cleans all executables from project)

Authors

Franco
Manuel
Juan 
