cpu:
	gcc -o beamform pa1.c
	.\beamform.exe 16

gpu:
	nvcc -o pa1_gpu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include" pa1_gpu.cu
	.\pa1_gpu.exe 16

check_gen:
	gcc -o solution_check solution_check.c
	.\solution_check.exe 16