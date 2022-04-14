cpu:
	gcc -mavx -pthread -o beamform -g aoyin1.c -lm
	./beamform 16
	# .\beamform.exe 16

gpu:
	nvcc -o pa1_gpu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include" pa1_gpu.cu
	.\pa1_gpu.exe 16

gpu_tx:
	nvcc -o pa1_gpu_tx -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include" pa1_gpu_tx.cu
	.\pa1_gpu_tx.exe 16

check_gen:
	gcc -o solution_check solution_check.c
	.\solution_check.exe 16