gpu:
	nvcc -o yuanqing_gpu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include" yuanqing_gpu.cu
	.\yuanqing_gpu.exe 16

check_gen:
	gcc -o solution_check solution_check.c -lm

check:
	.\solution_check.exe 16

perf:
	perf record --call-graph=dwarf ./beamform 16
	perf report --hierarchy -M intel

win:
	gcc -mavx2 -o aoyin_win -g aoyin2.c libpthreadGC2.a -lm

base:
	gcc -o base base.c
	.\base.exe 16