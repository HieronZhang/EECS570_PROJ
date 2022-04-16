# GPU Acceleration

We accelerate the algorithm on GPGPU with the help of NVidia CUDA toolkit. When a GPU program is executed, CPU works as a host to manage the device GPU. Host copies data from disk to its own memory, allocate corresponding memory space on GPU, copy data from CPU memory into GPU memory and triggers the kernel function. After GPU finishes calculation, host CPU copies data from GPU to its own memory system.  

Two methods of exploiting parallelism are explored and evaluated separately.  For this beamforming algorithm, we have $32\times32$ transducer matrix, each contributes to a 3D image space with $1560\times size^2$ pixels. To exploit parallelism, one method is to split work along transducer channels and the other method is to split work along scanlines. Evaluation result shows significant performance difference between two methods.

For the algorithm which paralyzes workload with transducer, there exists mutual excusive read-modify-write pattern, raising up a need for the synchronization between threads on GPU. Unlike CPU, GPU has much looser programming model because of its weak unicore performance and large amount of unicores. Therefore, coherence and consistency are more difficult and expensive to be implemented for GPU than CPU. 

## Parallelization Strategy

Firstly, we split work along scanlines. In this algorithm, $1560\times size^2$ threads are created in the GPU, each of which is responsible for calculating one final output of the image space. Instead of assigning the pixel id by host CPU, we have each thread calculating their own pixel id by their place in the GPU architecture: $pixel\_id = block\_size \times block\_id + thread\_id$, where $thread\_id$ is the thread number inside that specific block. To calculate the final result for one pixel, each thread need to access the whole transducer data, leading to a high global memory reading traffic. Since local node is impossible for storing a whole transducer data array, frequent load miss and stall is expected. However, since there are a large amount of threads, regular single thread stall can be efficiently hidden by the warp scheduling mechanism of GPU.

There exists another method to exploit parallelism, which is to split work along transducer channels. Each transducer calculates a whole image space array based on its own data, then 1024 images are added together to generate the final image output. Similar to the first idea, each thread also generates its transducer id based on its location in the GPU architecture so that the host does not need to assign an identical number to each thread. In this implementation, each thread load exclusive transducer data, so that the memory traffic is reduced to near optimal case. However, the process to sum up images of each thread becomes tricky because it needs synchronization between GPU, which may greatly reduce the parallelism and hurt performance. Next subsection have a more detailed discussion of GPU synchronization.

## Synchronization

Local memory of GPU is composed of two parts: shared memory and global memory. Shared memory can be accessed by all threads in the same block and maintains coherence at block level, so that all threads in the same block can realize message passing and synchronization through shared memory. However, it does not guarantee coherence and consistency between blocks.

Here shows a common implementation of shared memory in CUDA library. Each thread works on their own local data before reaching the synchronization barrier. After reaching the barrier, every thread will stall and wait for the thread 0 in this block to gather the local values and push them to the global memory. 

```
extern __shared__ int shared[]; 
const int tid = threadIdx.x; 
const int bid = blockIdx.x;
// local computation
shared[tid] = compute();
// barrier
__syncthreads();
// gather by thread 0
if (tid == 0) {
	gather_and_push();
}
```

Global synchronization, on the other hand, synchronize through all CUDA threads of the current program. Since global synchronization can efficiently serialize the program, it should be avoided as much as possible. Here is the global mutex  implemented with CUDA atomic functions and used in our accelerating algorithm. In this sample code, `mutex` plays the role of a coarse-grained synchronization point, which can greatly serialize the program to a single thread level. One method to improve parallelism is to use more fine-grained locks, such as creating one lock for each single pixel.

```
bool locked = false;
do
{
    if (locked = (atomicCAS(mutex, 0, 1) == 0))
    {
        image[point] += rx_data[index + offset];
    }
	atomicExch(mutex, 0);
} while (!locked);
```

To achieve high performance on GPU programming, it is preferred to modify the algorithm to avoid synchronization as much as possible to guarantee parallelism. If synchronization is necessary, it is a good approach to first make a copy of the global data into each blocks, maintain synchronization in the same block for this local copy, and push the modified data into global memory later with global atomic so that the frequency of global synchronization can be decreased.

## Evaluation

| size | CPU     | GPU(sc) | GPU(tx) |
| ---- | ------- | ------- | ------- |
| 16   | 19944m  | 174m    | 4160m   |
| 32   | 79302m  | 267m    | 15792m  |
| 64   | 316763m | 685m    | 60363m  |

We implemented two GPU accelerated algorithms based on two parallelization strategies mentioned above and compare them with the baseline single thread program running on CPU.  It shows that the code split across scanline obviously outperforms than the code split across transducer channels. This can be explained by that 1. scanline version exploits hundreds of times more threads, which is beneficial for GPU to exploit its high memory bandwidth and hide latency of single thread; 2. scanline version does not require thread synchronization, further enhances the parallelism of program execution. 

It is interesting to find that optimal parallelization strategy differs between GPU and other platforms such as FPGA. For GPU, dividing parallelism across scanline is much better than dividing parallelism across transducer channels because of large local memory. On the other hand, global memory synchronization is expensive and inefficient. So shared read only mem is acceptable, while shared writable mem should be avoided. For FPGA, however, dividing parallelism across transducer is much better than dividing parallelism across scanline because of low sync & wire overhead and high on-chip memory buffer overhead. For on-chip design, wiring is cheap while buffer is expensive.

## TODO

- cpu simd gpu run at param: 
  - cpu multi
  - simd
  - co
  - #thread | #hard core
  - gpu: cuda core
- algorithm change
  - gpu
- vtune:
- param: time; hardware( cpu core; ); precision 

