two parallel strategies: 

- dividing parallelism across scanline, each thread read all transducer data and output one pixel, high shared mem reading traffic.
- dividing parallelism across transducer channels, each transducer contributes to the whole image. Low shared mem traffic while need mutex access to output image[] array

parallel strategy differes based on arch

- for GPU, dividing parallelism across scanline is much better than dividing parallelism across transducer channels because of large local memory. On the other hand, global memory synchronization is extremely expensive and inefficient. So shared read only mem is ok, shared writable mem should be avoided.

- for FPGA, dividing parallelism across transducer is much better than dividing parallelism across scanline because of low sync & wire overhead and high memory overhead. For on-chip design, wiring is cheap while buffer is expensive.

