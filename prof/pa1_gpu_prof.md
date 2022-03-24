Profiled result of CPU version, measured by `Measure-Command {.\beamform.exe 16}`

```
Seconds           : 19
Milliseconds      : 36
Ticks             : 190366170
TotalDays         : 0.000220331215277778
TotalHours        : 0.00528794916666667
TotalMinutes      : 0.31727695
TotalSeconds      : 19.036617
TotalMilliseconds : 19036.617
```

Profiled result of GPU version, total time measured with `Measure-Command`, kernel time measured with Nsight Profiler:

| input size | transmit_distance | reflected_distance | Total |
| ---------- | ----------------- | ------------------ | ----- |
| 16         | 19.49u/24009      | 34.38m/51599025    | 174m  |
| 32         | 65.60u/89990      | 137.40m/206104352  | 267m  |
| 64         | 251.68u/388959    | 548.52m/822579280  | 685m  |
|            |                   |                    |       |