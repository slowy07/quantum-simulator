# usage of sample applications

clfsim and clfsimh are design to be extensible to variety of different application. The base version of each are ``clfsim_base`` and ``clfsimh_base``; sample extenions are provided in ``apps``. To compile the code, just run ``make clfsim``. Binaries of the form ``clfsim(h)_*.x`` will be added to the ``apps`` directory.

## clfsim_base usage

```
./clfsim_base.x -c circuit_file -d maxtime -t num_thread -f max_fuesed_size -v verbosity -z
```

| Flag | Description    |
|--------------- | --------------- |
| ``-c circuit_file``   | circuit file to run   |
| ``-d maxtime`` | maximum time |
| ``-t num_threads`` | number of threads to use |
| ``-f max_fuesed_size | maximum fused gate size |
| ``-v verbosity`` | verbosity level (0...5) |
| ``-z`` | set ``flush-to-zero`` and ``denormals-are-zeros`` MXCSR control flags |

Example:

```
./clfsim_base.x -c ../circuits/circuit_q24 -d 16 -t 8 -v 1
```


