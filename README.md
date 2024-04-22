# Overview

This is the Python multithreaded version of `cgmlst-dists` originally developed by Torsten Seemann. The original repository can be found at [https://github.com/tseemann/cgmlst-dists/](https://github.com/tseemann/cgmlst-dists/).

`cgmlst-dists` is a tool used for calculating pairwise Hamming distances for genome profiles in a core genome multilocus sequence typing (cgMLST) schema. This Python version utilizes multithreading to enhance performance during distance calculations.

For more information on cgMLST and the original `cgmlst-dists` tool, please refer to the [cgmlst-dists GitHub repository](https://github.com/tseemann/cgmlst-dists/).


# Usage

```bash
$ python cgmlst-dists.py --help
usage: cgmlst-dists.py [-h] [--input INPUT] [--output OUTPUT] [--skip_input_replacements] [--input_sep INPUT_SEP] [--output_sep OUTPUT_SEP] [--index_name INDEX_NAME]
                       [--matrix-format {full,lower-tri,upper-tri}] [--version]

Calculate pairwise Hamming distances. Version: 0.0.1

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the input TSV file
  --output OUTPUT       Path to save the output TSV file
  --skip_input_replacements               Skip input replacements when there are no strings in the input (to save unnecessary computations)
  --input_sep INPUT_SEP
                        Input file separator (default: ' ')
  --output_sep OUTPUT_SEP
                        Output file separator (default: ' ')
  --index_name INDEX_NAME
                        Name for the index column (default: 'cgmlst-dists')
  --matrix-format {full,lower-tri,upper-tri}
                        Format for the output matrix (default: full)
  --version             show program's version number and exit
```

# Validation

`cgmlst-dists.py` vs `cgmlst-dists`

## System Specifications

`cgmlst-dists.py` was evaluated on a system running AlmaLinux version 8.8, featuring a Kernel version of 4.18.0-477.13.1.el8_8.x86_64, and powered by an Intel(R) Xeon(R) Gold 6252N CPU at 2.30GHz with 192 CPUs.

Note: If `num_threads` is not specified, the script defaults to using half of the available CPUs plus one.

## boring.tab (5x6)

Source: https://github.com/tseemann/cgmlst-dists/blob/master/test/boring.tab

```bash
$ time ./cgmlst-dists test/boring.tab > validation/boring_c.tab
This is cgmlst-dists 0.4.0
Loaded 5 samples x 6 allele calls
Calculating distances: 100.00%
Writing distance matrix to stdout...

Done.

real    0m0.009s
user    0m0.002s
sys     0m0.003s

$ time python cgmlst-dists.py --input test/boring.tab --output validation/boring_py.tab 
Loaded matrix of 5 samples and 6 allele calls.
The final matrix will have 25 distances.
Calculations completed. Saving distances...
Process completed successfully.

Total time taken: 1.52 seconds

real    0m2.864s
user    0m4.569s
sys     0m12.362s


$ md5sum validation/boring_c.tab
f523a48b2339ab7d018fe9b69c3fc326  validation/boring_c.tab
$ md5sum validation/boring_py.tab
f523a48b2339ab7d018fe9b69c3fc326  validation/boring_py.tab
```

## chewie.tab (10x10)

Source: https://github.com/tseemann/cgmlst-dists/blob/master/test/chewie.tab

```bash
$ time ./cgmlst-dists test/chewie.tab > validation/chewie_c.tab
This is cgmlst-dists 0.4.0
Loaded 10 samples x 10 allele calls
Calculating distances: 100.00%
Writing distance matrix to stdout...

Done.

real    0m0.008s
user    0m0.002s
sys     0m0.005s

$ time python cgmlst-dists.py --input test/chewie.tab --output validation/chewie_py.tab 
Loaded matrix of 10 samples and 10 allele calls.
The final matrix will have 100 distances.
Calculations completed. Saving distances...
Process completed successfully.

Total time taken: 1.52 seconds

real    0m2.888s
user    0m4.341s
sys     0m12.611s

$ md5sum validation/chewie_c.tab
de4ba5b0bb0c93fb6fb1ea90467c02ab  validation/chewie_c.tab
$ md5sum validation/chewie_py.tab
de4ba5b0bb0c93fb6fb1ea90467c02ab  validation/chewie_py.tab
```

## 100.tab (100x3016)

Source: https://github.com/tseemann/cgmlst-dists/blob/master/test/100.tab

```bash
$ time ./cgmlst-dists test/100.tab > validation/100_c.tab
This is cgmlst-dists 0.4.0
Loaded 100 samples x 3016 allele calls
Calculating distances: 100.00%
Writing distance matrix to stdout...

Done.

real    0m0.086s
user    0m0.079s
sys     0m0.005s

$ time python cgmlst-dists.py --input test/100.tab --output validation/100_py.tab 
Loaded matrix of 100 samples and 3016 allele calls.
The final matrix will have 10000 distances.
Calculations completed. Saving distances...
Process completed successfully.

Total time taken: 2.24 seconds

real    0m3.784s
user    0m10.233s
sys     0m14.935s

$ md5sum validation/100_c.tab
5a62236c697ef1eb56b7065d406007af  validation/100_c.tab
$ md5sum validation/100_py.tab
5a62236c697ef1eb56b7065d406007af  validation/100_py.tab
```

## crc32.tab (3933x1748)

This input matrix does not contain strings, so in the Python version, replacements are skipped.

```bash
$ time ./cgmlst-dists test/crc32.tab > validation/crc32_c.tab
This is cgmlst-dists 0.4.0
Loaded 3933 samples x 1748 allele calls
Calculating distances: 100.00%
Writing distance matrix to stdout...

Done.

real    0m57.766s
user    0m57.325s
sys     0m0.245s

$ time python cgmlst-dists.py --input test/crc32.tab --output validation/crc32_py.tab --skip_input_replacements
Loaded matrix of 3933 samples and 1748 allele calls.
The final matrix will have 15468489 distances.
Calculations completed. Saving distances...
Process completed successfully.

Total time taken: 7.00 seconds

real    0m8.435s
user    1m42.790s
sys     0m16.848s

$ md5sum validation/crc32_c.tab 
e3f13c85c9028d49a4867eadba20c11a  validation/crc32_c.tab
$ md5sum validation/crc32_py.tab 
e3f13c85c9028d49a4867eadba20c11a  validation/crc32_py.tab
```

## 5000.tab (5000x3016)

```bash
$ time ./cgmlst-dists test/5000.tab > validation/5000_c.tab
This is cgmlst-dists 0.4.0
Loaded 5000 samples x 3016 allele calls
Calculating distances: 100.00%
Writing distance matrix to stdout...

Done.

real    2m43.855s
user    2m42.837s
sys     0m0.348s

$ time python cgmlst-dists.py --input test/5000.tab --output validation/5000_py.tab 
Loaded matrix of 5000 samples and 3016 allele calls.
The final matrix will have 25000000 distances.
Calculations completed. Saving distances...
Process completed successfully.

Total time taken: 24.89 seconds

real    0m26.361s
user    4m3.116s
sys     0m15.963s

$ md5sum validation/5000_c.tab 
2e6d5d6c8856ef408e4f596a1841bdf6  validation/5000_c.tab
$ md5sum validation/5000_py.tab 
2e6d5d6c8856ef408e4f596a1841bdf6  validation/5000_py.tab
```

# Docker

build
```bash
docker build -t cgmlst-dists-py .
```

help
```bash
docker run --rm cgmlst-dists-py
```

launch
```bash
docker run --rm -v "$(pwd):/app/data" cgmlst-dists-py --input data/test/100.tab --output data/100_py.tab
```

# Comments

* When the initial matrix is large is convenient to use the python version.
* The C version does not implement multithreading.
* The C version suffers from memory problems when the input is large.