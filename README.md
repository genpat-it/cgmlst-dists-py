# Overview

This is the Python multithreaded version of `cgmlst-dists` originally developed by Torsten Seemann. The original repository can be found at [https://github.com/tseemann/cgmlst-dists/](https://github.com/tseemann/cgmlst-dists/).

`cgmlst-dists` is a tool used for calculating pairwise Hamming distances for genome profiles in a core genome multilocus sequence typing (cgMLST) schema. This Python version utilizes multithreading to enhance performance during distance calculations.

For more information on cgMLST and the original `cgmlst-dists` tool, please refer to the [cgmlst-dists GitHub repository](https://github.com/tseemann/cgmlst-dists/).

# Usage

```console
$ python cgmlst-dists.py --help
usage: cgmlst-dists.py [-h] [--input INPUT] [--output OUTPUT] [--skip_input_replacements] 
                       [--input_sep INPUT_SEP] [--output_sep OUTPUT_SEP] [--index_name INDEX_NAME]
                       [--matrix-format {full,lower-tri,upper-tri}] [--num_threads NUM_THREADS] 
                       [--chunk_size CHUNK_SIZE] [--missing_char MISSING_CHAR]
                       [--locus-completeness LOCUS_COMPLETENESS]
                       [--sample-completeness SAMPLE_COMPLETENESS] [--version]

Calculate pairwise Hamming distances. Version: 0.0.3

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the input TSV file
  --output OUTPUT       Path to save the output TSV file
  --skip_input_replacements
                        Skip input replacements when there are no strings in the input (to save unnecessary computations)
  --input_sep INPUT_SEP
                        Input file separator (default: '\t')
  --output_sep OUTPUT_SEP
                        Output file separator (default: '\t')
  --index_name INDEX_NAME
                        Name for the index column (default: 'cgmlst-dists')
  --matrix-format {full,lower-tri,upper-tri}
                        Format for the output matrix (default: full)
  --num_threads NUM_THREADS
                        Number of threads for parallel execution (default: half of available CPUs + 1)
  --chunk_size CHUNK_SIZE
                        Size of chunks to save the output file (default: 1000)
  --missing_char MISSING_CHAR
                        Character used for missing data (default: '-')
  --locus-completeness LOCUS_COMPLETENESS
                        Minimum percentage of non-missing data required for a locus (0-100)
  --sample-completeness SAMPLE_COMPLETENESS
                        Minimum percentage of non-missing data required for a sample (0-100)
  --version            show program's version number and exit
```

## Data Filtering Example

The tool now supports filtering both loci and samples based on data completeness:

```console
$ python cgmlst-dists.py --input input.tsv --output output.tsv --locus-completeness 85 --sample-completeness 85

Loading data from input.tsv...
Initial data shape: 100 samples × 50 loci

Applying locus completeness filter (threshold: 85%)...
Loci filtering details:
--------------------------------------------------------------------------------
Locus                          Completeness %  Status     Missing/Total
--------------------------------------------------------------------------------
locus1                              98.00%    INCLUDED   2/100
locus2                              82.00%    EXCLUDED   18/100
...

Applying sample completeness filter (threshold: 85%)...
Sample filtering details:
--------------------------------------------------------------------------------
Sample ID                      Completeness %  Status     Missing/Total
--------------------------------------------------------------------------------
sample1                             95.00%    INCLUDED   2/50
sample2                             78.00%    EXCLUDED   11/50
...

Final data shape after filtering: 80 samples × 45 loci
Calculating distances...
```

# Validation

[Rest of the validation section remains unchanged...]

# Docker

build
```console
docker build -t cgmlst-dists-py .
```

help
```console
docker run --rm cgmlst-dists-py
```

launch
```console
docker run --rm -v "$(pwd):/app/data" cgmlst-dists-py --input data/test/100.tab --output data/100_py.tab
```

# Comments

* When the initial matrix is large is convenient to use the python version
* The C version does not implement multithreading
* The C version suffers from memory problems when the input is large
* The Python version supports data quality filtering through completeness thresholds for both loci and samples
* Detailed reporting helps identify problematic loci and samples before distance calculation