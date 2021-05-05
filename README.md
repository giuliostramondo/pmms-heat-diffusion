# UvA PMMS Course - Assignments 
This repo contains the boilerplate code for the course "Programming multicore manycore systems".
The course consists of 4 assignments. All assignments contain a part where students work on a heat dissipation simulation.
Additionally, assignments 2 to 4 also contain additional parts. 

## Structure:
The structure of this repo is as follows:
1) One folder per assignment, with subfolders for each sub-assignment. 
2) There is a bit of boilerplate code for each assignment and a make file. 
3) The *include* folder contains boilerplate headers for the heat dissipation simulation.
4) The *src* folder contains boilerplate code for the heat dissipation simulation.
5) The *images* folder contains input images for the heat dissipation simulation.
    - In the images folder you can also find a makefile that can generate new images e.g. "make areas_500x500.pgm"
6) You can find reference output for Heat Dissipation in */heat_dissipation_reference_output/*. 
The outputs were generated with the following command: `./heat_seq -n 150 -m 100 -i 42 -e 0.0001 -c ../../images/pat1_100x150.pgm -t ../../images/pat1_100x150.pgm -r 1 -k 10 -L 0 -H 100`
7) The **Latex_template** folder contains the latex template that we want you to use. The page limit is 10 pages. If you have too much information you can put it into an appendix, which we might or might not read.

## Submission
**How to submit your assignment on canvas**
1) Enter the correct information in the main Makefile (group id, student ids).
2) Run "make submission_%" where % is the assignment number. This will generate a tar thats named in the following way "heat_assignment_%_group_HERE_GROUP_ID_HERE_ID_OF_THE_STUDENT_HERE_ID_OF_THE_STUDENT.tar.gz"
3) Upload the tar to canvas. 
4) Name the PDF in the same way (i.e. heat_assignment_%_group_HERE_GROUP_ID_HERE_ID_OF_THE_STUDENT_HERE_ID_OF_THE_STUDENT.pdf). And also upload it on canvas. 
5) Only one person in the group needs to upload the assignment! 
6) Do **not** include the pdf in the tar! 
 
## Resources and Tools:

- A highly recommended tool for profiling your code is [Intel Vtune](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/vtune-profiler.html)

- Some information on [vectorization](https://software.intel.com/content/www/us/en/develop/articles/recognizing-and-measuring-vectorization-performance.html)

- Additionally, the [Godbolt Compiler Explorer](https://godbolt.org/) is very interesting and useful. 

## Heat Dissipation assignments: 
Have a look at the project description on Canvas (section 3).

### Command Line Option
 You are welcome to add your own options but do not change the options that already exist. We use these for testing your code. 
Superficially for all heat dissipation parts we use: `./heat_seq -n 150 -m 100 -i 42 -e 0.0001 -c ../../images/pat1_100x150.pgm -t ../../images/pat1_100x150.pgm -r 1 -k 10 -L 0 -H 100` to check for correctness.
To measure the GFLOPs we will use additional tests for different image sizes etc. 

- There are two flags (`-k` and `-r`) that seem similar but are a bit confusing. 
- `-k` sets how often a report is filled (i.e. how often it calculates max, min, average values etc).
- `-r` flag purely sets IF that result should be printed or not, so there could be a case that you compute the values for the report but dont print it.
- In most cases if you compute the values for the report you also want to print the report. 

For assignments 2 and 3 make sure that you use the `-p` flag to set the number of threads. Otherwise we will test your code with a single thread. 

### Assignment 2:
Speed measurements: 
`./heat_seq -n 1000 -m 1000 -i 10000 -e 0.0001 -c ../../images/pat1_1000x1000.pgm -t ../../images/pat1_1000x1000.pgm -k 70 -L 0 -H 100 -p 16`
Additionally we will set the following environmental variables:
- OMP_PROC_BIND=true
- OMP_WAIT_POLICY=active
- OMP_NUM_THREADS=XXX
- OMP_PLACES=cores

### Assignment 3:
Speed measurements: 
`./heat_seq -n 1000 -m 1000 -i 10000 -e 0.0001 -c ../../images/pat1_1000x1000.pgm -t ../../images/pat1_1000x1000.pgm -k 70 -L 0 -H 100 -p 16`

### Assignment 4:
Speed measurements: 
`./heat_seq -n 1000 -m 1000 -i 10000 -e 0.0001 -c ../../images/pat1_1000x1000.pgm -t ../../images/pat1_1000x1000.pgm -k 70 -L 0 -H 100 -p 16`


### Measuring time 
You'll have to measure time. If not predefined in the file please use clock_gettime(CLOCK_MONOTONIC ...) to measure the time. For explanation see here: 
- https://www.cs.rutgers.edu/~pxk/416/notes/c-tutorials/gettime.html
- https://linux.die.net/man/3/clock_gettime
- https://blog.habets.se/2010/09/gettimeofday-should-never-be-used-to-measure-time.html

# Assignment 2 - [Mergesort and Vecsort](https://github.com/juliusroeder/pmms-heat-diffusion/blob/master/assignment_2/README.md)


# Assignment 3 - [Histogram and Pipesort](https://github.com/juliusroeder/pmms-heat-diffusion/blob/master/assignment_3/README.md)


# Assignment 4 - [Convolution and Histogram](https://github.com/juliusroeder/pmms-heat-diffusion/blob/master/assignment_4/README.md)

# DAS 5 Information - [Cheat-sheet](https://github.com/juliusroeder/pmms-heat-diffusion/blob/master/DAS5_cheatsheet.md)



