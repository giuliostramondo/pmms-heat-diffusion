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
The outputs were generated with the following command: **./heat_seq -n 150 -m 100 -i 42 -e 0.0001 -c ../../images/pat1_100x150.pgm -t ../../images/pat1_100x150.pgm -r 1 -k 10 -L 0 -H 100**
7) The *Latex_template* folder contains the latex template that we want you to use. The page limit is 10 pages. If you have too much information you can put it into an appendix, which we might or might not read.

## Submission
**How to submit your assignment on canvas**
1) Enter the correct information in the main Makefile (group id, student ids).
2) Run "make submission_%" where % is the assignment number. This will generate a tar thats named in the following way "heat_assignment_%_group_HERE_GROUP_ID_HERE_ID_OF_THE_STUDENT_HERE_ID_OF_THE_STUDENT.tar.gz"
3) Upload the tar to canvas. 
4) Name the PDF in the same way (i.e. heat_assignment_%_group_HERE_GROUP_ID_HERE_ID_OF_THE_STUDENT_HERE_ID_OF_THE_STUDENT.pdf). And also upload it on canvas. 
5) Only one person in the group needs to upload the assignment! 
6) Do **not** include the pdf in the tar! 
 
## Resources and Tools:

A highly recommended tool for profiling your code is Intel Vtune. 
Intel: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/vtune-profiler.html

Some information on vectorization: https://software.intel.com/content/www/us/en/develop/articles/recognizing-and-measuring-vectorization-performance.html

Additionally, the Godbolt Compiler Explorer is very interesting and useful. https://godbolt.org/

## Heat Dissipation assignments: 
Have a look at the project description on Canvas (section 3).

### Command Line Option
 You are welcome to add your own options but do not change the options that already exist. We use these for testing your code. 
Superficially for all heat dissipation parts we use: **./heat_seq -n 150 -m 100 -i 42 -e 0.0001 -c ../../images/pat1_100x150.pgm -t ../../images/pat1_100x150.pgm -r 1 -k 10 -L 0 -H 100** to check for correctness.
To measure the GFLOPs we will use additional tests for different image sizes etc. 

There are two flags (-k and -r) that seem similar but are a bit confusing. 
The -k sets how often a report is filled (i.e. how often it calculates max, min, average values etc).
The -r flag purely sets IF that result should be printed or not, so there could be a case that you compute the values for the report but dont print it.
In most cases if you compute the values for the report you also want to print the report. 

For assignments 2 and 3 make sure that you use the -p flag to set the number of threads. Otherwise we will test your code with a single thread. 

## Other Assignments
Mergesort(part of assignment 2) and vecsort(part of assignment 2) all have command line options.

1) Assignment 2 - Mergesort Testing: TBD
2) Assignment 2 - Vecsort Testing: TBD
3) Assignment 3 - Histogram Testing: TBD
4) Assignment 3 - Pipesort Testing: TBD
5) Assignment 4 - Convolution Testing: TBD
6) Assignment 4 - Histogram Testing: TBD



## Measure time 
You'll have to measure time. If not predefined in the file please use clock_gettime(CLOCK_MONOTONIC ...) to measure the time. For explanation see here: 
- https://www.cs.rutgers.edu/~pxk/416/notes/c-tutorials/gettime.html
- https://linux.die.net/man/3/clock_gettime
- https://blog.habets.se/2010/09/gettimeofday-should-never-be-used-to-measure-time.html

# DAS-5 Usage
- In the first LC (02-03-2020) you will be assigned account names for DAS-5, a computing cluster that you will be expected to run some experiments on. 
- DAS5 uses a scheduler called SLURM, the basic procedure to use it is first reserving a node (or a number of them) and then running your script on that node.
- You are recommended to use the VU node of DAS-5 (fs0.das5.cs.vu.nl) 
- You are not allowed to use the account name given to you for this course for any other work such as training NN etc...
- For information about DAS-5 see https://www.cs.vu.nl/das5/

## DAS-5 Usage Policy
- Once you log in to DAS-5 you will be automatically in a shell environment on the headnode. 
- While on the headnode you are only supposed to edit files and compile your code and not execute code on the head node. 
- You need to use prun to run your code on a DAS-5 node.
- You should not execute on a DAS-5 node for more than 15 minutes at a time.

## Connecting to DAS-5
Use ssh to connect to the cluster:
```
ssh -Y username@fs0.das5.cs.vu.nl
```
Enter your password, If its correct then you should be logged in.  

If you are a MacOS or Linux user, ssh is already available to you in the terminal.

If you are a Windows user, you need to use a ssh client for Windows. The easiest option is to use putty: http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html

## Commands 
#### Check which modules are loaded at the moment

``` 
module list 
```
### Check which modules are available

```
module avail
```
### Load prun

```
module load prun
```

### Load cuda module
```
module load cuda80/toolkit
```
### Show nodes informations ( available hardware etc )

```
sinfo -o "%40N %40f"
```
### Show active and pending reservations
```
preserve -llist
```
#### Reserve a node with the ’cpunode’ flag for 15 minutes:
- **BE AWARE: 15 minutes is the maximum time slot you should reserve!**
- **In the output of your command you will see your reservation number**
```
preserve -np 1 -native '-C cpunode' -t 15:00
```

### Run code on your reserved node
```
prun -np 1 -reserve <reservation_id> cat /proc/cpuinfo
```
### Get a randomly assigned node and run code on it
```
prun -np 1 cat /proc/cpuinfo
```
### Schedule a reservation for a node with a GTX1080Ti GPU on christmas day ( 25th of December )starting at midnight and 15 minutes for 15 minutes
```
preserve -np 1 -native ’-C gpunode,GTX1080Ti’ -s 12-25-00:15 -t 15:00
```
### Cancel Reservation
```
preserve -c <reservation_id>
```
