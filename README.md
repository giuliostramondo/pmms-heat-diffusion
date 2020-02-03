# pmms-heat-diffusion
Programming multicore manycore systems, heat diffusion lab code.

## Structure:
1) One folder per assignment, with subfolders for each sub-assignment. 
2) There is a bit of boilerplate code for each assignment and a make file. 

## Heat Dissipation assignments: 
1) Have a look at the project description on Canvas (section 3). 
2) You can find input images for the heat dissipation assignments in the /images/ folder. 
    - In the images folder you can also find a makefile that can generate new images e.g. "make areas_500x500.pgm"

## Command Line Options
The assignments: heat dissipation (assignment1-4), mergesort(part of assignment 2) and vecsort(part of assignment 2) all have command line options. You are welcome to add your own options but do not change the options that already exist. We use these for testing your code. 

## Measure time 
You'll have to measure time. If not predefined in the file please use clock_gettime(CLOCK_MONOTONIC ...) to measure the time. For explanation see here: https://blog.habets.se/2010/09/gettimeofday-should-never-be-used-to-measure-time.html

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
preserve -np 1 -native ’-C cpunode’ -t 15:00
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
