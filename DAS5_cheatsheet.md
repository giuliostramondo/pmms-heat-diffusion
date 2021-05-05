# DAS-5 Cheat Sheet
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
PRESERVE -NP 1 -NATIVE '-C CPUNODE' -T 15:00
```

### RUN CODE ON YOUR RESERVED NODE
```
PRUN -NP 1 -RESERVE <RESERVATION_ID> CAT /PROC/CPUINFO
```
### GET A RANDOMLY ASSIGNED NODE AND RUN CODE ON IT
```
PRUN -NP 1 CAT /PROC/CPUINFO
```
### SCHEDULE A RESERVATION FOR A NODE WITH A GTX1080TI GPU ON CHRISTMAS DAY ( 25TH OF DECEMBER )STARTING AT MIDNIGHT AND 15 MINUTES FOR 15 MINUTES
```
PRESERVE -NP 1 -NATIVE ’-C GPUNODE,GTX1080TI’ -S 12-25-00:15 -T 15:00
```
### CANCEL RESERVATION
```
PRESERVE -C <RESERVATION_ID>
```
