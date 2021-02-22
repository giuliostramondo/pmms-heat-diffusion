# Assignment 2 - Mergesort and Vecsort
- You will probably implement multiple versions of mergesort and vecort. At least one sequential and one parallel version. 
- Please make sure that each version is in its own subdirectory for the right sub-assignment (i.e. if you have a parallel implementation of mergesort called "parallel_a" it should be in: **assignment_2/mergesort/parallel_a/**).
- The executable needs to have the same name as the version folder name, i.e. for the parallel mergesort version please make sure the executable is called "parallel_a".
- Make a new subfolder for each version you want to try. All versions should be based on the same boilerplate code.  
- If one version does not result in the correct results do **NOT** include it in your submission!!!
 
Make sure there are no spaces in the paths (i.e. I **don't** want to see: assignment_2/mergesort/ final final final version that you need to test/)!

Mergesort and vecsort have almost the same CLI. 
- In mergesort you have to sort one vector of length `-l`. There are a total of 7 flags. 
- The first three `adr` determine if the numbers in the vector are ascending, descending or random.
- Next, `l` determines the length of the vector. `g` is the debug flag, i.e. if debug then print the vector before and after sorting.
- `s` sets the seed for the random number generator. 
- `p` should set the number of threads. 

- In vecsort you have to sort `-l` number of vectors with random length. 
- `l` sets the number of vectors to be sorted (i.e. outer vector length). The lengths of the **inner_vectors** has to be different for each inner vector.
- `adr` sets how the numbers of the **inner_vectors** is generated. 
- `n` sets the minimum size of each inner vector. 
- `x` sets the maximum size of each inner vector. 
The other flags are the same.  You are welcome to change the vector of vector implementation. The main point is that your code still produces the same output. 
 
You may add additional commands but do not change the current options.
We will use the following to test the correctness of **all** your versions. 

 1) Assignment 2 - Mergesort Testing: **-r -l 20000 -p 2 -s 42 -g**
 2) Assignment 2 - Vecsort Testing: **-r -l 100 -n 10 -x 1000 -s 42 -g -p 2**
So it should print the vector first unsorted and then sorted. 
And between the two is the runtime of the sorting. 

For performance we will use:
  1) Mergesort Testing: **-r -l 100000000 -p 32 -s 42**
  2) Vecsort Testing: **-r -l 1000 -n 10 -x 100000 -s 42 -p 32**

For the speed we will use the *-p* flag to set the number of threads for the testing. 
So make sure that it is used to set the number of threads!
We will compare your implementations speed. The speed will not impact your grades!

