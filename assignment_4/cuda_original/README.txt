This folder provides all the needed template for the CUDA implementation of the heatwave and histogram applications.

Note the following important things:
- this is NOT yet integrated in the full Makefile tree. Thus, one has two options: 
1/ copy this folder in your tree, but use the local makefile in which you have changed the paths according to your own folder names and hierarchies.
OR
2/ replicate one of the existing folders - e.g., OpenMP, look for the differences between makefiles (check Makefile.cuda) and import the changes in 
the local Makefile. 

Option 1 seems easier.

- there are a few new files that have been added: histogram.cu and compute_cuda.cu. They are currently NOT implementing anything related to the 
actual requirements, but do include examples of kernels based on the vector_add. These would of course need modification. 
 
