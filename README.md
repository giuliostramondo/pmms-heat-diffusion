# pmms-heat-diffusion
Programming multicore manycore systems, heat diffusion lab code.

# Structure:
1) One folder per assignment, with subfolders for each sub-assignment. 
2) There is a bit of boilerplate code for each assignment and a make file. 

# Heat Dissipation assignments: 
1) Have a look at the project description on Canvas (section 3). 
2) You can find input images for the heat dissipation assignments in the /images/ folder. 
    - In the images folder you can also find a makefile that can generate new images e.g. "make areas_500x500.pgm"

# Command Line Options
The assignments: heat dissipation (assignment1-4), mergesort(part of assignment 2) and vecsort(part of assignment 2) all have command line options. You are welcome to add your own options but do not change the options that already exist. We use these for testing your code. 

# Measure time. 
You'll have to measure time. If not predefined in the file please use clock_gettime(CLOCK_MONOTONIC ...) to measure the time. For explanation see here: https://blog.habets.se/2010/09/gettimeofday-should-never-be-used-to-measure-time.html
