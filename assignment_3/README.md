# Assignment 3

The CLI of the heat dissipation part of the assignment is the same as for assignment 1 & 2. 

The CLI for the histogram part is as follows:
 - **-s** sets the seed
 - **-p** image path
 - **-i** if set then generate a random image
 - **-n** number of rows of the image
 - **-m** number of columns of the image
 - **-g** debug (print historgram)

 You may change everything in the boilerplate code. The only thing that **HAS TO** stay the same is the current options in the CLI and the output this produces. So you may change things like the data structures used **BUT** we still want the same output when printing the histogram. You may also extend the functionality (i.e. if you want to generate images of the same color you can write a second "generate image" function). We will only test your code using the images in the "images" folder.  

The CLI for the pipesort is as follows:
 - **-l** sets the length of the vector
 - **-s** sets the seed of the random number generator
 You may extend the CLI, but not alter the behaviour of "l" and "s". Set any additional flags to the default you want us to use. 


1) Assignment 3 - Histogram Testing: TODO

    For correctness:
        "- p ../../../images/pat1_100x150.pgm -n 150 -m 100 -g"
        
    For performance:
        "- p ../../../images/pat1_1000x1000.pgm -n 5000 -m 5000"

2) Assignment 3 - Pipesort Testing:

    For correctness:
         "- l 1000 -s 42"
        
    For performance:
        "- l 50000 -s 42"
        
    To push your code we will use (comment: have to see if 1000000 is reasonable):
        "- l 1000000 -s 42"
