Solver.py containts the code to attempt to solve the Puzzle. the program will prompt you to enter a file name of an image, the
three images in the file are made to work with the program. once all the data is loaded it will display and save an image off all the combined
Pieces.  it will then ask if the image is good, "no" will tell the program to make another image, anything else will end the program.  

EgdeAI.py will run on its own using the two txt files with the training data contained inside.  this uses a two layer nural network to try and determine if a given peice is
and edge peice or not. best case  runs got to around 65% accuraccy.  the training data consists of diffrent rotational data of
the peices from the 54 peice puzzle used for testing.   