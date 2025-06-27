# Lattice Boltzmann Method- PLASMA
The purpose of this library is to simulate, using a LB method, a three-populations plasma, including a simple thermal coupling through a DDF approach. <br />
The physics behind and the description of the methods used are discussed in details in the report.

## Compiling
In order not to have problems in the compilation we need to be sure that all the packets needed are correctly installed: <br /> 
1. sudo apt update <br />
2. sudo apt install pkg-config <br />
3. sudo apt-get install libopencv-dev <br />
4. sudo apt-get install libc6-dev
5. sudo apt-get install gcc-10 g++-10 <br />
6. sudo apt install ffmpeg <br />
7. sudo apt-get install libfftw3-dev <br />

At this point we need to locate the file "opencv4.pc": <br />
dpkg -L libopencv-dev <br /><br />
Now we need to re-configure the path through this command: <br />
export PKG_CONFIG_PATH=<insert/your/path/>/pkgconfig:$PKG_CONFIG_PATH <br /> 
You should type something like: <br />
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH <br /> <br />

To check add the line: <br />
pkg-config --modversion opencv4 <br />
If we get something like "4.6.0", then we're okay.

## Runnning
In order to run the code we need these lines: <br /> <br />
chmod +x compile_and_run.sh <br />
./compile_and_run.sh <number_of_cores> <br />
<br />
and sobstitute <number_of_cores> with the number of cores with which you want to run the program. If you don't insert it it will simply run with the maximum number of threads in your pc.
The first one is just to move in the folder of the code while the second compile the file with all the information needed to compute the program and the third one run that file.  

## Things left to do in increasing difficulties and maybe time
1. Refine the files bottleneck, valgrind, valgrind_complete so that they gives the correct informations in build
2. refine compile_and_run with some adjustments
3. rewrite the ReadMe with some instructions on running and some example copyable
4. move the ComputeEquilibrium() from plasma.cpp to collisions.cpp since it is used only there and so don't define f_eq_species and g_eq_species
5. make it Cache-friendly by indexing arrays in SoA way so from f_[s][idx] move to f[s*idx] so that the Cache is efficiently used
6. After all of that try to generalize to N_species given from the user (also the NX and NY should be given from the user)
