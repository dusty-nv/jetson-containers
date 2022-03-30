#!/usr/bin/env bash

echo "testing VPI samples..."
cd /opt/nvidia/vpi2/samples/

echo "testing C++ sample:  01-convolve_2d"
cd 01-convolve_2d/
mkdir build
cd build
cmake ../
make
./vpi_sample_01_convolve_2d cpu /test/data/test_0.jpg
./vpi_sample_01_convolve_2d cuda /test/data/test_0.jpg
./vpi_sample_01_convolve_2d pva /test/data/test_0.jpg
cd ../

echo "testing Python sample:  01-convolve_2d"
python3 main.py cpu /test/data/test_0.jpg
python3 main.py cuda /test/data/test_0.jpg
python3 main.py pva /test/data/test_0.jpg

echo "testing C++ sample:  03-harris_corners"
cd ../03-harris_corners/
mkdir build
cd build
cmake ../
make
ls
./vpi_sample_03_harris_corners cpu /test/data/test_0.jpg
./vpi_sample_03_harris_corners cuda /test/data/test_0.jpg
./vpi_sample_03_harris_corners pva /test/data/test_0.jpg
cd ../

echo "testing Python sample:  03-harris_corners"
python3 main.py cpu /test/data/test_0.jpg
python3 main.py cuda /test/data/test_0.jpg
python3 main.py pva /test/data/test_0.jpg

echo "testing C++ sample:  04-rescale"
cd ../04-rescale/
mkdir build
cd build
cmake ../
make
ls
./vpi_sample_04_rescale cpu /test/data/test_0.jpg
./vpi_sample_04_rescale cuda /test/data/test_0.jpg
./vpi_sample_04_rescale vic /test/data/test_0.jpg
cd ../

echo "testing Python sample:  04-rescale"
python3 main.py cpu /test/data/test_0.jpg
python3 main.py cuda /test/data/test_0.jpg
python3 main.py vic /test/data/test_0.jpg

echo "VPI samples OK"