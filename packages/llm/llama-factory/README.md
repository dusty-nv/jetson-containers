

git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

jetson-containers build llama-factory

jetson-containers run $(autotag llama-factory) 


