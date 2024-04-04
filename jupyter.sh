#!/bin/bash
#SBATCH --job-name="Jupyter-GPU-Demo"     # a name for your job
#SBATCH --partition=peregrine-gpu          # partition to which job should be submitted
#SBATCH --qos=gpu_debug                    # qos type
#SBATCH --nodes=1                          # node count
#SBATCH --ntasks=1                         # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                  # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                           # total memory per node
#SBATCH --gres=gpu:nvidia_a100_3g.39gb:1   # Request 1 GPU
#SBATCH --time=00:15:00                    # total run time limit (HH:MM:SS)

module purge
module load python/anaconda

# Function to check if a port is in use
is_port_in_use() {
    netstat -an | grep $1 | grep LISTEN > /dev/null
    return $?
}

# Find the next available port
port=8888
while true; do
    port=$((port+1))
    if ! is_port_in_use $port; then
        break
    fi
done

echo "Next available port is $port"

ssh -N -f -R $port:localhost:$port falcon
jupyter-notebook --no-browser  --port=$port