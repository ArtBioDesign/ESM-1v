# ESM 1v
## Project Introduction  
**ESM-1v** is a mutation prediction tool developed by Facebook AI Research in 2021, based on the ESM protein language model.

## Installation & Execution for HPC Deployment

### **1. Build the Environment**
```bash
conda env create -f environment.yml
conda activate esm1v
conda install -c conda-forge conda-pack
conda-pack -n esm1v -o esm1v.tar.gz
```

### **2. Build image from docker file**
```shell
docker build -f dockerfile -t esm-1v:latest .
```

### **3. Save the Docker Image**
```shell
docker save -o esm-1v.tar esm-1v:latest
```

### **4. Convert Docker Image to Singularity**
```shell
singularity build esm1v_latest.sif docker-archive://esm-1v.tar
```

### **5. Submit the Job to SLURM**
```shell
sbatch runhpc.sh
```
