# ESM 1v
## Project Introduction  
**ESM-1v** is a mutation prediction tool developed by Facebook AI Research in 2021, based on the ESM protein language model.

## Installation & Execution for Local Deployment
Build image from docker file
```shell
docker build -f dockerfile -t esm-1v:latest .
```

Save the Docker Image
```shell
docker save -o esm-1v.tar esm-1v:latest
```

Convert Docker Image to Singularity
```shell
singularity build esm1v_latest.sif docker-archive://esm-1v.tar
```

Submit the Job to SLURM
```shell
sbatch runhpc.sh
```
