#!/bin/bash

#SBATCH --job-name=pmpnn
#SBATCH --partition=qgpu_3090
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH --output=%j.out
#SBATCH --error=%j.err

singularity exec --nv \
    --bind  /hpcfs/fpublic/container/singularity/app/esm-1v/esm1v/predict.py:/workspace/esm1v/predict.py \
    --bind /hpcfs/fpublic/container/singularity/app/esm-1v/esm1v/predict.sh:/workspace/esm1v/predict.sh \
    --bind ./checkpoint/esm1v_t33_650M_UR90S_1.pt:/workspace/esm1v/esm1v_t33_650M_UR90S_1.pt \
     esm1v_latest.sif \
      /bin/bash  /workspace/esm1v/predict.sh "HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW" 1 \
      /hpcfs/fhome/yangchh/tools_deployed/esm-1v/input/input.csv \
      /hpcfs/fhome/yangchh/tools_deployed/esm-1v/output/result.csv
