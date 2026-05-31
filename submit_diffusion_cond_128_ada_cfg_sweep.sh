#!/bin/bash
#SBATCH --job-name=cluster-ml-diff-cond-128-ada-cfg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/diff_cond_128_ada_cfg_%j.out
#SBATCH --error=logs/diff_cond_128_ada_cfg_%j.err

mkdir -p logs

cd /oscar/data/idellant/cluster-ml
source venv/bin/activate

# CFG sweep on the trained AdaGN 128px EMA. Per-sample conditioning was
# already working at CFG=5 (NN-train-TSC tracked gen TSC across the
# range), but fidelity stats showed AdaGN+CFG=5 may be over-amplifying:
# normalized low-k 3.30x and gen brightness pushed too high. With AdaGN
# already providing strong conditional modulation, the "right" CFG is
# probably lower (1.5-3 is standard for AdaGN models).

for CFG in 1.5 2.0 3.0; do
    TAG=cfg${CFG%.*}                                # cfg1, cfg2, cfg3 (slightly off; matches sweep style)
    if [[ "$CFG" == "1.5" ]]; then TAG=cfg15; fi
    echo "=== sample-only at CFG=${CFG} (tag=${TAG}) ==="

    python train_diffusion.py \
        --data diffusion_radio_128_v2.h5 \
        --out-dir diffusion_out_cond_128_ada \
        --condition --ada \
        --labels TSC_Cutimages/TSC_eachhalo_snap99.hdf5 \
        --label-key tsc_gyr --cond-scale-norm 8.0 \
        --sample-only --out-tag $TAG \
        --batch-size 32 \
        --cfg-scale $CFG \
        --sample-tsc 0.3 0.5 0.7 0.9 1.1 1.3 1.5 1.7 2.0 \
        --n-per-tsc 60

    python plot_cond_nn_check.py \
        --samples diffusion_out_cond_128_ada/samples_cond_${TAG}.npz \
        --data diffusion_radio_128_v2.h5 \
        --out-nn diffusion_out_cond_128_ada/nn_check_cond_${TAG}.png \
        --out-leak diffusion_out_cond_128_ada/cond_leakage_${TAG}.png
done
