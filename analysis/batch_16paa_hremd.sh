#!/usr/bin/env bash
# Created by Alec Glisman (GitHub: @alec-glisman) on July 22nd, 2022

# Node configuration
#SBATCH --partition=all --qos=d --account=d
#SBATCH ---ntasks=16 --nodelist=zeal
#SBATCH --mem=40G
#SBATCH --gres=gpu:0 --gpu-bind=closest

# Job information
#SBATCH --job-name=Anl16-MDA7
#SBATCH --time=2-0:00:00

# Runtime I/O
#SBATCH -o logs/jid_%j-node_%N-%x.log -e logs/jid_%j-node_%N-%x.log

# built-in shell options
set -o errexit # exit when a command fails. Add || true to commands allowed to fail
set -o nounset # exit when script tries to use undeclared variables

# catch keyboard interrupt (control-c)4.5
function control_c() {
    echo " - Keyboard interrupt (control-c) detected. Exiting..."

    # if PID exists, kill it
    if [[ -n "${pid_dask_client}" ]]; then
        echo " - Killing dask client..."
        kill -9 "${pid_dask_client}"
    else
        echo " - No dask client to kill."
    fi

    echo " - Exiting..."
}
trap control_c SIGINT SIGTERM

# analysis method
python_script='mda_data_gen.py'
single_analysis='0'
sim_idx='0'
pid_dask_client=''

dir_sims_base="${HOME}/Data/1-electronic-continuum-correction/5-ECC-two-chain-PMF/2_production_completed/polyacrylate-homopolymer"
dir_append='replica_00/2-trajectory-concatenation'
dir_sims=(
    "sjobid_10071-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-0Ca-12.0nmbox-ion_pe_all_scaledEStatics-node01-2023-01-27-08:32:21.543949375/3-hremd-prod-16_replicas-100_steps-300_Kmin-440_Kmax"
    "sjobid_5-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-8Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-01-26-13:49:35.421512543/3-hremd-prod-16_replicas-100_steps-300_Kmin-440_Kmax"
    "sjobid_5-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-16Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-01-26-13:49:37.438178317/3-hremd-prod-16_replicas-100_steps-300_Kmin-440_Kmax"
    "sjobid_6-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-32Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-01-26-13:51:09.363289554/3-hremd-prod-16_replicas-100_steps-300_Kmin-440_Kmax"
    "sjobid_6-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-64Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-01-26-13:51:11.383105790/3-hremd-prod-24_replicas-100_steps-300_Kmin-440_Kmax"
    "sjobid_23-PAcr-2chain-16mer-atactic-Hend-chain-em-0Na-128Ca-12.0nmbox-ion_pe_all_scaledEStatics-desktop-2023-02-10-16:44:16.675883557/3-hremd-prod-24_replicas-100_steps-300_Kmin-440_Kmax"
)
tags=(
    "2PAcr-16mer-0Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_10071-idx_00"
    "2PAcr-16mer-8Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_5-idx_01"
    "2PAcr-16mer-16Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_5-idx_02"
    "2PAcr-16mer-32Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_6-idx_03"
    "2PAcr-16mer-64Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_6-idx_04"
    "2PAcr-16mer-128Ca-0Na-hremd_wtmetad_prod-12.0nm_box-jid_23-idx_05"
)
name_sims=(
    "nvt_hremd_prod_scaled"
    "nvt_hremd_prod_scaled"
    "nvt_hremd_prod_scaled"
    "nvt_hremd_prod_scaled"
    "nvt_hremd_prod_scaled"
    "nvt_hremd_prod_scaled"
)
n_sims="${#dir_sims[@]}"

mkdir -p "logs"
mkdir -p "output"

# run analysis script
if [[ "${single_analysis}" != "1" ]]; then

    # for loop through all simulations
    for ((sim_idx = 0; sim_idx < n_sims; sim_idx++)); do
        echo "- Analysis on index $((sim_idx + 1))/${n_sims}..."
        echo "python -m sklearnex ${python_script} --dir ${dir_sims_base}/${dir_sims[${sim_idx}]}/${dir_append} --tag ${tags[${sim_idx}]} --fname ${name_sims[${sim_idx}]}"
        mkdir -p "output/${tags[${sim_idx}]}"
        {
            python -m sklearnex "${python_script}" \
                --dir "${dir_sims_base}/${dir_sims[${sim_idx}]}/${dir_append}" \
                --tag "${tags[${sim_idx}]}" \
                --fname "${name_sims[${sim_idx}]}"
        } | tee "output/${tags[${sim_idx}]}/${python_script%%.*}.log" 2>&1
    done

# run single analysis job
else
    echo "- Single analysis on index $((sim_idx + 1))/${n_sims}..."
    echo "python -m sklearnex ${python_script} --dir ${dir_sims_base}/${dir_sims[${sim_idx}]}/${dir_append} --tag ${tags[${sim_idx}]} --fname ${name_sims[${sim_idx}]}"
    mkdir -p "output/${tags[${sim_idx}]}"
    {
        python -m sklearnex "${python_script}" \
            --dir "${dir_sims_base}/${dir_sims[${sim_idx}]}/${dir_append}" \
            --tag "${tags[${sim_idx}]}" \
            --fname "${name_sims[${sim_idx}]}"
    } | tee "output/${tags[${sim_idx}]}/${python_script%%.*}.log" 2>&1
fi
