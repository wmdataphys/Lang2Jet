## command to run this script
# python submit.py --cfg config/multijet_training.yaml --train_script train_iter_jet_class_pos_encode_allign_loss.py --slurm_template slurm_jetclass_alligned_loss --output_dir /sciclone/home/hnayak/scr10/Transfer_Learning/JetClass_alligned_loss/ --max_runs 5
# python submit.py --cfg config/mlp_classifier_head1_layer_norm.yaml --train_script mlp_classifier_head_bert_parquet.py --slurm_template WMslurm_sub --output_dir /sciclone/home/hnayak/scr10/Transfer_Learning/JetClass_NEW_h5/ --max_runs 2
import os
import argparse
import shutil
from omegaconf import OmegaConf
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Submit multiple SLURM jobs with config variations.")
    parser.add_argument('--cfg', type=str, required=True, help='Path to the config file (YAML or JSON).')
    parser.add_argument('--train_script', type=str, required=True, help="Path to the train_iter_multijet script")
    parser.add_argument('--slurm_template', type=str, required=True, help='Path to the SLURM job template file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Base output directory for job folders.')
    
    parser.add_argument('--max_runs', type=int, required=True, help='Maximum number of runs/jobs to create.')
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.cfg)
    with open(args.slurm_template, 'r') as f:
        slurm_template = f.read()
    for run in range(1, args.max_runs + 1):
        t=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        run_dir = os.path.join(args.output_dir, f'{cfg.get("model_name")}', f'run_{run}', t)
        os.makedirs(run_dir, exist_ok=True)
        print(f'Created directory: {run_dir}')
		# Update config for this run
        cfg['run'] = run
        cfg['save_dir'] = run_dir
        cfg['seed'] = run
        cfg_path = os.path.join(run_dir, 'config.yaml')
        OmegaConf.save(cfg, cfg_path)
        

		# Prepare SLURM script for this run
        log_dir = os.path.join(run_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        else:
            os.system(f"rm -rf {log_dir}/*")
            
        python_script = os.path.realpath(args.train_script)
        slurm_script = slurm_template.replace('__PYTHONSCRIPT__', f"{python_script}")
        slurm_script = slurm_script.replace('__CFG__', f'{cfg_path}')
        slurm_script = slurm_script.replace('__OUTPUT__', os.path.join(log_dir, f'job_output_{run}.out'))
        slurm_script = slurm_script.replace('__ERROR__', os.path.join(log_dir, f'job_error_{run}.err'))
        slurm_script = slurm_script.replace('__JOB_NAME__', f'{cfg.get("model_name")}_run_{run}')
        
        slurm_script_path = os.path.join(run_dir, 'slurm_job.sh')
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_script)
        
        print(f'Created run {run}: config at {cfg_path}, SLURM script at {slurm_script_path}')
        

        # lets submit the job
        os.system(f'sbatch {slurm_script_path}')

if __name__ == '__main__':
	main()