"""
main.py
========
Pipeline orchestrator - runs all active stages in order.

Each stage is called as a subprocess using the correct conda environment python.
This is necessary because YOLO, SAM2, and PaddleOCR cannot share one environment.

The python paths for each environment are set in config.yaml.

USAGE:
  conda activate SAM2
  cd /path/to/pipeline/
  python main.py config.yaml

  OR to run a single stage:
  python main.py config.yaml --stage detection

STAGES (in order):
  0. pre_ocr      - detect label + weight-screen regions (YOLO)
  1. ocr          - extract plant name + weight from images
  2. detection    - detect kernel bounding boxes (YOLO11x)
  3. segmentation - segment kernels, save subimages + binary masks (SAM2)
  4. measurements - predict main axis with ResNet and compute morphology
  5. shapes       - compute representative kernel shape per plant
  6. vae_encode   - encode 100-dim width profiles to 5-dim latent traits
  7. assembly     - export separate individual and plant-median result CSVs

Contours are now extracted on the fly inside downstream scripts, so there is
no standalone contour-data or merging stage in the active pipeline anymore.
"""

import os
import sys
import subprocess
import argparse
import time


from utils.config import load_config


STAGES = [
    {
        'name': 'pre_ocr',
        'script': 'ocr/yolo_label_detect.py',
        'python_key': 'yolo_python',
        'description': 'Detecting label and weight-screen regions (YOLO)...',
    },
    {
        'name': 'ocr',
        'script': 'ocr/metadata_extraction.py',
        'python_key': 'paddle_python',
        'description': 'Extracting plant name and weight (PaddleOCR)...',
    },
    {
        'name': 'detection',
        'script': 'detection/yolo_detect.py',
        'python_key': 'yolo_python',
        'description': 'Detecting kernel bounding boxes (YOLO11x)...',
    },
    {
        'name': 'segmentation',
        'script': 'segmentation/sam_segment.py',
        'python_key': 'sam2_python',
        'description': 'Segmenting kernels and saving subimages (SAM2)...',
    },
    {
        'name': 'measurements',
        'script': 'measurements/kernel_metrics.py',
        'python_key': 'base_python',
        'description': 'Predicting directed main axes with ResNet, then computing morphology...',
    },
    {
        'name': 'shapes',
        'script': 'processing/representative_shape.py',
        'python_key': 'base_python',
        'description': 'Computing representative kernel shape per plant...',
    },
    {
        'name': 'vae_encode',
        'script': 'vae/vae_encode.py',
        'python_key': 'base_python',
        'description': 'VAE latent encoding — 100-dim profiles → 5-dim latent traits...',
    },
    {
        'name': 'assembly',
        'script': 'output/assembler.py',
        'python_key': 'base_python',
        'description': 'Exporting final individual and plant-median CSVs...',
    },
]


def run_stage(stage, config_path, python_exe, pipeline_dir):
    """
    Run a single pipeline stage as a subprocess.

    Args:
        stage: stage dict from STAGES list
        config_path: absolute path to config.yaml
        python_exe: path to the python executable for this stage's environment
        pipeline_dir: absolute path to the pipeline directory

    Returns:
        True if the stage succeeded, False if it failed.
    """
    script = os.path.join(pipeline_dir, stage['script'])

    if not os.path.exists(script):
        print(f'  ERROR: Script not found: {script}')
        return False

    if not os.path.exists(python_exe):
        print(f'  ERROR: Python executable not found: {python_exe}')
        print('  Check the environments section in config.yaml')
        return False

    print(f'\n{"=" * 60}')
    print(f'STAGE: {stage["name"].upper()}')
    print(f'{stage["description"]}')
    print(f'  Script  : {stage["script"]}')
    print(f'  Python  : {python_exe}')
    print(f'{"=" * 60}')

    start_time = time.time()

    result = subprocess.run(
        [python_exe, script, config_path],
        check=False,
        text=True,
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f'\n  STAGE FAILED: {stage["name"]} (exit code {result.returncode})')
        print(f'  Elapsed: {elapsed:.1f}s')
        print('  Fix the error above before continuing.')
        return False

    print(f'  Stage completed in {elapsed:.1f}s')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Maize Kernel Phenotyping Pipeline'
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='config.yaml',
        help='Path to config.yaml (default: config.yaml)'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default=None,
        help=(
            'Run only a specific stage. '
            'Options: ' + ', '.join(s['name'] for s in STAGES)
        )
    )
    parser.add_argument(
        '--from-stage',
        type=str,
        default=None,
        dest='from_stage',
        help='Resume from a specific stage (skips earlier stages).'
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f'ERROR: Config file not found: {config_path}')
        sys.exit(1)

    config = load_config(config_path)
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    envs = config.get('environments', {})

    if args.stage:
        matching = [s for s in STAGES if s['name'] == args.stage]
        if not matching:
            valid = ', '.join(s['name'] for s in STAGES)
            print(f'ERROR: Unknown stage "{args.stage}". Valid: {valid}')
            sys.exit(1)
        stages_to_run = matching
    elif args.from_stage:
        names = [s['name'] for s in STAGES]
        if args.from_stage not in names:
            valid = ', '.join(names)
            print(f'ERROR: Unknown stage "{args.from_stage}". Valid: {valid}')
            sys.exit(1)
        start_idx = names.index(args.from_stage)
        stages_to_run = STAGES[start_idx:]
        print(f'Resuming from stage: {args.from_stage}')
    else:
        stages_to_run = STAGES

    print('\nMaize Kernel Phenotyping Pipeline')
    print(f'Config: {config_path}')
    print(f'Output: {config["output"]["base_dir"]}')
    print(f'Stages to run: {", ".join(s["name"] for s in stages_to_run)}')

    total_start = time.time()
    failed_stage = None

    for stage in stages_to_run:
        python_exe = envs.get(stage['python_key'], '')
        if not python_exe:
            print(f'\nERROR: No python path set for "{stage["python_key"]}" in config.yaml')
            print('Update the environments section in config.yaml.')
            sys.exit(1)

        success = run_stage(stage, config_path, python_exe, pipeline_dir)

        if not success:
            failed_stage = stage['name']
            break

    total_elapsed = time.time() - total_start

    print(f'\n{"=" * 60}')
    if failed_stage:
        print(f'PIPELINE STOPPED at stage: {failed_stage}')
        print(f'Fix the error and re-run with:  python main.py config.yaml --from-stage {failed_stage}')
    else:
        print('PIPELINE COMPLETE')
        print(f'Total time: {total_elapsed / 60:.1f} minutes')
        output_dir = config['output']['base_dir']
        final_individual_csv = os.path.join(
            output_dir,
            config['output'].get('final_individual_csv', 'final_output_individual.csv')
        )
        final_plant_median_csv = os.path.join(
            output_dir,
            config['output'].get('final_plant_median_csv', 'final_output_plant_median.csv')
        )
        print(f'Individual results  : {final_individual_csv}')
        print(f'Plant median results: {final_plant_median_csv}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
