source examples/libero/.venv/bin/activate


uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /home/ns1254/openpi/data/libero

/home/ns1254/.cache/huggingface/lerobot/nsojib/libero

uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero

uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero_low_mem_finetune --exp-name=my_experiment --overwrite

Fix checkpoint FileExistError: https://github.com/Physical-Intelligence/openpi/pull/520/


### run inference server
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero_low_mem_finetune --policy.dir=checkpoints/pi0_fast_libero_low_mem_finetune/my_experiment/4999

### run evaluation script
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
# Run the simulation
python examples/libero/main.py



