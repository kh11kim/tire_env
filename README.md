
1. uv sync (+ select interpreter as python venv)
2. uv pip install -e .
3. uv run scripts/tire_mesh_gen.py
4. uv run scripts/gen_train_data.py --save_dir ~/dataset/tire_mask/ --num_cores 16 --num_data 1000 --num_max_tires 11
