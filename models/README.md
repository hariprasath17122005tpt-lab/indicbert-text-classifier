# Model checkpoints

Fine-tuned checkpoints are stored locally under this directory. Model weights
and large tokenizer assets (including `*.safetensors` and `tokenizer.json`) are
excluded from Git; the current local weight file is about 1.06 GB.

For sharing or deployment, upload the fine-tuned checkpoint and its required
tokenizer/configuration files to a Hugging Face model repository. Set the
repository ID in `model.hf_repo_id` in the project-root `config.yaml`. Never
commit access tokens or private model credentials.