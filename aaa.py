from huggingface_hub import snapshot_download

repo_id = "facebook/sam-vit-huge"  # 模型在huggingface上的名称
local_dir = "/llm-cfs-nj/person/harryyhwang/ft_local/ft_local/ROOT/ckpts/sam-vit-huge"  # 本地模型存储的地址
# token = "XXX"  # 在hugging face上生成的 access token
# proxies = {'http': 'XXXX', 'https': 'XXXX',}

snapshot_download(
    repo_id=repo_id,
    repo_type="model",  # 可选 [dataset,model,space] 
    local_dir=local_dir,
    # revision="refs/pr/1", # default latest version
    # allow_patterns=["*.md", "*.json"],  # allow download
    # ignore_patterns="vocab.json",  # ignore download
    # token=token,
    # proxies=proxies
)