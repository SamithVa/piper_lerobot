# Hugging Face

## 登录

设置国内镜像加速：

````
export HF_ENDPOINT=https://hf-mirror.com
````

通过运行此命令将您的令牌添加到 CLI：

````
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
````

验证登录：

````
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
````

## 上传数据集

录制脚本 `--dataset.push_to_hub=true` 时会自动上传；手动上传：

````
hf upload jokeru/pick_and_place ~/.cache/huggingface/lerobot/jokeru/pick_and_place \
  --repo-type dataset \
  --revision "v3.0"
````

也可以用仓库里的辅助脚本：

````
python utils/push_dataset.py
````

## 上传模型 / checkpoints

````
hf upload jokeru/pick_and_place ~/.cache/huggingface/lerobot/jokeru/pick_and_place \
  --repo-type model \
  --revision "main"
````
