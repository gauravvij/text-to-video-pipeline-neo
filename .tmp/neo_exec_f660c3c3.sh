cd /root/i2v
git init -b main
cat <<EOF > .gitignore
venv/
__pycache__/
*.pth
*.bin
*.pt
*.ckpt
*.safetensors
data/outputs/*.mp4
output_intermediate.png
performance_metrics.json
.env
EOF
git add .
git commit -m "Initial commit: Text-to-Image-to-Video pipeline CLI"
env | grep -i GITHUB