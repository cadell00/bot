#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu/Amazon Linux EC2 instance.
#   git clone <your repo> ~/roostoo-bot && cd ~/roostoo-bot && bash deploy/setup_ec2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"
USER_NAME="$(whoami)"

if command -v apt-get >/dev/null; then
  sudo apt-get update -y && sudo apt-get install -y python3 python3-venv python3-pip git
else
  sudo yum install -y python3 python3-pip git
fi

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

[ -f .env ] || { cp .env.example .env; echo ">>> Edit $APP_DIR/.env and add your Roostoo API keys"; }

sed -e "s#__APP_DIR__#$APP_DIR#g" -e "s#__USER__#$USER_NAME#g" deploy/roostoo-bot.service \
  | sudo tee /etc/systemd/system/roostoo-bot.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable roostoo-bot
echo "Setup done. Start with:  sudo systemctl start roostoo-bot"
echo "Follow logs with:        journalctl -u roostoo-bot -f   (or tail -f logs/bot.log)"
