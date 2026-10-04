"""Telegram setup helper (talks only to Telegram, never to Roostoo).

1. In Telegram, message @BotFather -> /newbot -> copy the token into .env as TELEGRAM_BOT_TOKEN
2. Open a chat with your new bot and send it any message (e.g. "hi")
3. Run:  python3 scripts/telegram_setup.py
   - without TELEGRAM_CHAT_ID: prints your chat id -> put it in .env as TELEGRAM_CHAT_ID
   - with TELEGRAM_CHAT_ID: sends a test message
"""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.config import BotConfig  # noqa: E402

cfg = BotConfig()
if not cfg.telegram_token:
    sys.exit("TELEGRAM_BOT_TOKEN is not set in .env (get one from @BotFather with /newbot)")

api = f"https://api.telegram.org/bot{cfg.telegram_token}"
me = requests.get(f"{api}/getMe", timeout=10).json()
if not me.get("ok"):
    sys.exit(f"Token rejected by Telegram: {me}")
print(f"bot: @{me['result']['username']}")

if not cfg.telegram_chat_id:
    updates = requests.get(f"{api}/getUpdates", timeout=10).json().get("result", [])
    chats = {u["message"]["chat"]["id"]: u["message"]["chat"].get("first_name") or u["message"]["chat"].get("title")
             for u in updates if "message" in u}
    if not chats:
        sys.exit("No messages yet: open a chat with your bot in Telegram, send it 'hi', then run this again.")
    for cid, name in chats.items():
        print(f"chat id: {cid}  ({name})")
    print("Add to .env:  TELEGRAM_CHAT_ID=<chat id above>")
else:
    r = requests.post(f"{api}/sendMessage", data={"chat_id": cfg.telegram_chat_id,
                      "text": f"[{cfg.bot_name}] ✅ Telegram notifications are working."}, timeout=10).json()
    print("test message sent" if r.get("ok") else f"send failed: {r}")