"""
NexusQuant - Alert Channels.

Delivery layer for live signals. Every channel implements ``send(text)``
and failures are caught per-channel so one broken webhook never kills the
run. Channels are selected from ``config/settings.yaml`` (``live.channels``)
plus environment overrides:

* ``discord``  - POST the message to a Discord webhook URL (from
  ``NEXUS_DISCORD_WEBHOOK`` env var or ``live.discord_webhook`` in
  settings.yaml). Discord supports 2000-char messages; longer alerts are
  split on line boundaries.
* ``telegram`` - send via the Telegram Bot API ``sendMessage`` (token from
  ``NEXUS_TELEGRAM_BOT_TOKEN``, chat id from ``NEXUS_TELEGRAM_CHAT_ID``).
  Telegram supports 4096-char messages; alerts are split to fit.
* ``console``  - print to stdout (always available, ideal for cron + logs).
* ``file``     - append to ``live.log_file`` (default ``logs/live.log``).

Only the stdlib is used (``urllib`` for the HTTP calls) - no new
dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("nexus.live.alerts")

DISCORD_MAX = 2000
TELEGRAM_MAX = 4096
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


class Channel:
    """Base channel: ``send(text) -> bool``."""

    name = "base"

    def send(self, text: str) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        pass


class DiscordChannel(Channel):
    """Send a plain-text message to a Discord webhook URL."""

    name = "discord"

    def __init__(self, webhook_url: str):
        if not webhook_url:
            raise ValueError("Discord webhook URL is empty")
        self.webhook_url = webhook_url

    def send(self, text: str) -> bool:
        ok = True
        for chunk in _split_text(text, DISCORD_MAX):
            ok = _post_json(self.webhook_url, {"content": chunk}) and ok
        return ok


class TelegramChannel(Channel):
    """Send a plain-text message via the Telegram Bot API.

    Requires a bot token (from BotFather) and the numeric chat id of the
    recipient. Uses ``sendMessage`` with a form-encoded POST; the API
    returns HTTP 200 with ``ok:false`` on errors, so the response body is
    parsed (unlike the Discord webhook, where status < 300 is enough).
    """

    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str):
        if not bot_token:
            raise ValueError("Telegram bot token is empty")
        if not chat_id:
            raise ValueError("Telegram chat id is empty")
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send(self, text: str) -> bool:
        ok = True
        for chunk in _split_text(text, TELEGRAM_MAX):
            ok = self._send_chunk(chunk) and ok
        return ok

    def _send_chunk(self, chunk: str) -> bool:
        try:
            url = TELEGRAM_API.format(token=self.bot_token)
            data = urllib.parse.urlencode(
                {
                    "chat_id": self.chat_id,
                    "text": chunk,
                    "disable_web_page_preview": "true",
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "NexusQuant/0.1",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8") or "{}")
                ok = bool(body.get("ok"))
                if not ok:
                    # ok:false is the API's error channel (wrong chat id,
                    # bot not started, ...) - surface the reason.
                    desc = body.get("description") or "unknown error"
                    log.warning("telegram API error: %s", desc)
                return ok
        except Exception as exc:
            log.warning("telegram send failed: %s", exc)
            return False


class ConsoleChannel(Channel):
    name = "console"

    def send(self, text: str) -> bool:
        print("\n" + "=" * 64)
        print(text)
        print("=" * 64 + "\n", file=sys.stdout)
        return True


class FileChannel(Channel):
    name = "file"

    def __init__(self, path: str = "logs/live.log"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, text: str) -> bool:
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write("\n" + "=" * 64 + "\n")
                fh.write(text + "\n")
                fh.write("=" * 64 + "\n")
            return True
        except OSError as exc:
            log.warning("file channel write failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_text(text: str, limit: int) -> List[str]:
    """Split long text so every chunk fits ``limit``.

    Lines are kept whole where possible; a single line longer than the
    limit is hard-split on characters (guaranteeing the cap, so a long
    line can never silently exceed Discord's message limit).
    """
    if len(text) <= limit:
        return [text]
    lines = text.splitlines()
    chunks, cur = [], []
    cur_len = 0
    for line in lines:
        if len(line) > limit:  # hard-split the over-long line itself
            if cur:
                chunks.append("\n".join(cur))
                cur, cur_len = [], 0
            for i in range(0, len(line), limit):
                chunks.append(line[i : i + limit])
            continue
        if cur and cur_len + len(line) + 1 > limit:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _post_json(url: str, payload: Dict) -> bool:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "NexusQuant/0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status < 300
    except Exception as exc:
        log.warning("webhook POST failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Registry / hub
# ---------------------------------------------------------------------------


def build_channels(cfg: dict, env: Optional[dict] = None) -> List[Channel]:
    """
    Instantiate the channels enabled in ``cfg`` (a dict like
    ``settings['live']``). ``env`` is os.environ by default.
    """
    env = env if env is not None else os.environ
    channels: List[Channel] = []

    # Console is the default - always useful, never fails.
    if cfg.get("console", True):
        channels.append(ConsoleChannel())

    if cfg.get("file", True):
        channels.append(FileChannel(cfg.get("log_file", "logs/live.log")))

    if cfg.get("discord", False):
        url = env.get("NEXUS_DISCORD_WEBHOOK") or cfg.get("discord_webhook") or ""
        if url:
            try:
                channels.append(DiscordChannel(url))
            except ValueError:
                log.warning("discord enabled but webhook URL is missing")
        else:
            log.warning("discord enabled but NEXUS_DISCORD_WEBHOOK is unset")

    if cfg.get("telegram", False):
        token = (
            env.get("NEXUS_TELEGRAM_BOT_TOKEN") or cfg.get("telegram_bot_token") or ""
        )
        chat_id = env.get("NEXUS_TELEGRAM_CHAT_ID") or cfg.get("telegram_chat_id") or ""
        if token and chat_id:
            try:
                channels.append(TelegramChannel(token, chat_id))
            except ValueError:
                log.warning("telegram enabled but token/chat id invalid")
        else:
            log.warning(
                "telegram enabled but NEXUS_TELEGRAM_BOT_TOKEN / "
                "NEXUS_TELEGRAM_CHAT_ID are unset"
            )

    return channels


def send_all(channels: List[Channel], text: str) -> int:
    """Send to every channel; returns the number that succeeded."""
    if not text.strip():
        return 0
    n_ok = 0
    for ch in channels:
        try:
            if ch.send(text):
                n_ok += 1
        except Exception as exc:
            log.warning("channel %s failed: %s", ch.name, exc)
    return n_ok
