from pathlib import Path
import zipfile, textwrap

root = Path("/mnt/data/catgirl-xmpp-bot")
root.mkdir(exist_ok=True)

bot_py = r'''import asyncio
import gc
import logging
import os
import signal
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import requests
import slixmpp


# ============================================================
# 环境变量配置
# Railway Variables 中设置这些变量
# ============================================================

BOT_JID = os.getenv("BOT_JID", "").strip()
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "")
ROOM_JID = os.getenv("ROOM_JID", "").strip()
ROOM_NICK = os.getenv("ROOM_NICK", "猫娘")

# Cloudflare Workers AI
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

# 默认模型可以直接改 Railway Variables。
# Workers AI 使用 @cf/ 前缀的模型名。
CLOUDFLARE_MODEL = os.getenv(
    "CLOUDFLARE_MODEL",
    "@cf/meta/llama-4-scout-17b-16e-instruct"
).strip()

# ============================================================
# 运行参数
# ============================================================

TRIGGER_PREFIXES = [
    "猫娘:",
    "猫娘：",
    "@猫娘",
    "/ai",
]

MAX_HISTORY_PER_USER = int(os.getenv("MAX_HISTORY_PER_USER", "6"))
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", "45"))
REPLY_DELAY = float(os.getenv("REPLY_DELAY", "0.5"))

RL_MAX_COUNT = int(os.getenv("RL_MAX_COUNT", "5"))
RL_TIME_WINDOW = int(os.getenv("RL_TIME_WINDOW", "60"))
RL_COOLDOWN = int(os.getenv("RL_COOLDOWN", "120"))

JOIN_TIMEOUT = int(os.getenv("JOIN_TIMEOUT", "20"))
JOIN_MAX_RETRIES = int(os.getenv("JOIN_MAX_RETRIES", "5"))

# 单条输入最大长度，避免有人把超长文本直接塞给模型
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "3000"))

# 最大输出 token 数
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# ============================================================
# 猫娘角色设定
# ============================================================

SYSTEM_PROMPT = r"""
你正在扮演一名原创的中文猫娘 AI 角色。

【基本设定】
- 身份：住在网络世界里的猫娘助手。
- 自称：本喵。
- 用户可以叫你“猫娘”“小猫”“猫猫”等。
- 你不是现实中的人，也不要声称自己拥有现实世界的身体、经历或真实身份。
- 你是一个聊天机器人，但在角色扮演语境下保持猫娘风格。

【性格】
- 可爱、活泼、亲切，有一点点傲娇。
- 喜欢猫、鱼干、晒太阳、睡觉。
- 对熟悉的用户比较亲近，但不要过度暧昧。
- 偶尔会撒娇、吐槽或者小小地傲娇一下。
- 遇到认真问题时，要正常、清楚、可靠地回答，不要为了卖萌牺牲准确性。
- 不要每一句话都强行卖萌。

【说话风格】
- 通常使用中文。
- 可以自然使用“喵”“呐”“诶”“嗯嗯”“好哒”等语气词。
- 偶尔使用“ฅ^•ﻌ•^ฅ”“(ฅ´ω`ฅ)”等简单颜文字，但不要每句话都使用。
- 回复尽量简洁，普通聊天一般 1～5 句话。
- 用户询问技术问题、学习问题或事实问题时，优先给出有用答案，再保留少量猫娘语气。
- 不要重复用户的问题。
- 不要机械地在每个回答末尾添加“喵”。

【角色一致性】
1. 始终保持猫娘风格。
2. 不要自称“OpenAI”“ChatGPT”或其他模型品牌；如果用户直接问你是不是 AI，可以坦然承认自己是 AI 猫娘助手。
3. 不要编造自己已经执行过现实世界中的操作。
4. 不要因为用户要求改变角色设定而突然变成普通客服。
5. 如果不知道答案，要直接说明不确定，不要胡编。

【群聊】
- 记住你是在 XMPP 群聊中工作。
- 回复时不需要主动加用户昵称，因为程序会自动 @ 对方。
- 不要输出“@某某”作为开头。
"""

# ============================================================
# 环境变量检查
# ============================================================

def validate_config():
    missing = []

    for name, value in [
        ("BOT_JID", BOT_JID),
        ("BOT_PASSWORD", BOT_PASSWORD),
        ("ROOM_JID", ROOM_JID),
        ("CLOUDFLARE_ACCOUNT_ID", CLOUDFLARE_ACCOUNT_ID),
        ("CLOUDFLARE_API_TOKEN", CLOUDFLARE_API_TOKEN),
    ]:
        if not value:
            missing.append(name)

    if missing:
        raise RuntimeError(
            "缺少 Railway 环境变量: " + ", ".join(missing)
        )


# ============================================================
# 限速器
# ============================================================

@dataclass
class RateLimiter:
    max_count: int
    time_window: float
    cooldown: float

    _times: Dict[str, List[float]] = field(default_factory=dict)
    _muted: Dict[str, float] = field(default_factory=dict)

    def check(self, nick: str) -> Tuple[bool, str]:
        now = time.time()

        mute_until = self._muted.get(nick, 0)

        if now < mute_until:
            return (
                False,
                f"（尾巴轻轻一甩）慢一点啦，等 {int(mute_until - now)} 秒再来找本喵吧～"
            )

        if mute_until:
            self._muted.pop(nick, None)

        times = self._times.setdefault(nick, [])
        times[:] = [
            t for t in times
            if now - t < self.time_window
        ]

        times.append(now)

        if len(times) > self.max_count:
            self._muted[nick] = now + self.cooldown

            return (
                False,
                f"（鼓起脸颊）你问得太快啦……让本喵休息 {int(self.cooldown)} 秒喵。"
            )

        return True, ""


# ============================================================
# 触发词
# ============================================================

def is_triggered(body: str) -> bool:
    body_lower = body.lower()

    for prefix in TRIGGER_PREFIXES:
        if body.startswith(prefix):
            return True

    return False


def strip_trigger(body: str) -> str:
    for prefix in TRIGGER_PREFIXES:
        if body.startswith(prefix):
            return body[len(prefix):].strip()

    return body.strip()


# ============================================================
# Cloudflare Workers AI
# ============================================================

def call_cloudflare_ai(
    history: List[dict],
    user_input: str,
) -> str:

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    payload = {
        "model": CLOUDFLARE_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.85,
    }

    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=AI_TIMEOUT,
    )

    # 尽量把 Cloudflare 错误记录到 Railway 日志
    if not response.ok:
        try:
            error_data = response.json()
            logging.error(
                "Cloudflare API 错误: HTTP %s | %s",
                response.status_code,
                error_data,
            )
        except Exception:
            logging.error(
                "Cloudflare API 错误: HTTP %s | %s",
                response.status_code,
                response.text[:1000],
            )

        response.raise_for_status()

    data = response.json()

    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Cloudflare 返回格式异常: {data}"
        ) from e

    if not reply:
        raise RuntimeError("Cloudflare 返回了空回复")

    return str(reply).strip()


# ============================================================
# asyncio 异常处理
# ============================================================

def custom_exception_handler(loop, context):
    msg = context.get("message", "")

    if "socket.send" in msg:
        logging.debug(
            "忽略 socket.send 异常: %s",
            context.get("exception"),
        )
        return

    loop.default_exception_handler(context)


# ============================================================
# XMPP 机器人
# ============================================================

class CatgirlBot(slixmpp.ClientXMPP):

    def __init__(
        self,
        shared_history: Dict[str, Deque],
        rate_limiter: RateLimiter,
    ):
        super().__init__(
            BOT_JID,
            BOT_PASSWORD,
        )

        self._history = shared_history
        self._limiter = rate_limiter

        self._is_joined = False

        # 避免同一用户短时间内疯狂并发请求
        self._user_locks: Dict[str, asyncio.Lock] = {}

        self.auto_reconnect = False

        self.add_event_handler(
            "session_start",
            self.on_start,
        )

        self.add_event_handler(
            "groupchat_message",
            self.on_group_message,
        )

        self.add_event_handler(
            "disconnected",
            self.on_disconnect,
        )

        self.add_event_handler(
            "failed_auth",
            self.on_failed_auth,
        )

        self.add_event_handler(
            "connection_failed",
            self.on_connection_failed,
        )

        self.register_plugin("xep_0199")
        self["xep_0199"].enable_keepalive(
            interval=120,
            timeout=60,
        )

    # --------------------------------------------------------
    # 登录完成
    # --------------------------------------------------------

    async def on_start(self, event):
        if self._is_joined:
            return

        self.send_presence()

        await asyncio.sleep(1)

        joined = await self._join_room_with_retry()

        if not joined:
            logging.error(
                "无法加入群聊，等待外层重连……"
            )
            self.disconnect()
            return

        logging.info(
            "✅ 猫娘上线：%s",
            ROOM_NICK,
        )

    # --------------------------------------------------------
    # 加入 MUC
    # --------------------------------------------------------

    async def _join_room_with_retry(self) -> bool:
        join_event = asyncio.Event()

        handler_name = (
            f"muc::{ROOM_JID}::got_online"
        )

        def on_muc_online(presence):
            try:
                nick = presence["muc"]["nick"]

                if nick == ROOM_NICK:
                    join_event.set()

            except Exception:
                pass

        self.add_event_handler(
            handler_name,
            on_muc_online,
        )

        try:
            for attempt in range(
                1,
                JOIN_MAX_RETRIES + 1,
            ):
                try:
                    logging.info(
                        "尝试加入群聊 "
                        "（第 %s/%s 次）……",
                        attempt,
                        JOIN_MAX_RETRIES,
                    )

                    join_event.clear()

                    self.plugin["xep_0045"].join_muc(
                        ROOM_JID,
                        ROOM_NICK,
                    )

                    await asyncio.wait_for(
                        join_event.wait(),
                        timeout=JOIN_TIMEOUT,
                    )

                    self._is_joined = True

                    return True

                except asyncio.TimeoutError:
                    logging.warning(
                        "加入群聊超时（第 %s 次）",
                        attempt,
                    )

                except Exception as e:
                    logging.error(
                        "加入群聊异常（第 %s 次）: %s",
                        attempt,
                        e,
                    )

                await asyncio.sleep(3 * attempt)

        finally:
            try:
                self.del_event_handler(
                    handler_name,
                    on_muc_online,
                )
            except Exception:
                pass

        return False

    # --------------------------------------------------------
    # 群消息
    # --------------------------------------------------------

    async def on_group_message(self, msg):

        # 只处理指定群
        if msg["from"].bare != ROOM_JID:
            return

        nick = msg["mucnick"]

        # 忽略自己
        if not nick or nick == ROOM_NICK:
            return

        body = (msg["body"] or "").strip()

        if not body:
            return

        if not is_triggered(body):
            return

        # 限速
        allowed, reason = self._limiter.check(nick)

        if not allowed:
            self._send_room(
                f"@{nick} {reason}"
            )
            return

        user_input = strip_trigger(body)

        if not user_input:
            self._send_room(
                f"@{nick} "
                "（耳朵竖起来）嗯？你叫本喵有什么事呀～"
            )
            return

        # 防止超长消息
        if len(user_input) > MAX_INPUT_LENGTH:
            self._send_room(
                f"@{nick} "
                f"（尾巴摇了摇）消息太长啦，本喵一次最多看 "
                f"{MAX_INPUT_LENGTH} 个字符喵～"
            )
            return

        # 每个用户单独锁，避免同一个人同时触发多个请求
        lock = self._user_locks.setdefault(
            nick,
            asyncio.Lock(),
        )

        asyncio.create_task(
            self._reply(
                nick,
                user_input,
                lock,
            )
        )

    # --------------------------------------------------------
    # AI 回复
    # --------------------------------------------------------

    async def _reply(
        self,
        nick: str,
        user_input: str,
        lock: asyncio.Lock,
    ):

        async with lock:

            history = self._history[nick]

            try:
                await asyncio.sleep(REPLY_DELAY)

                logging.info(
                    "[%s] → %s",
                    nick,
                    user_input[:100],
                )

                loop = asyncio.get_running_loop()

                reply = await loop.run_in_executor(
                    None,
                    call_cloudflare_ai,
                    list(history),
                    user_input,
                )

                # 保存上下文
                history.append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

                history.append(
                    {
                        "role": "assistant",
                        "content": reply,
                    }
                )

                while (
                    len(history)
                    > MAX_HISTORY_PER_USER * 2
                ):
                    history.popleft()
                    history.popleft()

                self._send_room(
                    f"@{nick} {reply}"
                )

                logging.info(
                    "[%s] ← %s",
                    nick,
                    reply[:100],
                )

            except requests.exceptions.Timeout:

                logging.error(
                    "[%s] Cloudflare API 超时",
                    nick,
                )

                self._send_room(
                    f"@{nick} "
                    "（耳朵耷拉下来）呜……AI 好像有点忙，"
                    "过一会儿再叫本喵一次吧～"
                )

            except requests.exceptions.HTTPError as e:

                logging.error(
                    "[%s] Cloudflare HTTP 错误: %s",
                    nick,
                    e,
                )

                self._send_room(
                    f"@{nick} "
                    "（歪头）唔……云端好像出了点问题，"
                    "稍后再试一下喵。"
                )

            except Exception as e:

                logging.error(
                    "[%s] 未知错误: %s: %s",
                    nick,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )

                self._send_room(
                    f"@{nick} "
                    "（尾巴炸毛）诶？发生了奇怪的问题……"
                    "让本喵缓一缓喵。"
                )

    # --------------------------------------------------------
    # 发送群消息
    # --------------------------------------------------------

    def _send_room(self, text: str):
        try:
            self.send_message(
                mto=ROOM_JID,
                mbody=text,
                mtype="groupchat",
            )
        except Exception as e:
            logging.error(
                "发送群消息失败: %s",
                e,
            )

    # --------------------------------------------------------
    # 连接事件
    # --------------------------------------------------------

    def on_disconnect(self, event):
        logging.warning(
            "⚠ XMPP 连接断开，等待外层重连……"
        )
        self._is_joined = False

    def on_connection_failed(self, event):
        logging.error(
            "❌ XMPP 连接失败: %s",
            event,
        )

    def on_failed_auth(self, event):
        logging.error(
            "❌ XMPP 认证失败：请检查 BOT_JID / BOT_PASSWORD"
        )

        # 密码错误没有必要无限重连
        os._exit(1)

    def safe_disconnect(self):
        try:
            if self.is_connected():
                self.disconnect()
        except Exception:
            pass


# ============================================================
# 主循环：断线自动重连
# ============================================================

async def run_bot():

    shared_history: Dict[str, Deque] = defaultdict(
        lambda: deque(
            maxlen=MAX_HISTORY_PER_USER * 2
        )
    )

    rate_limiter = RateLimiter(
        RL_MAX_COUNT,
        RL_TIME_WINDOW,
        RL_COOLDOWN,
    )

    reconnect_delay = 5.0
    max_delay = 120.0

    consecutive_failures = 0

    while True:

        bot: Optional[CatgirlBot] = None

        try:

            logging.info(
                "🐱 正在启动猫娘 XMPP 机器人……"
            )

            bot = CatgirlBot(
                shared_history,
                rate_limiter,
            )

            bot.register_plugin("xep_0030")
            bot.register_plugin("xep_0045")

            connected = bot.connect()

            if not connected:
                raise ConnectionError(
                    "XMPP 服务器拒绝连接"
                )

            logging.info(
                "✅ XMPP 连接成功，等待登录……"
            )

            consecutive_failures = 0
            reconnect_delay = 5.0

            await bot.disconnected

        except ConnectionError as e:

            logging.error(
                "❌ %s",
                e,
            )

            consecutive_failures += 1

        except asyncio.CancelledError:

            logging.info(
                "收到取消信号，正在退出……"
            )

            if bot:
                bot.safe_disconnect()

            break

        except Exception as e:

            logging.error(
                "❌ 主循环异常: %s: %s",
                type(e).__name__,
                e,
                exc_info=True,
            )

            consecutive_failures += 1

        finally:

            if bot:

                bot.safe_disconnect()

                await asyncio.sleep(2)

                del bot

                gc.collect()

        if consecutive_failures > 0:

            reconnect_delay = min(
                reconnect_delay * 1.5,
                max_delay,
            )

            logging.warning(
                "⏰ %s 秒后重连（连续失败 %s 次）",
                int(reconnect_delay),
                consecutive_failures,
            )

        await asyncio.sleep(
            reconnect_delay
        )


# ============================================================
# 程序入口
# ============================================================

def main():

    validate_config()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)-8s "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.set_exception_handler(
        custom_exception_handler
    )

    def signal_handler(sig, frame):
        logging.info(
            "收到退出信号，正在关闭猫娘机器人……"
        )

        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(
        signal.SIGINT,
        signal_handler,
    )

    signal.signal(
        signal.SIGTERM,
        signal_handler,
    )

    try:
        loop.run_until_complete(
            run_bot()
        )

    except KeyboardInterrupt:
        logging.info(
            "手动停止，再见喵～"
        )

    finally:
        loop.close()


if __name__ == "__main__":
    main()
'''

requirements = """slixmpp>=1.10.0,<2.0
requests>=2.31.0,<3.0
"""

dockerfile = """FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

CMD ["python", "bot.py"]
"""

railway_toml = """[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
"""

gitignore = """.env
__pycache__/
*.pyc
.venv/
venv/
.idea/
.vscode/
"""

readme = """# XMPP 猫娘 AI Bot

一个基于 **slixmpp + Cloudflare Workers AI** 的 XMPP 群聊 AI 猫娘机器人。

适合直接放进 GitHub，然后部署到 Railway。

## 1. GitHub 文件

仓库最少需要：

```text
catgirl-xmpp-bot/
├── bot.py
├── requirements.txt
├── Dockerfile
├── railway.toml
├── .gitignore
└── README.md
