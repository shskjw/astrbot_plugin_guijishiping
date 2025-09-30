import asyncio
import base64
import json
import time
import os
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import aiohttp
import aiofiles
import aiofiles.os
from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core import AstrBotConfig
import astrbot.api.message_components as Comp
from astrbot.core.platform.astr_message_event import AstrMessageEvent
class SiliconflowPlugin(Star):
    """
    astrbot_plugin_guijishiping by shskjw
    Version: 1.0.0 (Refactored)
    Description: 硅基流动api视频，可以制作动态壁纸之类的
    """
    class APIClient:
        def __init__(self, proxy_url: Optional[str] = None):
            self.proxy = proxy_url
            self.session = aiohttp.ClientSession()
            if self.proxy:
                logger.info(f"[SiliconFlow] APIClient 使用代理: {self.proxy}")

        async def _download_image(self, url: str) -> Optional[bytes]:
            try:
                async with self.session.get(url, proxy=self.proxy, timeout=30) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except Exception as e:
                logger.error(f"[SiliconFlow] 图片下载失败: {e}", exc_info=True)
                return None

        async def _load_bytes(self, src: str) -> Optional[bytes]:
            if src.startswith("http"):
                return await self._download_image(src)
            elif src.startswith("base64://"):
                return base64.b64decode(src[9:])
            return None

        async def get_image_from_event(self, event: AstrMessageEvent) -> Optional[bytes]:
            for seg in event.message_obj.message:
                if isinstance(seg, Comp.Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Comp.Image):
                            if s_chain.url and (img := await self._load_bytes(s_chain.url)): return img
                            if s_chain.file and (img := await self._load_bytes(s_chain.file)): return img
            for seg in event.message_obj.message:
                if isinstance(seg, Comp.Image):
                    if seg.url and (img := await self._load_bytes(seg.url)): return img
                    if seg.file and (img := await self._load_bytes(seg.file)): return img
            return None

        async def terminate(self):
            if self.session and not self.session.closed:
                await self.session.close()

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()
        self.user_counts_file = self.plugin_data_dir / "sf_user_counts.json"
        self.user_counts: Dict[str, int] = {}
        self.group_counts_file = self.plugin_data_dir / "sf_group_counts.json"
        self.group_counts: Dict[str, int] = {}
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.api_client: Optional[SiliconflowPlugin.APIClient] = None

    async def initialize(self):
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.api_client = self.APIClient(proxy_url)
        await self._load_user_counts()
        await self._load_group_counts()
        logger.info("SiliconFlow 视频生成插件已加载")
        if not self.conf.get("api_keys"):
            logger.warning("[SiliconFlow] 未配置任何 API 密钥，插件无法工作")

    # --- 次数管理 ---
    async def _load_user_counts(self):
        if not self.user_counts_file.exists(): self.user_counts = {}; return
        try:
            content = self.user_counts_file.read_text("utf-8")
            self.user_counts = {str(k): v for k, v in json.loads(content).items()}
        except Exception as e:
            logger.error(f"加载用户次数文件时发生错误: {e}", exc_info=True)

    async def _save_user_counts(self):
        try:
            self.user_counts_file.write_text(json.dumps(self.user_counts, ensure_ascii=False, indent=4), "utf-8")
        except Exception as e:
            logger.error(f"保存用户次数文件时发生错误: {e}", exc_info=True)

    def _get_user_count(self, user_id: str) -> int:
        return self.user_counts.get(str(user_id), 0)

    async def _decrease_user_count(self, user_id: str):
        count = self._get_user_count(str(user_id))
        if count > 0: self.user_counts[str(user_id)] = count - 1; await self._save_user_counts()

    async def _load_group_counts(self):
        if not self.group_counts_file.exists(): self.group_counts = {}; return
        try:
            content = self.group_counts_file.read_text("utf-8")
            self.group_counts = {str(k): v for k, v in json.loads(content).items()}
        except Exception as e:
            logger.error(f"加载群组次数文件时发生错误: {e}", exc_info=True)

    async def _save_group_counts(self):
        try:
            self.group_counts_file.write_text(json.dumps(self.group_counts, ensure_ascii=False, indent=4), "utf-8")
        except Exception as e:
            logger.error(f"保存群组次数文件时发生错误: {e}", exc_info=True)

    def _get_group_count(self, group_id: str) -> int:
        return self.group_counts.get(str(group_id), 0)

    async def _decrease_group_count(self, group_id: str):
        count = self._get_group_count(str(group_id))
        if count > 0: self.group_counts[str(group_id)] = count - 1; await self._save_group_counts()

    async def _download_video_async(self, url: str) -> Optional[str]:
        filename = f"siliconflow_video_{uuid.uuid4()}.mp4"
        filepath = str(self.plugin_data_dir / filename) # 转换为str
        logger.info(f"开始异步下载视频到: {filepath}")
        try:
            async with self.api_client.session.get(url, timeout=300) as resp:
                resp.raise_for_status()
                async with aiofiles.open(filepath, 'wb') as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        await f.write(chunk)
            logger.info(f"视频已下载保存为: {filename}")
            return filepath
        except Exception as e:
            logger.error(f"异步下载视频时发生异常: {e}", exc_info=True)
            if await aiofiles.os.path.exists(filepath):
                await aiofiles.os.remove(filepath)
            return None

    # --- 管理指令 ---
    @filter.command("视频增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        match = re.search(r"(\d+)\s+(\d+)", event.message_str.strip())
        if not match: yield event.plain_result('格式错误: #视频增加用户次数 <QQ号> <次数>'); return
        target_qq, count = match.group(1), int(match.group(2))
        current_count = self._get_user_count(target_qq)
        self.user_counts[str(target_qq)] = current_count + count
        await self._save_user_counts()
        yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次，TA当前剩余 {current_count + count} 次。")

    @filter.command("视频增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        match = re.search(r"(\d+)\s+(\d+)", event.message_str.strip())
        if not match: yield event.plain_result('格式错误: #视频增加群组次数 <群号> <次数>'); return
        target_group, count = match.group(1), int(match.group(2))
        current_count = self._get_group_count(target_group)
        self.group_counts[str(target_group)] = current_count + count
        await self._save_group_counts()
        yield event.plain_result(f"✅ 已为群组 {target_group} 增加 {count} 次，该群当前剩余 {current_count + count} 次。")

    @filter.command("视频查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()
        user_count = self._get_user_count(user_id)
        reply_msg = f"您好，您当前个人剩余次数为: {user_count}"
        group_id = event.get_group_id()
        if group_id and self.conf.get("enable_group_limit", False):
            group_count = self._get_group_count(group_id)
            reply_msg += f"\n本群共享剩余次数为: {group_count}"
        yield event.plain_result(reply_msg)

    async def terminate(self):
        if self.api_client: await self.api_client.terminate()
        logger.info("[SiliconFlow] 插件已终止")

    def is_global_admin(self, event: AstrMessageEvent):
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    async def _get_api_key(self) -> Optional[str]:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    # --- API 调用 ---
    async def _submit_task(self, prompt: str, image_bytes: Optional[bytes], num_frames: int) -> Tuple[Optional[str], str]:
        api_url = self.conf.get("api_url", "https://api.siliconflow.cn")
        api_key = await self._get_api_key()
        if not api_key: return None, "无可用的 API Key"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"model": self.conf.get("default_model"),"prompt": prompt, "negative_prompt": "low quality, bad quality, blurry","steps": 25, "guidance_scale": 7, "num_frames": num_frames}
        if image_bytes:
            payload["image"] = base64.b64encode(image_bytes).decode("utf-8")
            payload["motion_bucket_id"] = 127
            payload["cond_aug"] = 0.02
        try:
            async with self.api_client.session.post(f"{api_url}/v1/video/submit", json=payload, headers=headers, proxy=self.api_client.proxy, timeout=60) as resp:
                data = await resp.json()
                if resp.status != 200: return None, f"任务提交失败: {data.get('error', {}).get('message', str(data))}"
                return data.get("requestId"), "提交成功"
        except Exception as e: return None, f"网络错误: {e}"

    async def _poll_for_result(self, request_id: str) -> Tuple[Optional[str], str]:
        api_url = self.conf.get("api_url", "https://api.siliconflow.cn")
        timeout = self.conf.get("polling_timeout", 300)
        interval = self.conf.get("polling_interval", 5)
        start_time = time.time()
        while time.time() - start_time < timeout:
            api_key = await self._get_api_key()
            if not api_key: await asyncio.sleep(interval); continue
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"requestId": request_id}
            try:
                async with self.api_client.session.post(f"{api_url}/v1/video/status", json=payload, headers=headers, proxy=self.api_client.proxy, timeout=30) as resp:
                    if resp.status != 200: await asyncio.sleep(interval); continue
                    data = await resp.json()
                    status = data.get("status")
                    if status in ["Succeed", "completed"]:
                        video_url = None
                        if results := data.get("results"):
                            if videos := results.get("videos"):
                                if isinstance(videos, list) and len(videos) > 0 and isinstance(videos[0], dict):
                                    video_url = videos[0].get("url")
                        
                        if not video_url:
                            video_url = data.get("video_url")  # Fallback
                        
                        if video_url: 
                            return video_url, "生成成功"
                        else: 
                            logger.error(f"[SiliconFlow] 成功响应但未找到视频链接: {json.dumps(data)}"); return None, "成功响应但未找到视频链接"
                    elif status in ["Failed", "failed"]:
                        return None, f"任务生成失败: {data.get('reason', data.get('error', '未知错误'))}"
                    await asyncio.sleep(interval)
            # 捕获异常时记录详细信息
            except Exception as e:
                logger.warning(f"[SiliconFlow] 轮询状态时发生异常: {e}", exc_info=True)
                await asyncio.sleep(interval)
        return None, "任务超时"

    async def _check_permissions(self, event: AstrMessageEvent) -> Tuple[bool, Optional[str]]:
        """检查用户是否有权限执行操作，返回 (是否通过, 错误信息)"""
        if self.is_global_admin(event):
            return True, None

        sender_id = event.get_sender_id()
        group_id = event.get_group_id()

        # 黑名单检查
        if self.conf.get("user_blacklist", []) and sender_id in self.conf.get("user_blacklist", []):
            return False, None # 黑名单用户静默失败
        if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist", []):
            return False, None # 非白名单群聊静默失败
        
        # 白名单检查
        if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []):
            return False, "抱歉，您不在本功能的使用白名单中。"

        # 次数检查
        user_limit_on = self.conf.get("enable_user_limit", True)
        group_limit_on = self.conf.get("enable_group_limit", False) and group_id
        user_count = self._get_user_count(sender_id)
        group_count = self._get_group_count(group_id) if group_id else 0

        has_group_permission = not group_limit_on or group_count > 0
        has_user_permission = not user_limit_on or user_count > 0

        if group_id:
            if not has_group_permission and not has_user_permission:
                return False, "❌ 本群次数与您的个人次数均已用尽，请联系管理员补充。"
        else: # 私聊
            if not has_user_permission:
                return False, "❌ 您的使用次数已用完，请联系管理员补充。"
        
        return True, None


    # --- 核心指令 ---
    @filter.command("生成视频", prefix_optional=True)
    async def on_video_generate(self, event: AstrMessageEvent):
        message_text = event.message_str.strip()
        DEFAULT_FPS = self.conf.get("default_fps", 8)
        DEFAULT_SECONDS = 4
        seconds_match = re.search(r"--s\s+(\d+)", message_text)
        seconds = DEFAULT_SECONDS
        if seconds_match:
            seconds = int(seconds_match.group(1))
            prompt = re.sub(r"--s\s+\d+", "", message_text).strip()
        else:
            prompt = message_text
        num_frames = seconds * DEFAULT_FPS
        if not prompt: yield event.plain_result("🤔 用法: #生成视频 [--s 秒数] <提示词> [图片]"); return

        can_proceed, error_message = await self._check_permissions(event)
        if not can_proceed:
            if error_message: # 如果有错误信息，则发送
                yield event.plain_result(error_message)
            return

        sender_id = event.get_sender_id()
        group_id = event.get_group_id()

        image_bytes = await self.api_client.get_image_from_event(event)
        yield event.plain_result(f"✅ 任务已提交 ({'图生视频' if image_bytes else '文生视频'}, 期望 {seconds}秒 @ {DEFAULT_FPS}fps)，正在排队生成...")
        
        request_id, error_msg = await self._submit_task(prompt, image_bytes, num_frames)
        if not request_id: yield event.plain_result(f"❌ 提交失败: {error_msg}"); return

        video_url, status_msg = await self._poll_for_result(request_id)
        if not video_url: yield event.plain_result(f"❌ 处理失败: {status_msg}"); return
            
        yield event.plain_result("✅ 生成成功，正在下载视频到本地...")
        filepath = await self._download_video_async(video_url)
        if not filepath: yield event.plain_result(f"❌ 视频下载失败，请尝试手动下载:\n{video_url}"); return
        
        yield event.plain_result("✅ 下载完成，正在发送文件...")

        if not self.is_global_admin(event):
            if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                await self._decrease_group_count(group_id)
            elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                await self._decrease_user_count(sender_id)

        try:
            video_component = Comp.Video.fromFileSystem(path=filepath, name="generated_video.mp4")
            yield event.chain_result([video_component])
            yield event.plain_result(f"🎬 视频文件已发送！\n下载链接：{video_url}")
        except Exception as e:
            logger.error(f"发送文件时失败: {e}", exc_info=True)
            yield event.plain_result(f"🎬 文件发送失败，请点击链接下载：\n{video_url}")
        finally:
            if await aiofiles.os.path.exists(filepath):
                await aiofiles.os.remove(filepath)
                logger.info(f"已清理临时文件: {filepath}")

        caption_parts = []
        is_master = self.is_global_admin(event)
        if is_master: caption_parts.append("剩余次数: ∞")
        else:
            if self.conf.get("enable_user_limit", True): caption_parts.append(f"个人剩余: {self._get_user_count(sender_id)}")
            if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(f"本群剩余: {self._get_group_count(group_id)}")
        if caption_parts: yield event.plain_result(" | ".join(caption_parts))
