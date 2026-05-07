import sys
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

from settings import config
from network import api_client


class AddKeyRequest(BaseModel):
    name: str
    key: str


class AddModelRequest(BaseModel):
    name: str
    model_id: str
    category: str = "chat"


class MoveModelRequest(BaseModel):
    direction: str


app = FastAPI(title="ModelScopeApiRouter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return {"message": "ModelScopeApiRouter is running"}


@app.get("/v1/models")
async def list_models():
    """OpenAI 兼容的模型列表端点"""
    models_list = []
    for model in config.MODELS:
        models_list.append({
            "id": model.get("name", model.get("model_id", "")),
            "object": "model",
            "created": 0,
            "owned_by": model.get("category", "chat")
        })
    # 加上简化的别名：chat / txt2img / img2img / vision
    for alias in ["chat", "txt2img", "img2img", "vision"]:
        if not any(m["id"] == alias for m in models_list):
            models_list.append({
                "id": alias,
                "object": "model",
                "created": 0,
                "owned_by": "router-alias"
            })
    return {"object": "list", "data": models_list}


@app.get("/api/keys")
async def get_keys():
    keys_with_quota = []
    for key_info in config.API_KEYS:
        key_data = key_info.copy()
        quota = config.get_quota(key_info["id"])
        if quota:
            key_data["quota"] = quota
        keys_with_quota.append(key_data)
    return {"keys": keys_with_quota}


@app.post("/api/keys")
async def add_key(req: AddKeyRequest):
    new_key = config.add_api_key(req.key, req.name)
    return {"success": True, "key": new_key}


@app.post("/api/keys/test-all")
async def test_all_keys(request: Request = None):
    """一键测试所有 Key 的额度 — 发轻量请求获取实时配额"""
    import httpx
    import asyncio as _asyncio

    if not config.API_KEYS:
        return {"success": False, "error": "没有可用的 API Key"}

    # 支持 POST body 中传入 model 字段指定要测试的模型
    model_override = None
    if request:
        try:
            body = await request.json()
            model_override = body.get("model", "").strip()
        except Exception:
            pass

    if model_override:
        test_model_id = model_override
    else:
        # 默认用第一个 chat 模型
        models_by_cat = config.get_models_by_category()
        chat_models = models_by_cat.get("chat", [])
        test_model_id = chat_models[0]["model_id"] if chat_models else "deepseek-v3"

    concurrency = 3

    async def test_one(key: dict) -> dict:
        try:
            url = f"{config.BASE_URL}/chat/completions"
            test_data = {
                "model": test_model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1
            }
            headers = {
                "Authorization": f"Bearer {key['key']}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=test_data, headers=headers, timeout=15)

            daily_limit = resp.headers.get("modelscope-ratelimit-requests-limit")
            daily_remaining = resp.headers.get("modelscope-ratelimit-requests-remaining")
            model_limit = resp.headers.get("modelscope-ratelimit-model-requests-limit")
            model_remaining = resp.headers.get("modelscope-ratelimit-model-requests-remaining")

            quota = {}
            if daily_limit: quota["daily_limit"] = int(daily_limit)
            if daily_remaining: quota["daily_remaining"] = int(daily_remaining)
            if model_limit: quota["model_limit"] = int(model_limit)
            if model_remaining: quota["model_remaining"] = int(model_remaining)

            if quota:
                config.update_quota(key["id"], quota)
                return {
                    "key_id": key["id"],
                    "key_name": key["name"],
                    "success": True,
                    "quota": quota,
                    "status": resp.status_code
                }
            else:
                return {
                    "key_id": key["id"],
                    "key_name": key["name"],
                    "success": resp.status_code < 400,
                    "status": resp.status_code,
                    "quota": None,
                    "note": "响应头无额度信息"
                }
        except Exception as e:
            return {
                "key_id": key["id"],
                "key_name": key["name"],
                "success": False,
                "error": str(e)
            }

    sem = _asyncio.Semaphore(concurrency)
    async def worker(key):
        async with sem:
            return await test_one(key)

    results = await _asyncio.gather(*[worker(key) for key in config.API_KEYS])

    ok = sum(1 for r in results if r.get("success"))
    return {
        "success": True,
        "total": len(results),
        "ok": ok,
        "fail": len(results) - ok,
        "results": results
    }


@app.delete("/api/keys/{key_id}")
async def delete_key(key_id: str):
    success = config.delete_api_key(key_id)
    return {"success": success}


@app.get("/api/models")
async def get_models():
    return {
        "models": config.MODELS,
        "categories": config.MODEL_CATEGORIES,
        "models_by_category": config.get_models_by_category()
    }


@app.post("/api/models")
async def add_model(req: AddModelRequest):
    new_model = config.add_model(req.name, req.model_id, req.category)
    return {"success": True, "model": new_model}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    success = config.delete_model(model_id)
    return {"success": success}


@app.post("/api/models/{model_id}/move")
async def move_model(model_id: str, req: MoveModelRequest):
    success = config.move_model(model_id, req.direction)
    return {"success": success}


@app.post("/api/models/{model_id}/test")
async def test_model(model_id: str):
    """测试模型是否可用 - 多 Key 轮换 + 重试，避免因上游偶发空响应误报"""
    import httpx
    from settings import config
    import json as _json
    import asyncio as _asyncio

    # 找到模型
    target_model = None
    for m in config.MODELS:
        if m.get("id") == model_id or m.get("name") == model_id:
            target_model = m
            break

    if not target_model:
        return {"success": False, "error": "模型不存在"}

    if not config.API_KEYS:
        return {"success": False, "error": "没有可用的 API Key"}

    category = target_model.get("category", "chat")
    max_retries = 2  # 每个 key 最多重试 2 次（含首次）
    last_error = None

    # 遍历所有 Key，直到有一个成功
    for key in config.API_KEYS:
        for attempt in range(max_retries):
            try:
                if category in ("text2img", "img2img"):
                    url = f"{config.BASE_URL}/images/generations"
                    test_data = {
                        "model": target_model["model_id"],
                        "prompt": "test"
                    }
                    headers = {
                        "Authorization": f"Bearer {key['key']}",
                        "Content-Type": "application/json",
                        "X-ModelScope-Async-Mode": "true"
                    }
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(url, content=_json.dumps(test_data), headers=headers, timeout=20)
                        if resp.status_code == 429:
                            last_error = f"[{key['name']}] 被限流(429)"
                            await _asyncio.sleep(2)
                            continue
                        if resp.status_code >= 400:
                            last_error = f"[{key['name']}] HTTP {resp.status_code}"
                            break  # 非限流 4xx/5xx 不重试，换 key
                        result = resp.json()
                        if not isinstance(result, dict):
                            last_error = f"[{key['name']}] 返回非JSON格式"
                            break
                        task_id = result.get("task_id", "")
                        if task_id:
                            return {"success": True, "task_id": task_id, "model": target_model["model_id"], "key_name": key["name"]}
                        if result.get("image_url") or result.get("choices"):
                            return {"success": True, "model": target_model["model_id"], "key_name": key["name"]}
                        # 空壳响应，重试
                        last_error = f"[{key['name']}] 返回空响应(choices=null)"
                        if attempt < max_retries - 1:
                            await _asyncio.sleep(1)
                        continue
                else:
                    url = f"{config.BASE_URL}/chat/completions"
                    test_data = {
                        "model": target_model["model_id"],
                        "messages": [{"role": "user", "content": "hi"}]
                    }
                    headers = {
                        "Authorization": f"Bearer {key['key']}",
                        "Content-Type": "application/json"
                    }
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(url, content=_json.dumps(test_data), headers=headers, timeout=20)
                        if resp.status_code == 429:
                            last_error = f"[{key['name']}] 被限流(429)"
                            await _asyncio.sleep(2)
                            continue
                        if resp.status_code >= 400:
                            last_error = f"[{key['name']}] HTTP {resp.status_code}"
                            break
                        result = resp.json()
                        choices = result.get("choices")
                        if isinstance(choices, list) and choices:
                            msg = choices[0].get("message", {})
                            content = msg.get("content", "")
                            if content and content.strip():
                                return {"success": True, "model": target_model["model_id"], "content": content[:80], "key_name": key["name"]}
                        # 空壳响应，重试
                        last_error = f"[{key['name']}] 返回空响应(choices=null)"
                        if attempt < max_retries - 1:
                            await _asyncio.sleep(1)
                        continue

            except Exception as e:
                last_error = f"[{key['name']}] {e}"
                if attempt < max_retries - 1:
                    await _asyncio.sleep(1)
                    continue
                break  # 这个 key 的网络/超时异常，换 key

    return {"success": False, "error": last_error or "所有 Key 均失败"}


@app.get("/api/examples")
async def get_examples():
    examples = {
        "chat": {
            "name": "对话 (chat)",
            "description": "文本对话模型，只需要传 model='chat'",
            "curl": """curl -X POST http://localhost:2166/v1/chat/completions \\
  -H \"Content-Type: application/json\" \\
  -H \"Authorization: Bearer multi-proxy-2025-2000q\" \\
  -d '{
    \"model\": \"chat\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"你好\"}
    ]
  }'""",
            "python": """import openai

client = openai.OpenAI(
    base_url=\"http://localhost:2166/v1\",
    api_key=\"multi-proxy-2025-2000q\"
)

response = client.chat.completions.create(
    model=\"chat\",
    messages=[
        {\"role\": \"user\", \"content\": \"你好\"}
    ]
)

print(response.choices[0].message.content)""",
            "openai": {
                "base_url": "http://localhost:2166/v1",
                "api_key": "multi-proxy-2025-2000q",
                "model": "chat"
            }
        },
        "vision": {
            "name": "视觉理解 (vision)",
            "description": "视觉理解模型，支持单图或多图，只需要传 model='vision'",
            "curl": """curl -X POST http://localhost:2166/v1/chat/completions \\
  -H \"Content-Type: application/json\" \\
  -H \"Authorization: Bearer multi-proxy-2025-2000q\" \\
  -d '{
    \"model\": \"vision\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"这张图片里有什么？\"},
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"https://qcloud.dpfile.com/pc/d6A1POwDkj8vKTNgbAZswnAaIM2fuXnejIO0X7lJQb9NIYslSlGEPeQVyA4hZRCP.jpg\"}}
        ]
      }
    ]
  }'""",
            "python": """import openai

client = openai.OpenAI(
    base_url=\"http://localhost:2166/v1\",
    api_key=\"multi-proxy-2025-2000q\"
)

response = client.chat.completions.create(
    model=\"vision\",
    messages=[
        {
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"这张图片里有什么？\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"https://qcloud.dpfile.com/pc/d6A1POwDkj8vKTNgbAZswnAaIM2fuXnejIO0X7lJQb9NIYslSlGEPeQVyA4hZRCP.jpg\"}}
            ]
        }
    ]
)
print(response.choices[0].message.content)""",
            "openai": {
                "base_url": "http://localhost:2166/v1",
                "api_key": "multi-proxy-2025-2000q",
                "model": "vision"
            }
        },
        "txt2img": {
            "name": "文生图 (txt2img)",
            "description": "文本生成图片，只需要传 model='txt2img'",
            "curl": """curl -X POST http://localhost:2166/v1/chat/completions \\
  -H \"Content-Type: application/json\" \\
  -H \"Authorization: Bearer multi-proxy-2025-2000q\" \\
  -d '{
    \"model\": \"txt2img\",
    \"messages\": [{\"role\": \"user\", \"content\": \"一只可爱的猫咪，高清，柔和光线\"}]
  }'""",
            "python": """import openai

client = openai.OpenAI(
    base_url=\"http://localhost:2166/v1\",
    api_key=\"multi-proxy-2025-2000q\"
)

response = client.chat.completions.create(
    model=\"txt2img\",
    messages=[{\"role\": \"user\", \"content\": \"一只可爱的猫咪，高清，柔和光线\"}]
)

print(f\"图片链接: {response.choices[0].message.content}\")
print(f\"图片链接 (直接访问): {response.image_url}\")
print(f\"图片链接 (数组): {response.images[0]}\")""",
            "openai": {
                "base_url": "http://localhost:2166/v1",
                "api_key": "multi-proxy-2025-2000q",
                "model": "txt2img",
                "note": "技术实现：采用 ModelScope 异步模式（X-ModelScope-Async-Mode: true），优先处理非空 task_id 并轮询任务状态（最多 30 次，每 2 秒一次），如果上游直接返回图片链接也会直接提取并返回，再从 output_images 数组中提取图片链接"
            }
        },
        "img2img": {
            "name": "图生图 (img2img)",
            "description": "图片生成图片，当前会提取首张输入图片并转为上游要求的单个 image_url 字符串，只需要传 model='img2img'",
            "curl": """curl -X POST http://localhost:2166/v1/chat/completions \\
  -H \"Content-Type: application/json\" \\
  -H \"Authorization: Bearer multi-proxy-2025-2000q\" \\
  -d '{
    \"model\": \"img2img\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"text\", \"text\": \"优化这张图片，让它更清晰，颜色更自然\"},
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"https://qcloud.dpfile.com/pc/d6A1POwDkj8vKTNgbAZswnAaIM2fuXnejIO0X7lJQb9NIYslSlGEPeQVyA4hZRCP.jpg\"}}
        ]
      }
    ]
  }'""",
            "python": """import openai

client = openai.OpenAI(
    base_url=\"http://localhost:2166/v1\",
    api_key=\"multi-proxy-2025-2000q\"
)

response = client.chat.completions.create(
    model=\"img2img\",
    messages=[
        {
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"优化这张图片，让它更清晰，颜色更自然\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"https://qcloud.dpfile.com/pc/d6A1POwDkj8vKTNgbAZswnAaIM2fuXnejIO0X7lJQb9NIYslSlGEPeQVyA4hZRCP.jpg\"}}
            ]
        }
    ]
)

print(f\"图片链接: {response.choices[0].message.content}\")
print(f\"图片链接 (直接访问): {response.image_url}\")
print(f\"图片链接 (数组): {response.images[0]}\")""",
            "openai": {
                "base_url": "http://localhost:2166/v1",
                "api_key": "multi-proxy-2025-2000q",
                "model": "img2img"
            }
        }
    }
    return examples


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        requested_model = body.get("model", "chat")

        # 暂不支持流式响应，去掉 stream 标志强制非流式返回
        if body.get("stream"):
            body.pop("stream")
            import logging
            logging.getLogger("uvicorn").info("检测到 stream=true，当前暂不支持流式，已自动转为非流式返回")
        
        category_mapping = {
            "chat": "chat",
            "txt2img": "text2img",
            "img2img": "img2img",
            "vision": "vision"
        }
        
        target_category = category_mapping.get(requested_model)
        
        if target_category:
            models_by_cat = config.get_models_by_category()
            cat_models = models_by_cat.get(target_category, [])
            if cat_models:
                body["model"] = cat_models[0]["name"]
                model_name = cat_models[0]["name"]
            else:
                model_name = requested_model
        else:
            model_name = requested_model
        
        headers = dict(request.headers)
        headers.pop("Authorization", None)
        
        # 图片生成类请求需要更长超时（异步轮询最多 80×3=240秒）
        call_timeout = 180 if target_category in ("text2img", "img2img") else 60

        result, status, resp_headers = await api_client.call_model(
            model_name, body, headers, timeout=call_timeout
        )

        safe_response_headers = {
            k: v for k, v in resp_headers.items()
            if k.lower() not in {
                "content-length",
                "transfer-encoding",
                "connection",
                "date",
                "server",
                "content-encoding"
            }
        }

        return JSONResponse(content=result, status_code=status, headers=safe_response_headers)
        
    except Exception as e:
        return JSONResponse(
            content={"error": {"message": str(e)}},
            status_code=500
        )


@app.on_event("startup")
async def startup_event():
    print("Server is running on port 2166...")
    print("Web UI: http://localhost:2166")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2166, log_level="info")
