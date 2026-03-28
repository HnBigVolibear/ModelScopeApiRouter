import httpx
import logging
import json
from typing import Tuple, Dict, List, Optional
from .settings import config

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self):
        self.client = None
    
    def _update_quota_from_headers(self, key_id: str, headers: dict):
        """从 ModelScope 响应头提取额度信息"""
        quota = {}
        
        for h in headers:
            if h.lower() == "modelscope-ratelimit-tpm":
                quota["tpm"] = int(headers[h])
            elif h.lower() == "modelscope-ratelimit-rpm":
                quota["rpm"] = int(headers[h])
            elif h.lower() == "modelscope-ratelimit-model-limit":
                quota["model_limit"] = int(headers[h])
            elif h.lower() == "modelscope-ratelimit-daily-remaining":
                quota["daily_remaining"] = int(headers[h])
            elif h.lower() == "modelscope-ratelimit-daily-limit":
                quota["daily_limit"] = int(headers[h])
        
        config.update_quota(key_id, quota)
        return quota
    
    def is_key_exhausted(self, key_id: str, quota: Dict) -> bool:
        """判断 Key 是否完全不可用（额度全用完或报错失败）"""
        model_exhausted = (
            quota.get("model_limit", 0) > 0 and 
            quota.get("daily_remaining", 0) <= 0
        )
        return model_exhausted
    
    def should_try_next_key(self, quota: Dict) -> bool:
        """判断是否应该尝试下一个 Key（当前 Key 失败或模型额度用完，但每日还有额度）"""
        model_quota_exhausted = (
            quota.get("model_limit", 0) > 0 and 
            quota.get("daily_remaining", 0) < quota.get("model_limit", 0) and
            quota.get("daily_remaining", 0) > 0
        )
        return model_quota_exhausted
    
    async def call_model(self, model_name: str, data: dict, headers: dict, timeout: int) -> Tuple[dict, int, dict]:
        """
        新的核心切换逻辑：
        1. 先确定请求的分类（根据 model_name 或默认 chat）
        2. 获取该分类的模型列表（按 order 排序）
        3. 对每个模型，尝试所有可用的 Key
        4. Key 失败 → 换 Key；当前模型所有 Key 全废 → 换模型
        """
        
        # 步骤 1：确定分类
        target_category = "chat"
        for model in config.MODELS:
            if model.get("name") == model_name:
                target_category = model.get("category", "chat")
                break
        
        # 步骤 2：获取该分类的模型列表
        models_by_category = config.get_models_by_category()
        category_models = models_by_category.get(target_category, [])
        
        if not category_models:
            category_models = config.MODELS
        
        all_keys = config.API_KEYS
        
        if not all_keys:
            raise Exception("没有可用的 API Key")
        
        logger.info(f"请求分类: {target_category}, 模型数: {len(category_models)}, Key 数: {len(all_keys)}")
        
        # 步骤 3 & 4：逐个模型尝试
        for model in category_models:
            logger.info(f"尝试模型: {model['name']} ({model['model_id']})")
            
            exhausted_keys_for_this_model = set()
            
            for key in all_keys:
                if key["id"] in exhausted_keys_for_this_model:
                    continue
                
                try:
                    # 正确的 URL 是 /chat/completions，model 放请求体里！
                    url = f"{config.BASE_URL}/chat/completions"
                    
                    headers_copy = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key['key']}"
                    }
                    
                    # 请求体里设置 model
                    request_data = data.copy()
                    request_data["model"] = model["model_id"]
                    
                    logger.info(f"  使用 Key: {key['name']}")
                    
                    json_data = json.dumps(request_data)
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            url,
                            content=json_data,
                            headers=headers_copy,
                            timeout=timeout
                        )
                    
                    quota = self._update_quota_from_headers(key["id"], response.headers)
                    
                    if response.status_code >= 400:
                        try:
                            error_text = response.text
                        except:
                            error_text = str(response)
                        logger.warning(f"  Key {key['name']} HTTP {response.status_code} 错误: {error_text[:300]}")
                        exhausted_keys_for_this_model.add(key["id"])
                        continue
                    
                    if self.should_try_next_key(quota):
                        logger.info(f"  Key {key['name']} 模型额度用完，换 Key")
                        exhausted_keys_for_this_model.add(key["id"])
                        continue
                    
                    if self.is_key_exhausted(key["id"], quota):
                        logger.info(f"  Key {key['name']} 完全耗尽")
                        exhausted_keys_for_this_model.add(key["id"])
                        continue
                    
                    logger.info(f"✅ 成功！模型: {model['name']}, Key: {key['name']}")
                    return response.json(), response.status_code, dict(response.headers)
                    
                except Exception as e:
                    logger.error(f"  Key {key['name']} 调用异常: {e}")
                    exhausted_keys_for_this_model.add(key["id"])
                    continue
            
            logger.warning(f"⚠️  模型 {model['name']} 所有 Key 都失败，换下一个模型")
        
        raise Exception("所有模型和 Key 都调用失败，请检查配置")

api_client = APIClient()
