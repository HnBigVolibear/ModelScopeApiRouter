import os
import json
import time
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Optional

class Settings:
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent
        self.BUNDLED_DATA_DIR = self.BASE_DIR / "default_data"
        self.DATA_DIR = Path(os.getenv("ROUTER_DATA_DIR", str(self.BASE_DIR / "router_data")))
        self.STATS_FILE = self.DATA_DIR / "model_stats.json"
        self.CONFIG_FILE = self.DATA_DIR / "config.json"
        self.BUNDLED_CONFIG_FILE = self.BUNDLED_DATA_DIR / "config.json"
        self.ENV_FILE = self.BASE_DIR / ".env"
        self.API_KEYS_FILE = self.DATA_DIR / "api_keys.json"
        self.BUNDLED_API_KEYS_FILE = self.BUNDLED_DATA_DIR / "api_keys.json"
        self.QUOTA_FILE = self.DATA_DIR / "quota_info.json"
        self._quota_lock = threading.Lock()
        self._api_keys_lock = threading.Lock()
        self._models_lock = threading.Lock()
        
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_data_file(self.CONFIG_FILE, self.BUNDLED_CONFIG_FILE, default_content="[]")
        self._ensure_data_file(self.API_KEYS_FILE, self.BUNDLED_API_KEYS_FILE, default_content="[]")
        
        self._load_env()
        self.BASE_URL = os.getenv("MS_BASE_URL", "https://api-inference.modelscope.cn/v1")
        self.PORT = 2166
        self.ROUTER_ALIAS = "modelscope-router"
        
        # 模型分类
        self.MODEL_CATEGORIES = {
            "chat": "对话",
            "vision": "视觉理解",
            "txt2img": "文生图",
            "img2img": "图生图"
        }
        
        self.MODELS = self._load_models()
        self.API_KEYS = self._load_api_keys()
        self.QUOTA_INFO = self._load_quota_info()

    def _load_env(self):
        """简单的 .env 解析器，避免引入 python-dotenv 依赖"""
        if not self.ENV_FILE.exists():
            return
        with open(self.ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

    def _ensure_data_file(self, target_path: Path, source_path: Path, default_content: str = ""):
        if target_path.exists():
            return
        if source_path.exists() and source_path.resolve() != target_path.resolve():
            shutil.copy2(source_path, target_path)
            return
        if default_content != "":
            self._atomic_write_text(target_path, default_content)

    def _atomic_write_text(self, path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = path.with_suffix(f"{path.suffix}.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(content)
        tmp_file.replace(path)

    def _atomic_write_json(self, path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = path.with_suffix(f"{path.suffix}.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_file.replace(path)

    def _normalize_category(self, category: str) -> str:
        return "txt2img" if category == "text2img" else category

    def _normalize_model_categories(self, models: List[Dict]) -> List[Dict]:
        changed = False
        for model in models:
            if isinstance(model, dict) and model.get("category") == "text2img":
                model["category"] = "txt2img"
                changed = True
        if changed:
            self._save_models_data(models)
        return models

    def _save_api_keys_data(self, keys: List[Dict]):
        self._atomic_write_json(self.API_KEYS_FILE, keys)

    def _save_models_data(self, models: List[Dict]):
        self._atomic_write_json(self.CONFIG_FILE, models)

    def _save_quota_data(self, quota_info: Dict):
        self._atomic_write_json(self.QUOTA_FILE, quota_info)

    def _load_models(self) -> List[Dict]:
        if not self.CONFIG_FILE.exists():
            return []
        with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
            models = json.load(f)
        return self._normalize_model_categories(models if isinstance(models, list) else [])

    def _load_api_keys(self) -> List[Dict]:
        if not self.API_KEYS_FILE.exists():
            old_key = os.getenv("MS_API_KEY", "")
            if old_key:
                keys = [{"id": "default", "key": old_key, "name": "默认 Key"}]
                self._save_api_keys(keys)
                return keys
            return []
        with open(self.API_KEYS_FILE, "r", encoding="utf-8") as f:
            keys = json.load(f)
        return keys if isinstance(keys, list) else []

    def _save_api_keys(self, keys: List[Dict]):
        with self._api_keys_lock:
            self._save_api_keys_data(keys)

    def _load_quota_info(self) -> Dict:
        if not self.QUOTA_FILE.exists():
            return {}
        with open(self.QUOTA_FILE, "r", encoding="utf-8") as f:
            quota_info = json.load(f)
        return quota_info if isinstance(quota_info, dict) else {}

    def _save_quota_info(self, quota_info: Dict):
        self._save_quota_data(quota_info)

    def update_quota(self, key_id: str, quota_data: Dict):
        with self._quota_lock:
            existing = self.QUOTA_INFO.get(key_id, {}).copy()
            existing.update(quota_data)
            existing["updated_at"] = time.time()
            self.QUOTA_INFO[key_id] = existing
            self._save_quota_info(self.QUOTA_INFO)

    def get_quota(self, key_id: str) -> Optional[Dict]:
        """获取某个 Key 的额度信息"""
        return self.QUOTA_INFO.get(key_id)

    def add_api_key(self, key: str, name: str = "") -> Dict:
        """添加新的 API key"""
        import uuid
        new_key = {
            "id": str(uuid.uuid4()),
            "key": key,
            "name": name or f"Key {len(self.API_KEYS) + 1}"
        }
        self.API_KEYS.append(new_key)
        self._save_api_keys(self.API_KEYS)
        return new_key

    def delete_api_key(self, key_id: str) -> bool:
        """删除 API key"""
        original_len = len(self.API_KEYS)
        self.API_KEYS = [k for k in self.API_KEYS if k["id"] != key_id]
        if key_id in self.QUOTA_INFO:
            with self._quota_lock:
                if key_id in self.QUOTA_INFO:
                    del self.QUOTA_INFO[key_id]
                    self._save_quota_info(self.QUOTA_INFO)
        if len(self.API_KEYS) < original_len:
            self._save_api_keys(self.API_KEYS)
            return True
        return False

    def _save_models(self):
        with self._models_lock:
            self._save_models_data(self.MODELS)

    def add_model(self, name: str, model_id: str, category: str = "chat") -> Dict:
        import uuid
        category = self._normalize_category(category)
        new_model = {
            "id": str(uuid.uuid4()),
            "name": name,
            "model_id": model_id,
            "category": category,
            "order": len([m for m in self.MODELS if m.get("category") == category])
        }
        self.MODELS.append(new_model)
        self._save_models()
        return new_model

    def delete_model(self, model_id: str, category: str) -> bool:
        category = self._normalize_category(category)
        original_len = len(self.MODELS)
        
        model_to_delete = None
        for m in self.MODELS:
            if m.get("id") == model_id and m.get("category") == category:
                model_to_delete = m
                break
            if m.get("name") == model_id and m.get("category") == category:
                model_to_delete = m
                break
        
        if not model_to_delete:
            return False
        
        self.MODELS = [m for m in self.MODELS if m.get("id") != model_to_delete.get("id")]
        
        cat = model_to_delete.get("category")
        same_cat_models = [m for m in self.MODELS if m.get("category") == cat]
        same_cat_models.sort(key=lambda x: x.get("order", 0))
        for i, m in enumerate(same_cat_models):
            m["order"] = i
        
        self._save_models()
        return len(self.MODELS) < original_len

    def move_model(self, model_id: str, direction: str) -> bool:
        """移动模型排序（up/down）"""
        # 找到要移动的模型
        model_idx = None
        target_model = None
        for i, m in enumerate(self.MODELS):
            if m.get("id") == model_id or m.get("name") == model_id:
                model_idx = i
                target_model = m
                break
        
        if not target_model:
            return False
        
        cat = target_model.get("category")
        # 获取同分类的所有模型
        same_cat_models = [m for m in self.MODELS if m.get("category") == cat]
        same_cat_models.sort(key=lambda x: x.get("order", 0))
        
        # 找到目标模型在同分类中的位置
        current_pos = None
        for i, m in enumerate(same_cat_models):
            if m.get("id") == model_id or m.get("name") == model_id:
                current_pos = i
                break
        
        if current_pos is None:
            return False
        
        # 计算新位置
        new_pos = current_pos
        if direction == "up" and current_pos > 0:
            new_pos = current_pos - 1
        elif direction == "down" and current_pos < len(same_cat_models) - 1:
            new_pos = current_pos + 1
        else:
            return False
        
        # 交换位置
        same_cat_models[current_pos], same_cat_models[new_pos] = same_cat_models[new_pos], same_cat_models[current_pos]
        
        # 更新 order
        for i, m in enumerate(same_cat_models):
            m["order"] = i
        
        self._save_models()
        return True

    def get_models_by_category(self) -> Dict[str, List[Dict]]:
        """按分类获取模型（按 order 排序）"""
        result = {}
        for cat in self.MODEL_CATEGORIES:
            result[cat] = []
        for model in self.MODELS:
            cat = self._normalize_category(model.get("category", "chat"))
            if cat not in result:
                cat = "chat"
            result[cat].append(model)
        
        # 按 order 排序每个分类
        for cat in result:
            result[cat].sort(key=lambda x: x.get("order", 0))
        
        return result

# 单例模式
config = Settings()
