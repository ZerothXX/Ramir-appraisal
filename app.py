# -*- coding: utf-8 -*-
"""Web 服务：托管前端页面 + 图片鉴定接口（连接 test.py 的推理管线）。

启动方式：PyCharm 直接运行本文件（或 python app.py），
浏览器打开 http://127.0.0.1:8000/ 即可看到封面页。

结构：
* 静态前端位于 web/（index.html 封面 / interact.html 选图 / thinking.html 思考中 / result.html 结果）；
* POST /api/predict  接收上传图片 -> 复用 test.py 的 predict_image 完整推理管线 -> 返回鉴定文本；
* GET  /api/status   返回模型预热状态（启动时后台线程预加载模型，避免首个请求等太久）；
* 模型是全局单例（test.py 内部），本文件另加一把锁串行化推理 —— Qwen2.5-VL 的
  generate 非线程安全，Flask 默认多线程处理请求，并发时必须排队。
"""

import os
import threading
from io import BytesIO

from flask import Flask, jsonify, request
from PIL import Image

import test  # 复用 test.py 的 load_model_once / predict_image（推理管线单一来源，见 test.py）
from utils import get_logger

logger = get_logger("web")

# ===== 服务配置（调端口/大小限制改这里） =====
HOST = "127.0.0.1"
PORT = 8000
MAX_UPLOAD_MB = 15
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

app = Flask(__name__, static_folder="web", static_url_path="")

# 模型单例 + generate 非线程安全：所有推理请求排队执行
_inference_lock = threading.Lock()
_model_ready = False
_model_error = None


# ===== 模型预热 =====

def _warmup_model():
    """后台线程预加载模型（基座 + LoRA），完成后置 ready；失败记录原因供 /api/status 返回。"""
    global _model_ready, _model_error
    try:
        test.load_model_once()
        _model_ready = True
        logger.info("模型预热完成，可以开始鉴定")
    except Exception as e:  # noqa: BLE001 预加载失败不应让服务进程挂掉，失败原因下发给前端
        _model_error = f"{type(e).__name__}: {e}"
        logger.exception("模型后台预热失败")


def _start_preload():
    """幂等：只启动一次后台预热线程（重复调用 / 多次运行都安全）。"""
    if _model_ready or _model_error:
        return
    threading.Thread(target=_warmup_model, daemon=True, name="model-preload").start()


# ===== API =====

@app.get("/")
def index():
    """封面页：static_folder 的根路径不会自动映射到 index.html，需显式指定。"""
    return app.send_static_file("index.html")


@app.get("/api/status")
def api_status():
    return jsonify({"ready": _model_ready, "error": _model_error})


@app.post("/api/predict")
def api_predict():
    """接收图片文件（表单字段名 file）-> test.predict_image -> {result: 鉴定文本}。"""
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "未收到图片文件（表单字段名应为 file）"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"不支持的图片格式 {ext or '(无扩展名)'}，支持 {sorted(ALLOWED_EXT)}"}), 400

    data = f.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify({"error": f"图片过大（最大 {MAX_UPLOAD_MB}MB）"}), 400
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image.load()  # 提前解码，把"不是合法图片"的报错收敛成统一提示
    except Exception:  # noqa: BLE001
        return jsonify({"error": "无法解析该图片，请换一张试试"}), 400

    with _inference_lock:
        try:
            result = test.predict_image(image)
        except Exception:  # noqa: BLE001 推理失败返回可读错误，不把栈打到前端
            logger.exception("推理失败")
            return jsonify({"error": "推理失败，请稍后重试（详见后端日志）"}), 500

    if not result:
        result = "（模型未输出结果）"
    return jsonify({"result": result})


if __name__ == "__main__":
    _start_preload()
    logger.info("Web 服务启动: http://%s:%d/（后台正在预热模型，首次鉴定稍候即可使用）",
                HOST, PORT)
    # debug=False：避免 reloader 重复起进程导致模型被加载两份；多线程并发由锁串行化
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
