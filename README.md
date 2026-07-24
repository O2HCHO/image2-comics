# 图生图批量上色工具

这是一个基于 CustomTkinter 的桌面工具，用于逐张调用图生图 API，为漫画线稿或灰阶稿进行自动化上色。网络请求和文件读写全部在后台线程执行，界面不会因请求阻塞；停止按钮会等待正在进行的请求结束后安全停止。

## 创建环境

当前已经处于 Anaconda Python 3.12 环境时，可以在项目根目录执行：

```powershell
conda env create -f environment.yml
conda activate image2-comics
python app.py
```

若希望直接在当前已激活的 Python 3.12 Conda 环境安装：

```powershell
conda install -c conda-forge pip
pip install "customtkinter>=5.2,<6" "requests>=2.32,<3"
python app.py
```

## 使用方式

1. 选择包含 `.png`、`.jpg` 或 `.jpeg` 图片的输入文件夹，以及输出文件夹。
2. 填写服务端的 API URL、API Key 和上色 Prompt，选择重绘幅度。API URL 必须是服务商文档中标明的图生图请求路径，不能填写文档站或官网首页。例如 `https://doc.example.com` 是文档网页，不是可调用的 API 接口。
3. 点击“开始批量处理”。文件会按自然文件名顺序处理，例如 `page_2.png` 会排在 `page_10.png` 前。
4. 输出会保存为 `colored_原文件名`。每张图片间隔 2 秒，避免触发接口限流。

### 测试 OpenAI 兼容中转站

填写 API URL 和 API Key 后，先点击“测试连接并获取模型”。程序会在同一域名内安全地依次检测常见的模型列表路径：若 API URL 包含 `/v1/`，优先检测同一路径前缀下的 `/v1/models`，例如：

```text
https://image.xcode.best/v1/images/generations
https://image.xcode.best/v1/models
```

### 使用 gpt-image-2 批量上色

当模型列表中有 `gpt-image-2` 时，选择“OpenAI 图像编辑（multipart）”模式，并将 API URL 精确填写为：

```text
https://xcode.best/v1/images/edits
```

不要填写 `https://xcode.best/v1/images`，它不是可 POST 的接口路径；`/v1/images/generations` 通常只用于文生图。图像编辑模式会以 `multipart/form-data` 上传每张原图，并提交 `model`、`prompt` 与 `image` 字段。标准 OpenAI 图像编辑接口不接受 `denoising_strength`，因此该模式会忽略界面中的重绘幅度参数。

### 超时、设置与输出比例

- 单张图片请求最长等待 300 秒，适用于较慢的图片生成任务。
- API URL、模型、提示词、接口模式和输出比例会保存在项目目录的 `local_settings.json` 中。勾选“保存 API Key（本机明文）”后才会保存 API Key；该文件已被 `.gitignore` 排除，不能提交到 GitHub。
- 输出比例可选择“保持原始比例”、`1:1`、`3:4` 或 `16:9`。后三项会扩展透明 PNG 画布以实现固定比例，不会裁切、缩放或删除原图内容。

### 只重新处理失败图片

程序启动时会自动读取项目内的 `inputs` 文件夹；选择其他输入文件夹后会自动刷新“待处理图片”列表。列表中的图片默认全选，可使用“清空”后仅勾选生成失败的图片，再点击“开始批量处理”。工作线程只会发送已勾选的图片。

当窗口高度不足时，整个界面支持鼠标滚轮滚动，图片选择列表也支持独立滚动。

成功后，服务端返回的 `data[].id` 会显示在“模型”下拉框。开始处理前必须选择一个模型。连接失败时，日志会显示每个检测地址的 HTTP 状态码和服务端错误信息。

## API 协议

程序请求头为：

```http
Authorization: Bearer <API Key>
Content-Type: application/json
```

请求 JSON 结构为：

```json
{
  "model": "从模型列表选择的模型 ID",
  "image": "<原始图片的Base64字符串>",
  "prompt": "漫画上色提示词",
  "denoising_strength": 0.55
}
```

程序可读取以下响应中的首张 Base64 图像：`image`、`images[0]`、`data.image`、`data.images[0]`。也兼容 `data:image/png;base64,...` 格式以及对象中的 `b64_json` 字段。

不同服务的字段名称和路径可能不同。若目标服务不符合这个协议，只需调整 `app.py` 中的 `build_payload()`（请求格式）和 `extract_image_base64()`（响应格式）两个函数。程序若收到 HTML 网页，会在日志中显示最终 URL，通常意味着填写了文档页、官网页或错误路径。
