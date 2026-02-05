# 导入Gradio库，用于快速构建Web界面
import gradio as gr
# 导入操作系统相关模块，用于读取环境变量
import os
# 导入系统相关模块，用于日志配置
import sys

# ====================== 1. 核心功能模块：中医问答引擎初始化 ======================
# 导入日志模块，用于输出运行日志
import logging
# 导入类型注解模块，用于函数返回值类型提示
from typing import Optional

# 导入llama-index核心模块，用于构建向量索引和问答引擎
from llama_index.core import (
    PromptTemplate,  # 提示词模板类，用于自定义中医问答提示词
    Settings,  # 全局设置类，配置LLM和嵌入模型
    StorageContext,  # 存储上下文类，用于加载/保存向量索引
    load_index_from_storage,  # 从存储加载向量索引的函数
    VectorStoreIndex,  # 向量存储索引类，核心检索组件
    SimpleDirectoryReader,  # 目录文档读取器，用于加载中医知识库文档
)
from llama_index.core.node_parser import SentenceSplitter  # 文本分割器，用于切分长文本
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager  # 回调管理器，用于调试

# 导入通义千问相关模块（阿里云DashScope）
from llama_index.llms.dashscope import (
    DashScope,  # 通义千问LLM封装类
    DashScopeGenerationModels  # 通义千问模型枚举（如QWEN_MAX）
)
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,  # 通义千问嵌入模型封装类
    DashScopeTextEmbeddingModels  # 嵌入模型枚举
)

# ---------------------- 1.1 初始化日志和大模型配置 ----------------------
# 配置日志输出：输出到标准输出流，日志级别为INFO
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# 为日志器添加额外的处理器，确保日志正常输出
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 从系统环境变量中读取通义千问API密钥（需提前配置环境变量DASHSCOPE_API_KEY）
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
# 校验API密钥是否存在，不存在则抛出异常
if not dashscope_api_key:
    raise ValueError("未找到DASHSCOPE_API_KEY环境变量！请检查环境变量配置是否正确。")

# 配置全局LLM模型（通义千问QWEN_MAX）
Settings.llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX,  # 使用QWEN_MAX大模型
    api_key=dashscope_api_key,  # 传入API密钥
    temperature=0.0,  # 温度值0.0，保证回答确定性
    max_tokens=2048  # 最大生成token数
)

# 配置全局嵌入模型（通义千问文本嵌入模型）
Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,  # 文本嵌入模型
    api_key=dashscope_api_key,  # 传入API密钥
)

# 配置调试回调管理器（用于调试，不影响核心功能）
llama_debug = LlamaDebugHandler(print_trace_on_end=False)
Settings.callback_manager = CallbackManager([llama_debug])


# ---------------------- 1.2 定义中医专属提示词模板 ----------------------
def get_tcm_prompts():
    """
    构建中医问答的提示词模板
    作用：约束大模型以专业中医医师的角度回答问题，确保回答符合中医辨证逻辑
    """
    qa_prompt_tmpl_str = (
        "上下文信息如下（中医典籍/诊疗指南）：\n"
        "---------------------\n"
        "{context_str}\n"  # 占位符：检索到的中医知识库上下文
        "---------------------\n"
        "请严格根据上下文，以专业中医医师的角度回答以下问题，回答需严谨、简洁，符合中医辨证逻辑：\n"
        "Query: {query_str}\n"  # 占位符：用户的症状查询问题
        "Answer: "  # 回答起始标记
    )
    # 返回构建好的提示词模板对象
    return PromptTemplate(qa_prompt_tmpl_str)


# ---------------------- 1.3 构建/加载中医知识库向量索引 ----------------------
def build_or_load_index(
        data_dir: str = "./data",  # 中医知识库文档存放目录（默认./data）
        persist_dir: str = "./doc_emb",  # 向量索引持久化存储目录（默认./doc_emb）
        chunk_size: int = 256  # 文本分割大小（256个字符/块）
) -> VectorStoreIndex:
    """
    构建或加载中医知识库的向量索引
    首次运行：从data_dir加载文档→分割文本→构建向量索引→保存到persist_dir
    非首次运行：直接从persist_dir加载已构建的索引（提升启动速度）
    """
    # 检查向量索引存储目录是否存在且非空
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"加载已存储的向量索引（{persist_dir}）...")
        # 从存储目录加载向量索引
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
    else:
        print(f"从{data_dir}加载中医文档...")
        # 加载指定目录下的所有txt格式中医文档
        documents = SimpleDirectoryReader(data_dir, required_exts=[".txt"]).load_data()
        # 初始化文本分割器（按256字符分割，适配嵌入模型）
        splitter = SentenceSplitter(chunk_size=chunk_size)
        # 从文档构建向量索引（自动完成文本分割→嵌入→入库）
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
        print(f"保存向量索引到{persist_dir}...")
        # 将向量索引持久化到本地目录，避免重复构建
        index.storage_context.persist(persist_dir=persist_dir)
    return index


# 初始化中医知识库向量索引（程序启动时执行）
tcm_index = build_or_load_index()

# ====================== 2. Web界面样式配置 ======================
# 自定义CSS样式，实现深色聊天框、绿色圆形箭头按钮等视觉效果
CUSTOM_CSS = """
.gradio-container .input_text label,
.gradio-container .input_text .label-wrap {
    background-color: transparent !important;  /* 透明背景，彻底移除紫色 */
    color: #4ECDC4 !important;                 /* 文字改为青绿色，和界面风格统一 */
    padding: 0 !important;                     /* 取消默认内边距，避免空白 */
}


/* 标题样式：红色居中显示 */
h1 {
    color: #4ECDC4;         /* 青绿色标题 */
    text-align: center;     /* 居中对齐 */
    font-size: 24px;        /* 字体大小 */
    margin-bottom: 10px;    /* 底部外边距 */
}

/* 副标题样式：灰色小字 */
h3 {
    color: #666666;         /* 灰色文字 */
    text-align: center;     /* 居中对齐 */
    font-size: 14px;        /* 小号字体 */
    font-weight: normal;    /* 取消加粗 */
    margin-bottom: 20px;    /* 底部外边距 */
}

/* 聊天框样式：深色背景+白色文字 */
.chatbot {
    border-radius: 15px !important;    /* 圆角边框 */
    background-color: #1a1b26 !important; /* 深色背景（接近截图样式） */
    color: #ffffff !important;         /* 白色文字 */
    height: 400px !important;          /* 固定高度400px */
    border: none !important;           /* 取消边框 */
}



/* 输入框样式：深色背景+紫色边框+右侧预留按钮空间 */
.input_text textarea {
    border-radius: 8px !important;     /* 圆角边框 */
    border: 1px solid #4ECDC4 !important; /* 青绿色边框 */
    font-size: 14px !important;        /* 字体大小 */
    background-color: #2c2d3e !important; /* 深色输入框背景 */
    color: #ffffff !important;         /* 白色文字 */
    padding: 10px 15px !important;     /* 内边距 */
    padding-right: 50px !important;    /* 右侧预留50px空间给按钮，避免遮挡 */
    width: 100% !important;            /* 宽度100% */
    box-sizing: border-box !important; /* 内边距不影响总宽度 */
}

/* 自定义提交按钮：绿色圆形+飞机样式向上箭头 */
.custom-submit-btn {
    position: absolute;                /* 绝对定位 */
    right: 12px;                       /* 距离右侧12px */
    bottom: 12px;                      /* 距离底部12px */
    /* 位置微调+旋转：向左上移2px + 旋转-90度（将右向飞机箭头转为向上） */
    transform: translate(-10px, -15px) rotate(-90deg);
    width: 40px;                       /* 按钮宽度40px */
    height: 40px;                      /* 按钮高度40px（圆形） */
    border-radius: 50% !important;     /* 圆角50%实现圆形 */
    background-color: #4ECDC4 !important; /* 青绿色背景 */
    color: white !important;           /* 白色箭头 */
    border: none !important;           /* 取消边框 */
    font-size: 20px !important;        /* 箭头字体大小 */
    display: flex !important;                     /* Flex布局，让箭头居中 */
    align-items: center ;               /* 垂直居中 */
    justify-content: center;           /* 水平居中 */
    cursor: pointer;                   /* 鼠标悬停显示手型 */
    z-index: 999 !important;           /* 提升层级，避免被输入框遮挡 */
}

/* 按钮悬停效果：加深青绿色 */
.custom-submit-btn:hover {
    background-color: #45B7B0 !important;
}

/* 输入框容器：相对定位，作为按钮绝对定位的参考 */
.textbox-container {
    position: relative;    /* 相对定位 */
    width: 100%;           /* 宽度100% */
    padding: 0 !important; /* 取消内边距 */
    margin: 0 !important;  /* 取消外边距 */
}

/* 修复Gradio默认表单控件样式，确保输入框右侧内边距生效 */
.textbox-container .form-control {
    padding-right: 50px !important;
}
"""


# ====================== 3. 核心交互函数：处理用户问答请求 ======================
def web_tcm_chat(message, history):
    """
    处理Web界面的用户问答请求（流式返回结果）
    参数说明：
        message: 当前用户输入的问题字符串
        history: 聊天历史记录（Gradio 6.x格式：[{"role": "user/assistant", "content": "内容"}]）
    返回：生成器，逐字返回回答结果（流式输出）
    """
    # 第一步：空输入校验 - 如果用户未输入内容直接提交
    if not message or message.strip() == "":
        # 复制历史记录（避免修改原数据）
        new_history = history.copy() if history else []
        # 添加助手提示信息
        new_history.append({"role": "assistant", "content": "请输入你的中医症状问题，我才能帮你辨证哦～"})
        # 返回提示信息（生成器形式）
        yield new_history
        return

    # 第二步：构建查询引擎（流式输出+相似度Top5检索）
    query_engine = tcm_index.as_query_engine(
        streaming=True,  # 开启流式输出，逐字返回回答
        similarity_top_k=5  # 检索相似度最高的5个文档片段
    )

    # 第三步：绑定中医专属提示词模板到查询引擎
    qa_prompt = get_tcm_prompts()
    query_engine.update_prompts({"text_qa_template": qa_prompt})

    # 第四步：执行查询，获取流式响应对象
    response = query_engine.query(message)

    # 第五步：构建新的聊天历史（保留原有记录+新增用户问题）
    new_history = history.copy() if history else []
    new_history.append({"role": "user", "content": message})  # 添加用户问题
    new_history.append({"role": "assistant", "content": ""})  # 初始化助手回答（空字符串）

    # 第六步：流式返回回答结果（逐token生成）
    for token in response.response_gen:
        # 将当前token追加到助手回答中
        new_history[-1]["content"] += token
        # 生成新的历史记录（实现前端实时刷新）
        yield new_history


# ====================== 4. Web界面构建 ======================
# 创建Gradio Blocks应用（高级布局模式）
with gr.Blocks() as demo:
    # 主标题：中医智能诊疗小助手（带熊猫emoji）
    gr.Markdown("# 🐼 中医智能诊疗小助手")
    # 副标题：提示用户输入症状问题
    gr.Markdown("### 💬 输入你的中医症状问题，我来帮你辨证～")

    # 聊天框组件：显示诊疗对话记录
    chatbot = gr.Chatbot(
        label="诊疗对话",  # 组件标签
        elem_classes="chatbot"  # 绑定自定义CSS类
    )

    # 输入框容器：用于放置输入框和自定义提交按钮
    with gr.Column(elem_classes="textbox-container"):
        # 文本输入框：用户输入症状问题
        msg = gr.Textbox(
            label="请输入症状（如：不耐疲劳，口燥、咽干可能是哪些证候？）",  # 输入框标签（示例提示）
            elem_classes="input_text",  # 绑定自定义CSS类
            placeholder="输入你的问题...",  # 占位提示文字
            lines=3  # 输入框高度（3行）
        )
        # 自定义提交按钮：飞机样式向上箭头（通过CSS旋转实现）
        submit_btn = gr.Button("➤", elem_classes="custom-submit-btn")

    # ---------------------- 4.1 绑定按钮点击事件 ----------------------
    # 点击提交按钮：执行问答函数→清空输入框
    submit_btn.click(
        fn=web_tcm_chat,  # 绑定的核心函数
        inputs=[msg, chatbot],  # 输入参数：用户输入+聊天历史
        outputs=[chatbot]  # 输出参数：更新后的聊天历史
    ).then(
        fn=lambda: gr.Textbox(value=""),  # 回调函数：清空输入框
        inputs=[],  # 无输入
        outputs=[msg]  # 输出：清空后的输入框
    )

    # ---------------------- 4.2 绑定回车提交事件 ----------------------
    # 输入框按回车：执行问答函数→清空输入框（提升用户体验）
    msg.submit(
        fn=web_tcm_chat,  # 绑定的核心函数
        inputs=[msg, chatbot],  # 输入参数：用户输入+聊天历史
        outputs=[chatbot]  # 输出参数：更新后的聊天历史
    ).then(
        fn=lambda: gr.Textbox(value=""),  # 回调函数：清空输入框
        inputs=[],  # 无输入
        outputs=[msg]  # 输出：清空后的输入框
    )

# ====================== 5. 启动Web服务 ======================
if __name__ == "__main__":
    # 启动Gradio应用
    demo.launch(
        server_name="0.0.0.0",
        server_port=7880,
        show_error=True,
        debug=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Base()  # 替换Soft为Base
    )