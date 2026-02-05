# 导入日志模块，用于打印程序运行日志、调试信息
import logging
# 导入系统模块，用于日志输出流配置
import sys
# 导入操作系统模块，用于文件/目录操作、环境变量读取
import os
# 导入类型注解模块，用于可选类型标注
from typing import Optional

# 从llama_index核心库导入核心组件
from llama_index.core import (
    PromptTemplate,  # 提示词模板类，用于自定义问答提示词格式
    Settings,        # 全局配置类，统一配置LLM、嵌入模型、回调等
    StorageContext,  # 存储上下文类，用于加载/持久化向量索引
    load_index_from_storage,  # 从持久化目录加载已构建的向量索引
    VectorStoreIndex,  # 向量存储索引类，核心的向量索引构建/查询类
    SimpleDirectoryReader,  # 目录读取器，用于加载本地文档
)
from llama_index.core.node_parser import SentenceSplitter  # 句子分割器，用于文档分块
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager  # 回调管理器，用于调试/监控
# 导入通义千问LLM模型适配层
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
# 导入通义千问嵌入模型适配层
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels


# 初始化项目基础配置（日志、通义千问模型、全局回调）
def init_basic_config():
    # 配置日志：输出到标准输出流、日志级别为INFO（打印关键运行信息）
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # 为根日志器添加额外的标准输出流处理器，确保日志正常打印
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # 从系统环境变量中读取通义千问API密钥（避免硬编码，提高安全性）
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    # 校验API密钥是否存在，不存在则抛出异常终止程序
    if not dashscope_api_key:
        raise ValueError("未找到DASHSCOPE_API_KEY环境变量！请检查环境变量配置是否正确。")

    # 配置全局大语言模型（LLM）：使用通义千问QWEN_MAX模型
    Settings.llm = DashScope(
        model_name=DashScopeGenerationModels.QWEN_MAX,  # 模型名称：通义千问旗舰版
        api_key=dashscope_api_key,  # 通义千问API密钥
        temperature=0.0,  # 温度系数：0表示输出结果确定性强（适合专业问答）
        max_tokens=2048  # 最大生成令牌数：限制回答的长度
    )
    # 配置全局嵌入模型：使用通义千问文本嵌入V1模型（用于生成文档/查询的向量表示）
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V1,  # 通义千问文本嵌入模型
        api_key=dashscope_api_key,  # 通义千问API密钥
    )

    # 初始化LLamaIndex调试处理器，关闭结束时的轨迹打印（避免冗余日志）
    llama_debug = LlamaDebugHandler(print_trace_on_end=False)
    # 配置全局回调管理器，添加调试处理器（用于监控模型调用、索引查询等流程）
    Settings.callback_manager = CallbackManager([llama_debug])


# 定义中医专属问答Prompt模板（贴合中医诊疗逻辑，约束模型回答风格）
def get_tcm_prompts():
    # 拼接中医问答提示词模板字符串，包含上下文占位符、查询占位符
    qa_prompt_tmpl_str = (
        "上下文信息如下（中医典籍/诊疗指南）：\n"
        "---------------------\n"
        "{context_str}\n"  # 向量索引检索到的中医相关上下文文档片段
        "---------------------\n"
        "请严格根据上下文，以专业中医医师的角度回答以下问题，回答需严谨、简洁，符合中医辨证逻辑：\n"
        "Query: {query_str}\n"  # 用户的中医问题查询
        "Answer: "  # 模型回答的占位符
    )
    # 将字符串模板转换为llama_index可识别的PromptTemplate对象并返回
    return PromptTemplate(qa_prompt_tmpl_str)


# 构建/加载向量索引（优先加载已持久化的索引，避免重复构建；移除了原Prompt更新逻辑）
# data_dir：本地中医文档存储目录，persist_dir：向量索引持久化目录，chunk_size：文档分块的大小
def build_or_load_index(data_dir: str = "./data", persist_dir: str = "./doc_emb",
                        chunk_size: int = 256) -> VectorStoreIndex:
    # 判断向量索引持久化目录是否存在且非空（存在已构建的索引）
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"加载已存储的向量索引（{persist_dir}）...")
        # 从持久化目录初始化存储上下文
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        # 从存储上下文加载已构建的向量索引
        index = load_index_from_storage(storage_context)
    else:
        # 无已构建索引，从本地文档目录加载中医文档
        print(f"从{data_dir}加载中医文档...")
        # 读取指定目录下的txt格式文档（过滤其他格式，避免无效文件）
        documents = SimpleDirectoryReader(data_dir, required_exts=[".txt"]).load_data()
        # 初始化句子分割器：按指定chunk_size对文档进行分块（适配嵌入模型输入长度）
        splitter = SentenceSplitter(chunk_size=chunk_size)
        # 从加载的文档构建向量索引，通过分块器对文档进行预处理
        index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
        # 将构建好的向量索引持久化到指定目录，方便后续直接加载
        print(f"保存向量索引到{persist_dir}...")
        index.storage_context.persist(persist_dir=persist_dir)

    # ========== 移除这里的 index.update_prompts() ==========
    # 原因：索引层无需绑定Prompt，Prompt应绑定到查询引擎层，保证灵活性
    return index


# 中医问答核心函数（在查询引擎创建后更新Prompt，核心修复点；支持流式/非流式回答）
# query：用户的中医问题，index：已构建的向量索引，similarity_top_k：检索相似文档的数量，streaming：是否开启流式输出
def tcm_chat(query: str, index: VectorStoreIndex, similarity_top_k: int = 5, streaming: bool = True) -> Optional[str]:
    # 1. 将向量索引转换为查询引擎：开启流式输出、设置检索相似文档数为5
    query_engine = index.as_query_engine(streaming=streaming, similarity_top_k=similarity_top_k)

    # 2. 获取中医专属Prompt模板并更新到查询引擎（核心修复）
    # 修复原因：Prompt与查询引擎绑定，而非索引绑定，符合llama_index的设计逻辑
    qa_prompt = get_tcm_prompts()
    # 更新查询引擎的问答模板：使用自定义的中医Prompt约束模型回答
    query_engine.update_prompts({"text_qa_template": qa_prompt})

    # 3. 执行查询：将用户问题传入查询引擎，返回回答结果
    response = query_engine.query(query)
    # 判断是否为流式输出模式
    if streaming:
        print("回答：")
        # 流式打印回答结果（逐字/逐句输出，提升交互体验）
        response.print_response_stream()
        print("\n")
        # 流式模式下直接打印，返回None
        return None
    else:
        # 非流式模式下，将回答结果转为字符串并返回
        return str(response)


# 程序主入口（仅当直接运行该脚本时执行）
if __name__ == "__main__":
    # 初始化项目基础配置（日志、模型、回调）
    init_basic_config()
    # 构建/加载中医文档的向量索引
    tcm_index = build_or_load_index()
    # 测试查询问题：中医常见证候判断
    test_query = "不耐疲劳，口燥、咽干可能是哪些证候？"
    # 调用中医问答函数，执行测试查询
    tcm_chat(test_query, tcm_index)