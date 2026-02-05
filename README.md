项目名称：中医智能在线诊疗
项目介绍：基于Llamaindex构建的RAG实践项目，构建垂域（向量）知识库，补充llm能力，实现Web在线中医在线诊疗。  
使用说明：调用api模型，结合本地处理过的数据，适合学习ai大模型开发的刚入门小白，不挑设备，可放心食用的简单轻量可运行案例。  
  
快速开始：  
环境：python=3.11 (使用conda或者虚拟环境，本人用conda，命令为：conda create -n 你的环境名称 python=3.11）  

在创建的环境下安装依赖包：  
pip install llama-index  
pip install llama-index-llms-dashscope  
pip install llama-index-embeddings-dashscope  
pip install gradio==6.5.1 
配置api-key:  
在阿里云百炼申请即可免费试用100万token，然后在本地配置api-key的环境变量  
变量名为：DASHSCOPE_API_KEY  变量值为你申请的api-key

目录结构：  
├── data/ # 项目核心数据目录（存放中医语料/测试文本）  
│   ├── demo.txt # 演示用测试文本  
│   └── 《中医临床诊疗术语第2部分：... # 中医诊疗术语核心语料文件  
├── doc_emb/ # 文档嵌入/向量数据库目录（AI语义检索核心数据）  
│   ├── default_vector_store.json # 默认文本向量存储文件  
│   ├── docstore.json # 原始文档与向量映射存储文件  
│   ├── graph_store.json # 知识图谱结构存储文件（中医关联关系）  
│   ├── image_vector_store.json # 图片向量存储文件（如中医相关图片检索）  
│   └── index_store.json # 向量/文档检索索引文件（提升查询效率）   
├── README.md # 项目说明文档（介绍/使用/配置等）  
├── Untitled.ipynb # Jupyter Notebook（使用jupyter进行分步学习调试，理解模型运行底层原理）  
├── tcm_chatbot.py # 中医聊天机器人核心逻辑文件（TCM=中医）  
└── tcm_web_ui.py # 中医机器人Web交互界面文件（可视化对话入口）  

运行项目：  
首先运行tcm_chatbot.py文件查看是否报错，再运行tcm_web_ui.py

在线体验：  
输入网址：http://localhost:7880/   

<img width="2300" height="1001" alt="image" src="https://github.com/user-attachments/assets/ed1bda5a-d509-443b-a44c-99cdf11b1c5e" />


