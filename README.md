项目名称：中医智能在线诊疗
项目介绍：基于Llamaindex构建的RAG实践项目，构建垂域（向量）知识库，补充llm能力，实现Web在线中医在线诊疗。  
使用说明：调用api模型，结合本地处理过的数据，适合学习ai大模型开发的刚入门小白，不挑设备，可放心食用的简单轻量可运行案例。  
快速开始：  
环境：python=3.11 (使用conda或者虚拟环境，本人用conda，命令为：conda create -n 你的环境名称 python=3.11）  



安装依赖包：  
pip install llama-index  
pip install llama-index-llms-dashscope  
pip install llama-index-embeddings-dashscope  


目录结构：  

├── src/          # 核心代码  
│   ├── main.ts   # 入口文件  
│   └── utils/    # 工具函数  
├── public/       # 静态资源  
└── config/       # 配置文件  

  
注意事项：gradio=6.5.1  
pip install gradio==6.5.1  


