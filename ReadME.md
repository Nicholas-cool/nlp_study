# 仓库说明

【`Github` 仓库地址】https://github.com/Nicholas-cool/nlp_study



## 〇、参考教程

### 0.1、【五道口纳什】`BERT、T5、GPT` 

【课程链接】https://www.bilibili.com/video/BV1W34y157tA

【代码链接】https://github.com/chunhuizhang/bert_t5_gpt

讲的非常棒，干货十足。对模型的原理源码做了拆解，如果只是想使用模型，这个课程讲的部分底层原理可以不用了解。



## 一、环境搭建

### 1.1、服务器租赁

在 `AutoDL` 上租赁算力服务器，选择 `PyTorch 2.0.0` 及以上镜像。

### 1.2、源码克隆

克隆源码 `git clone https://github.com/Nicholas-cool/nlp_study.git`

### 1.3、虚拟环境创建

创建新的虚拟环境 `conda create -n nlp_study python=3.8`

初始化终端对 `conda` 的支持 `conda init`

打开新的终端执行 `conda activate nlp_study` 切换到该虚拟环境

将新的 `conda` 虚拟环境加入 `jupyterlab` 中 `conda install ipykernel` | `ipython kernel install --user --name=nlp_study`

### 1.4、库包安装

打开 `pytorch` 官网选择和复制下载命令 https://pytorch.org/。注意此处使用 `pip` 而非 `conda` 进行安装，以防止解析环境时间过长。

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

安装其他依赖包 `pip install -r requirements.txt`

### 1.5、运行代码

注意部分模型需要从国外网站下载，因此需注意网络通畅。

如果使用 `AutoDL` 运行，可添加相应代码段，参见其说明文档 https://www.autodl.com/docs/network_turbo/。

选择新建的 `nlp_study` 环境，即可运行代码。

如果遇到 `SSLError` 问题，如下。

```bash
SSLError: (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /distilbert-base-uncased-finetuned-sst-2-english/resolve/main/tokenizer_config.json (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:xxxx)')))"), '(Request ID: 07ebce0c-8df0-4e38-82b0-xxxxxxxxx70a2)')
```

可以降级以下两个包的版本。

```bash
pip install requests==2.27.1
pip install urllib3==1.25.11
```

> 【注意】安装 `dataset` 的时候又会将 `requests` 包自动升级，需要再次手动降级！！

并禁用证书验证。

```python
os.environ['CURL_CA_BUNDLE'] = ''
```

> 【注】在代码中添加该行对环境变量的修改，仅对当前 `python` 进程有效。

如果遇到如下 `TqdmWarning`。

```bash
TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
```

可以安装以下包进行解决。

```bash
pip install ipywidgets
pip install --upgrade jupyter  # 可选
```



## 二、代码说明

见代码注释。
