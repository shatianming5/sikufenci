# sikufenci - 繁体中文古籍分词工具

这是一个基于sikuBERT预训练模型的自动分词工具，主要用于繁体中文古籍文本的自动分词。不仅能用于带有标点信息的繁体中文语料，也能够很好的适应不含标点语料的分词。

## 特性

- 纯CPU分词模式，无需GPU环境
- 自动调用全部CPU核心进行分词
- 模块化架构，代码结构清晰
- 支持批量处理文本文件
- 优化的CPU性能，适合服务器部署

## 安装方式

### 1. 克隆项目
```bash
git clone https://github.com/shatianming5/sikufenci.git
cd sikufenci
```

### 2. 创建虚拟环境（推荐）
```bash
python3 -m venv sikufenci_env
source sikufenci_env/bin/activate  # Linux/Mac
# 或 sikufenci_env\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
# 安装CPU版本的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install pytorch_pretrained_bert==0.6.1 seqeval boto3 tqdm numpy
```

或者直接使用requirements.txt安装：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytorch_pretrained_bert==0.6.1 seqeval boto3 tqdm numpy
```

### 4. 下载预训练模型文件

下载pytorch_model.bin文件并放置到项目目录：

**百度网盘:**
|链接                                       |提取码     |
| :------------------------------------------ | :-------- |
| https://pan.baidu.com/s/1ePPlCpoZ4UTsUaQumMpZTQ   | c9hb |

**Google Drive:**
| Model                                       | Link      |
| :------------------------------------------ | :-------- |
| sikubert_vocabtxt(fine-tuned)   | https://drive.google.com/drive/folders/1uA7m54Cz7ZhNGxFM_DsQTpElb9Ns77R5?usp=sharing |

下载完成后，将`pytorch_model.bin`文件放到：
```
sikufenci/train_fenci_sikuroberta_vocabtxt/pytorch_model.bin
```

## 使用方法

### 基本用法

```python
from sikufenci import wordsegall_txt

# 分词处理
wordsegall_txt.TCfenci_all(
    raw_path='datatest',        # 输入文件夹
    resultpath='resulttest',    # 输出文件夹  
    max_seq_length=128,         # 最大序列长度
    eval_batch_size=3           # 批处理大小
)
```

### 参数说明

- **raw_path**: 存放待分词txt文件的文件夹路径
- **resultpath**: 分词结果保存的文件夹路径  
- **max_seq_length**: 最大截断长度（1-512），超过此长度的序列会被切分
- **eval_batch_size**: 模型一次处理的序列数量

### 使用步骤

1. **准备输入数据**
   - 创建输入文件夹（如`datatest`）
   - 将待分词的txt文件放入文件夹
   - 确保文件为UTF-8编码

2. **创建输出文件夹**
   ```bash
   mkdir resulttest
   ```

3. **运行分词**
   ```python
   from sikufenci import wordsegall_txt
   wordsegall_txt.TCfenci_all('datatest', 'resulttest')
   ```


## 数据格式要求

### 输入文件要求

1. **文件格式**: txt文件，UTF-8编码
2. **文件结构**: 每行一个句子，使用换行符`\n`分隔  
3. **句子长度**: 建议单句长度在512字符以下
4. **字符支持**: 确保文件中的字符能在UTF-8编码下正常显示

### 输入示例

```
魏帝召而謂之曰："卿風度峻整，姿貌秀異，後當升進，何以處官？"琡曰："宗廟之禮，不敢不敬，朝廷之事，不敢不忠，自此以外，非庸臣所及。

正光中，行洛陽令，部內肅然。

有犯法者，未加拷掠，直以辭理窮核，多得其情。
```

### 输出示例

```
魏帝/召/而/謂/之/曰/：/"/卿/風度/峻整/，/姿貌/秀異/，/後/當/升進/，/何以/處/官/？/"/琡/曰/：/"/宗廟/之/禮/，/不/敢/不/敬/，/朝廷/之/事/，/不/敢/不/忠/，/自/此/以/外/，/非/庸臣/所/及/。/

正光/中/，/行/洛陽/令/，/部/內/肅然/。/

有/犯/法/者/，/未/加/拷掠/，/直/以/辭理/窮核/，/多/得/其/情/。/
```

## 项目结构

```
sikufenci/
├── core/               # 核心分词功能
│   ├── wordsegall_txt.py    # 主要分词接口
│   ├── simple_wordseg.py    # 简化分词接口
│   └── json_wordseg.py      # JSON文件处理
├── models/             # 模型相关
│   ├── tokenizer.py         # 分词器
│   ├── model_loader.py      # 模型加载器
│   └── predictor.py         # 预测器
├── utils/              # 工具函数
│   ├── file_utils.py        # 文件处理
│   ├── text_utils.py        # 文本处理
│   └── device_utils.py      # 设备检测
└── train_fenci_sikuroberta_vocabtxt/  # 模型文件目录
    └── pytorch_model.bin     # 预训练模型（需下载）
```

## 系统要求

- Python 3.7+
- 至少2GB内存
- 支持多核CPU（自动使用所有核心）

## 性能说明

- **CPU模式**: 本项目已优化为纯CPU版本，无需GPU环境
- **内存使用**: 加载模型约需要1GB内存
- **处理速度**: 根据CPU核心数自动调整，多核CPU处理速度更快
- **批处理**: 支持批量处理大量文档

## 注意事项

- 首次运行前确保已下载pytorch_model.bin文件
- 建议使用虚拟环境避免依赖冲突
- 本版本为纯CPU版本，无需CUDA环境
- 分词结果使用`/`符号分隔词语
- 建议在多核CPU环境下运行以获得最佳性能
