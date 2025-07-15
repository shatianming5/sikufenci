# 安装方式
pip install sikufenci


# introduction
这是一个基于sikuBERT预训练模型的自动分词工具，主要用于繁体中文古籍文本的自动分词,不仅能用于带有标点信息的繁体中文语料，也能够很好的适应不含标点语料的分词。
工具包具有cpu分词与gpu分词两种模式，如果您的设备没有安装gpu,可以自动调用全部的cpu核心进行分词。
而在安装gpu后，代码则会利用gpu加速分词速度，gpu与cpu分词的结果完全一致。

# prepare
运行需要的依赖:

torch>1.1.0

boto3

pytorch_pretrained_bert==0.6.1

seqeval

tqdm

建议创建一个虚拟环境，以正常安装sikufenci。

除上述依赖包外，如果要正常运行代码，还需要下载一个用于分词的pytorch_model.bin文件。

该文件可以通过如下的百度云链接下载:

|链接                                       |提取码     |
| :------------------------------------------ | :-------- |
| https://pan.baidu.com/s/1ePPlCpoZ4UTsUaQumMpZTQ   | c9hb |


Foreign users can download the fine-tuned model through Google Drive:
| Model                                       | Link      |
| :------------------------------------------ | :-------- |
| sikubert_vocabtxt(fine-tuned)   | https://drive.google.com/drive/folders/1uA7m54Cz7ZhNGxFM_DsQTpElb9Ns77R5?usp=sharing |


下载完成后，需要将pytorch_model.bin文件放到sikufenci安装目录的子文件夹的'train_fenci_sikuroberta_vocabtxt'文件夹中。

安装目录就是您默认的安装此python工具包的位置。例如，在我的电脑中安装目录就是D:\ProgramData\Anaconda3\envs\pyqt5_py38\Lib\site-packages\sikufenci\train_fenci_sikuroberta_vocabtxt

如果上述工作都已完成，就可以进入运行阶段。

# Run

# 调用模块方式
from sikufenci import wordsegall_txt

# 使用方式
wordsegall_txt.TCfenci_all(raw_path='datatest',resultpath='resulttest',max_seq_length=128,eval_batch_size=3)

TCfenci_all函数含有四个参数:

raw_path:代表您当前存放待分词语料的文件夹，可以存放多个txt文件。

resultpath:代表您希望分词后文件的存储位置,在案例中是一个被命名为resulttest的空文件夹

max_seq_length:最大截断长度，超过这一长度的待分词序列会被以该值大小等分，例如，当我有一个长度为257的句子时，而max_seq_length值为128时，
会将句子切分为长度128，128，1的三个子句。所以，为保持语义的完整性，应根据您的分词语料具体情况确定该值。但最高不能超过512。值越大代码运行速度越慢。

eval_batch_size:模型一次性分词的序列数。


# 数据实例
您应该按照如下原则安排待分词语料的文件夹:

1.单个句子长度不宜过长，建议单句长度在512以下。使用换行符"\n"来切分不同的句子。

2.文件夹中的文件应当以txt为后缀名。

3.应尽量确保分词文件中不包含在utf-8编码下无法呈现的字符。

真实的数据样例:

魏帝召而謂之曰："卿風度峻整，姿貌秀異，後當升進，何以處官？"琡曰："宗廟之禮，不敢不敬，朝廷之事，不敢不忠，自此以外，非庸臣所及。

"正光中，行洛陽令，部內肅然。

有犯法者，未加拷掠，直以辭理窮核，多得其情。

於是豪猾畏威，事務簡靜。

時以久旱，京師見囚悉召集華林，理問冤滯，洛陽系獄，唯有三人。

魏孝明嘉之，賜縑百匹。

遷吏部，尚書崔亮奏立停年之格，不簡人才，專問勞舊。


分词后的数据样例:

魏帝/召/而/謂/之/曰/：/"/卿/風度/峻整/，/姿貌/秀異/，/後/當/升進/，/何以/處/官/？/"/琡/曰/：/"/宗廟/之/禮/，/不/敢/不/敬/，/朝廷/之/事/，/不/敢/不/忠/，/自/此/以/外/，/非/庸臣/所/及/。/

"/正光/中/，/行/洛陽/令/，/部/內/肅然/。/

有/犯/法/者/，/未/加/拷掠/，/直/以/辭理/窮核/，/多/得/其/情/。/

於是/豪猾/畏/威/，/事/務/簡靜/。/

時/以/久/旱/，/京師/見/囚/悉/召集/華林/，/理問/冤滯/，/洛陽/系/獄/，/唯/有/三/人/。/

魏孝明/嘉/之/，/賜/縑/百/匹/。/

遷/吏部/，/尚書/崔亮/奏/立/停/年/之/格/，/不/簡/人才/，/專/問/勞舊/。/


可以看到模型具有较好的分词效果。有效解决当前缺少面向繁体中文的古文分词工具问题。
