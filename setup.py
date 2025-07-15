from setuptools import setup, find_packages

setup(
    name="sikufenci",
    version="1.0.0",
    description="基于sikuBERT预训练模型的自动分词工具",
    packages=find_packages(),
    install_requires=[
        "torch>=1.1.0",
        "boto3",
        "pytorch_pretrained_bert==0.6.1",
        "seqeval",
        "tqdm",
    ],
    python_requires=">=3.6",
    author="sikufenci",
    author_email="sikufenci@example.com",
    url="https://github.com/sikufenci/sikufenci",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)