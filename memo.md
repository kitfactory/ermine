# shphinx

# flask

# PyPI

* 自作モジュールを用意する
まずは作業をするための自作モジュールのディレクトリを作り、その中に移動します。ディレクトリはなんでもいいですが、わかりやすいようモジュール名と同じにしておきます。

```
$ mkdir libhollow
$ cd libhollow

```
このディレクトリ内部に __init__.py を作成します。中身は空でよいので touch コマンドで作ります。
```
$ touch libhollow/__init__.py

```

同じくモジュールに含める Python のコードも入れておきます。ここでは libhollow/jackolantern.py というファイル名で、以下の Python コードを置いておきます。

``` 
def call():
    print "Trick or Treet"
```

次に、pip がこのモジュールを認識できるよう、setup.py という Python ファイルを作ります。中身は以下の通りです。

```
import setuptools
 
setuptools.setup(
    name="libhollow",
    version="1.0",
    author="momokan",
    author_email="momokan@example.com",
    description="libhollow is my own python package",
    long_description="The halloween party has gone over...",
    long_description_content_type="text/markdown",
    url="https://blog.chocolapod.net/momokan",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7.14",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
```

自作モジュールの呼び出し側では、pip install コマンドの引数に自作モジュールの Git リポジトリのパスを指定することで、自作モジュールをインストールすることができます。ここではインストールする Git リポジトリのパスを /home/momokan/python/libhollow とします。



```
$ pip install git+file:///home/momokan/python/libhollow -t lib

```

上記の解説ではeasy_installを使用していますが、pip install Sphinxや、Anaconda/Minicondaを使っていればconda install Sphinxでも導入可能です。
