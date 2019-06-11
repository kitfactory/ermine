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

The scripts Keyword Argument
The first approach is to write your script in a separate file, such as you might write a shell script.:

funniest/
    funniest/
        __init__.py
        ...
    setup.py
    bin/
        funniest-joke
    ...
The funniest-joke script just looks like this:

#!/usr/bin/env python

import funniest
print funniest.joke()
Then we can declare the script in setup() like this:

setup(
    ...
    scripts=['bin/funniest-joke'],
    ...
)





flask

# Flask などの必要なライブラリをインポートする
from flask import Flask, render_template, request, redirect, url_for
import numpy as np

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)

# メッセージをランダムに表示するメソッド
def picked_up():
    messages = [
        "こんにちは、あなたの名前を入力してください",
        "やあ！お名前は何ですか？",
        "あなたの名前を教えてね"
    ]
    # NumPy の random.choice で配列からランダムに取り出し
    return np.random.choice(messages)

# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理
@app.route('/')
def index():
    title = "ようこそ"
    message = picked_up()
    # index.html をレンダリングする
    return render_template('index.html',
                           message=message, title=title)

# /post にアクセスしたときの処理
@app.route('/post', methods=['GET', 'POST'])
def post():
    title = "こんにちは"
    if request.method == 'POST':
        # リクエストフォームから「名前」を取得して
        name = request.form['name']
        # index.html をレンダリングする
        return render_template('index.html',
                               name=name, title=title)
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))


@app.route('/name/<name>.json')
def hello_world(name):
  greet = "Hello %s from flask!" % name
  result = {
    "Result":{
      "Greeting": greet
      }
  }
  return jsonify(ResultSet=result)

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    app.run(host='0.0.0.0') # どこからでもアクセス可能に




# pytest

@pytest.mark.xxxx

def reducer(array_val, la_bel):
    c = tf.cast(label, tf.int32)
    x = tf.gather(array_val,c)
    x = x + 1
    array_val = tf.tensor_scatter_update(array_val,[[c]],[x])
    return array_val

array_val = train_dataset.reduce(array_val, reducer)
<<<<<<< HEAD


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)

d = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)]
ds = tf.data.Dataset.from_tensor_slices(d)

def to_class(d):
    x = tf.gather(d,indices=[0],axis=0)
    x = tf.reshape(x, shape=[] )
    return x

ds = ds.apply(tf.data.experimental.rejection_resample(to_class,target_dist=[0.1,0.1,0.8],initial_dist=[0.33,0.34,0.33]))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iterator = ds.make_one_shot_iterator()
    next = iterator.get_next()
    try:
        while True:
            print('Got !!' , sess.run(next))
    except tf.errors.OutOfRangeError as error:
        print('end')


 190010467
 190010443
 190010445
 190010449
 
=======
>>>>>>> e5ac523ad93226b384a05a1257124502774dc217



# TFX


#### サブクラスの取得
追記
やっぱりあった。

id:mopemope さん『何もしなくてもOutputBase.__subclasses__()でいけるはずです』


class OptunaCallback(Callback):
    def __init__(self, trial):
        self.trial = trial

    def on_epoch_end(self, epoch, logs):
        current_val_error = 1.0 - logs["val_acc"]
        self.trial.report(current_val_error, step=epoch)
        # 打ち切り判定
        if self.trial.should_prune(epoch):
            raise optuna.structs.TrialPruned()


#### モデルの評価

- AUC/ROC しきい値の設定
- 
tf.keras.metrics.AUC


### 評価タスクの実行

- データセットの選択
- 前処理条件の選択
- モデルファイルの選択
- クラス敷居値オプション
- CSV結合オプション
- 画像保存オプション
- yellowbrick


### Angular

#### ElectronとAngularの関係
- https://github.com/maximegris/angular-electron/blob/master/README.md


#### Angular
コンポーネントの作成

```
> ng new 

```
### サービス



### ルーティング


- ngx-build-plus

Create a new Angular project with the CLI

Add ngx-build-plus: ng add ngx-build-plus


Note: If you want to add it to specific sub project in your projects folder, use the --project switch to point to it: ng add ngx-build-plus --project getting-started

Remark: This step installs the package via npm and updates your angular.json so that your project uses custom builders for ng serve and ng build.

Add a file webpack.partial.js to the root of your (sub-)project:

const webpack = require('webpack');

module.exports = {
    plugins: [
        new webpack.DefinePlugin({
            "VERSION": JSON.stringify("4711")
        })
    ]
}
Use the global variable VERSION in your app.component.ts:

import { Component } from '@angular/core';

declare const VERSION: string;

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css']
})
export class AppComponent {
    title = 'Version: ' + VERSION;
}
Start your application with the --extra-webpack-config switch pointing to your partial webpack config:

ng serve --extra-webpack-config webpack.partial.js -o
If your project is a CLI based sub project, use the --project switch too:

ng serve --project getting-started -o --extra-webpack-config webpack.partial.js
Hint: Consider creating a npm script for this command.

Make sure that the VERSION provided by your webpack config is displayed.



- 
- 
