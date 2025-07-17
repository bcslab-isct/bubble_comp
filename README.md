# bubble_comp
マイクロ・ナノバブルの動的挙動を解析するCFDコード
圧縮性気液二相流をBVD法で解く。


## ダウンロードとインストール（初期設定）
Linux環境で、C++とpythonのコンパイラはインストール済みと仮定
まずはソースコードをダウンロード：
$ git clone https://github.com/Xiao-Lab-Titech/MLBVD_Data
ディレクトリを移動：
$ cd bubble_comp
pythonの仮想環境を作成：
$ python3 -m venv plot_env
モジュールvenvが見つからない場合は，次のコマンドでインストールします。
$ sudo apt install python3.xx-venv
仮想環境を有効化：
$ source plot_env/bin/activate
指定されたバージョンのPythonライブラリを一括ダウンロード：
$ pip3 install -r requirements.txt


## 使用方法
- C++パート
コマンド例
コンパイル：
$ make
実行：
$ ./run.out
デバッグ：
$ make debug

