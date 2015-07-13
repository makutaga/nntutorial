BP (Jump Every Time) による3層ニューラルネットワーク学習プログラム
=========================================================================

--- 目次 ---



０．著作権等について

１．はじめに

２．使用法

３．結合ファイルフォーマット

４．今後の発展


※ 古い記述が多々あるため，現在README.mdはRevise中です．


０．著作権等について
-------------------------------------------

本リポジトリ（nntutorial）に含まれるソースリストはPublic Domainです．
作者はこれらのプログラムソースに関する著作権を主張しません．
また内容の改変，配布等についても何ら制限を設けません．
ただし作者や関係者に対する著しい中傷や名誉毀損を伴う場合はこの限りではありません．

利用者は各自の責任に於いてこれらのソースを使用してください．
本プログラムの利用に伴う利益，不利益に対しては著者は何ら感知せず，責任も負いません．
作者は本プログラムのバージョンアップ，バグフィックス，サポート等を，今後継続して行うとは限りません．


１．はじめに
-------------------------------------------

`tutorial.c` は Back Propagation Algorithm (Jump Every Time Version)によるフィードフォワード型3層ニューラルネットワークの学習を行うプログラムです．
C言語の基本を既に身に付けているまたは，身に付けつつある人を対象としており，作成に当たっては以下の点について留意しました．

* 技巧的なコーディングは極力避ける

* ある程度の拡張性を持たせる

* なるべくコメントを入れる

* 「きれいな」コーディングに心がける

構造体は使用しませんでした．
このため各関数の引数リストはかなり長くなり可読性が低くなっている．
ただし構造体を使用するようにも変更しやすいよう，関数の引数は種類や順番をできる限り統一したつもりです．
このため実際に関数内では使用されていないにも関わらず，引数リストに含まれているものもあります．
またC言語をはじめた人が誰でも必ずつまづくポインタについては，使わなければいつまでたっても使えないという考えのもと，積極的に導入しています．


２．使用法
-------------------------------------------

本プログラムの使用手順を示します．

1. 作業を行う適当なディレクトリを作成し，カレントディレクトリを変更します．

```
$ cd
$ mkdir neuro
$ mkdir neuro/parity
$ cd neuro/parity
```

2. tutorial.c をコピーします．

```
$ cp ~murakami/usr/neuro/tutorial/tutorial.c .
```

3. 学習パターンファイル(入力パターン，出力パターンの2つのファイルからなる)
   を作成します．書き方は後述．

```
$ vi parity.in
   (入力パターンファイルの編集)
$ vi parity.tg
   (出力パターンファイルの編集)
```

4. tutorial.c で定義している学習パラメータ，ファイル名などのマクロを編集します．定義されているマクロは後述．

```
$ vi tutorial.c
 (ファイルの編集)
```

5. コンパイルします．

```
$ gcc -O -o tutorial tutorial.c -lm
```

   エラーがでなければOk．エラーが出たら... 何とかしてエラーなくします．

6. 実行します．

```
$ ./tutorial
     (色々な情報が表示される)
```

   誤差の変化，結合荷重が記録されたファイルが一つずつ作られます．

7. 学習結果の善し悪しを判断します．

 * パターンファイル：
入出力パターンファイルは，ホワイトスペース(空白，タブ，改行)によって区
切られたASCII文字列です．例えば，n番目のパターンのi番目の入力ユニット
への入力値を Xni，o番目のユニットの目標出力を Yno とすると，入力パター
ンファイルは，

```
X00 X01 X02 X03
X10 X11 X12 X13
X20 X21 X22 X23
 :   :   :   :
```

出力パターンファイルは

```
Y00 Y01
Y10 Y11
Y20 Y21
 :   :
```

となります．

###マクロ
本プログラムは学習に用いる各種パラメータをソースリストの始めの方にある
マクロによって設定します．使用するファイルは入力，出力にそれぞれ2つずつ
であり，これらのファイル名もマクロ定義によって行っている．定義されるマ
クロを列挙します．

```
INP        入力ユニット数
HID        中間層ユニット数
OUT        出力層ユニット数

EX_NUM     学習パターン数
MAX_ITER   最大学習回数
MIN_ERROR  誤差(rms値)の学習終了条件

IWR        初期結合係数の幅
ETA        学習係数
RND_SEED   乱数の系列

EX_FILE    学習パターン入力ファイル名
TG_FILE    学習パターン出力ファイル名
WT_FILE    結合ファイル名
RMS_FILE   誤差(rms値)ファイル名
```

学習パラメータを変更するだけであれば，この部分以外に手を加える必要はありません．
任意のマクロ定義を変更し再コンパイルすればそれで十分です．
もちろんユニットの入出力関数や，アルゴリズムを変更する場合は，関数の記述自体を変更する必要があります．

その他，プログラミングを容易にするために，結合荷重値を参照するマクロも
用意しています．このマクロは，例えば以下のようにして使用します．

```
#define INP 10            /* 入力ユニット数         */
#define HID 20            /* 中間層ユニット数       */
double w[(INP+1)*HID]     /* 結合荷重配列の宣言     */
double v;                 /* 荷重値                 */

v = WeightValue( w, INP, HID, 1, 2 ); /* 入力層1番から中間層2番への結合 */
```

ユニット番号は0から数えます．つまり0 originです．
また，バイアスユニットから中間層2番ユニットへの結合値は

```
  v = WeightValue( w, INP, HID, INP, 2 );
```

で得られます．

## ３．結合ファイルフォーマット

本プログラムは，学習の結果得られる結合を tutorial1 フォーマットで，ファイルとして保存します．
もっとも，それほど大それたファイルフォーマットではありません．
このファイルフォーマットは，ASCIIテキスト形式による多層ニューラルネット記述フォーマットで，任意層数，任意ユニット数のネットワークを記述することが可能です．
ASCIIテキストで構成されているためプラットフォームに依存しにくいです．
反面，ファイルフォーマットが単純であるため，特性関数や学習回数
などの情報を記憶しておくことができません．
これは利用者がファイル名と各種パラメータを別ファイルにメモしておかなければならないことを意味しています．

なお tutorial1 フォーマットについては，`tutorial.txt` で説明しているので
参考にしてください．


## ４．今後の発展

実は作者自身はこれらのソースコードを使用していません．
任意層数，任意ユニット数，スケーリング機能など柔軟性や拡張性に欠けるためです．
これらのソースコードはムはあくまでも，C言語，BPNN，エディタのチュートリアル用として作成されたものだからです．
利用者はそれぞれの課題に合わせて本ソースを修正するなり，新しく作り直すなりするほうが良いと思います．
以下に本プログラムを修正すべき点を列挙しますが，
これらを項目が，tutorial.cに実装することは本来の目的に反することになるため将来的にも組み込まれることはないでしょう．

* 学習係数などの各種パラメータのコマンドラインからの取得

* ポインタ操作による動作の高速化

* 構造体を用いたオブジェクト指向コーディング

* 柔軟性に富む層数，ユニット数等の指定

* 汎用モジュールとしてのライブラリ化


---
makutaga
