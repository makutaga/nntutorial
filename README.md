BP (Jump Every Time) による3層ニューラルネットワーク学習プログラム
=========================================================================

--- 目次 ---


０．著作権等について
１．はじめに
２．使用法
３．結合ファイルフォーマット
４．今後の発展


０．著作権等について
-------------------------------------------

作者は本プログラムのソース tutorial.c 及びに関する著作権を主張しない．
内容の改変，配布等についても何ら制限を設けない．ただし作者や関係者に対
する著しい中傷や名誉毀損を伴う場合はこの限りではない．

利用者は各自の責任に於いてこのパッケージを使用すること．本プログラムの
利用に伴う利益，不利益に対しては著者は何ら感知せず，責任も負わない．

作者は本プログラムのバージョンアップ，バグフィックス，サポート等を，今
後継続して行うとは限らない．


１．はじめに
-------------------------------------------

tutorial.c は Back Propagation Algorithm (Jump Every Time Version)によ
るフィードフォワード型3層ニューラルネットワークの学習を行うプログラム
である．C言語の基本を既に身に付けているまたは，身に付けつつある人を対
象としており，作成に当たっては以下の点について留意した．

o 技巧的なコーディングは極力避ける

o ある程度の拡張性を持たせる

o なるべくコメントを入れる

o 「きれいな」コーディングに心がける

なおオブジェクト指向という概念が不可欠となる構造体は使用していない．こ
のため各関数の引数リストはかなり長くなっている．ただし構造体を使用する
ようにも変更しやすいよう，関数の引数は種類や順番をできる限り統一したつ
もりである．このため実際に関数内では使用されていないにも関わらず，引数
リストに含まれているものもある．またC言語をはじめた人が誰でも必ずつま
づくポインタについては，使わなければいつまでたっても使えないという考え
のもと，積極的に導入している．もっともポインタを使わずにコーディングす
るのは不可能ではないだろうか(mainで全部処理するとか)．

ソースはGNUのRCS(Revision Control System)によりリビジョン(バージョン)
管理を行っている．したがってリビジョン番号やタイムスタンプを見れば，そ
のソースがどのくらい新しいものかがわかるようになっている．最新版は所定
のディレクトリ(/home/dors/murakami/usr/neuro/tutorial)から入手できる．

関数のプロトタイプ宣言はK&Rタイプとしているが，今後ANSIに変更していく
予定である．


２．使用法
-------------------------------------------

本プログラムの使用手順を示す．

1. 作業を行う適当なディレクトリを作成し，カレントディレクトリをそこへ移す．
```
% cd
% mkdir neuro
% mkdir neuro/parity
% cd neuro/parity
```

2. tutorial.c をコピーする．
```
% cp ~murakami/usr/neuro/tutorial/tutorial.c .
```

3. 学習パターンファイル(入力パターン，出力パターンの2つのファイルからなる．)
   を作成する．書き方は後述．
```
% vi parity.in
   (入力パターンファイルの編集)
% vi parity.tg
   (出力パターンファイルの編集)
```

4. tutorial.c で定義している．学習パラメータ，ファイル名などのマクロを編
   集する．定義されているマクロは後述
```
% vi tutorial.c
 (ファイルの編集)
```

5. コンパイルする．
```
% gcc -O -o tutorial tutorial.c -lm
```
   エラーがでなければOk．エラーが出たら... 何とかしてエラーを出なくする．

6. 実行する．
```
    % ./tutorial
     (色々な情報が表示される)
```
   誤差の変化，結合荷重が記録されたファイルが一つずつ作られる．

7. 学習結果の善し悪しを判断する．

 * パターンファイル：
入出力パターンファイルは，ホワイトスペース(空白，タブ，改行)によって区
切られたASCII文字列である．例えば，n番目のパターンのi番目の入力ユニット
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

となる．

###マクロ
本プログラムは学習に用いる各種パラメータをソースリストの始めの方にある
マクロによって設定する．使用するファイルは入力，出力にそれぞれ2つずつ
であり，これらのファイル名もマクロ定義によって行っている．定義されるマ
クロを列挙する．

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

学習パラメータを変更するだけであれば，この部分以外に手を加える必要はな
い．任意のマクロ定義を変更し再コンパイルすればそれでよい．もちろんユニッ
トの入出力関数や，アルゴリズムを変更する場合は，関数の記述自体を変更し
なければならない．

その他，プログラミングを容易にするために，結合荷重値を参照するマクロも
用意している．このマクロは，例えば以下のようにして使用する．

```
#define INP 10            /* 入力ユニット数         */
#define HID 20            /* 中間層ユニット数       */
double w[(INP+1)*HID]     /* 結合荷重配列の宣言     */
double v;                 /* 荷重値                 */

v = WeightValue( w, INP, HID, 1, 2 );
				/* 入力層1番から中間層2番への結合 */
```

ユニット番号は0から数える(0 origin)事に注意する．
また，バイアスユニットから中間層2番ユニットへの結合値は
```
  v = WeightValue( w, INP, HID, INP, 2 );
```
で得られる．

## ３．結合ファイルフォーマット

本プログラムは，学習の結果得られる結合を tutorial1 フォーマットで，ファ
イルとして保存する．このファイルフォーマットは，ASCIIテキスト形式によ
る多層ニューラルネット記述フォーマットで，任意層数，任意ユニット数のネッ
トワークを記述することが可能である．このフォーマットは，ASCIIテキスト
で構成されているためプラットフォームに依存しない．したがってDOS上，
Macintosh上，UNIX上でどのOS上で作成されたファイルでも，相互に使用可能
である．反面，ファイルフォーマットが単純であるため，特性関数や学習回数
などの情報を記憶しておくことができない．これは利用者がファイル名と各種
パラメータをメモしておかなければならないことを意味する．

このように決して完全とは言えないファイルフォーマットであるにもかかわら
ず，あえて実装した理由は，ニューログループでの数年間に作者の経験からき
ている．従来，我々のグループにはtutorial1フォーマットのような明文化し
たフォーマットは存在しなかった．また学習プログラムも，（雛型は存在した
が）これと言ったベースとなり得るものも存在しなかった．このため新しくグ
ループに加わった人間は，自分でプログラムをつくり，各々の規則で独自のファ
イルフォーマットを作り発展させていた．しかし年を経るにつれ先輩から後輩
へと受け継ぐべき知識やファイルも高度化し，分量も増加してきた．ここで毎
年困った問題が生じるようになった．先輩のシミュレーション結果が再現でき
ないのである．特に乱数を使用するニューラルネットワークは，これが顕著で
ある．先輩と同じ乱数系列，発生させる幅を用いても，それらを結合係数配列
に記憶する順番が異なれば全く違った結果が生じてしまうのである．こうなる
と事実上最初からやりなおしである．後輩はまた同じ道を辿って無駄な時間を
過ごさなければならない．

何故この様な事が起こるのだろうか．原因はシステム化したグループ内共通の
規範の欠如である．もう各自がバラバラにそれぞれのプログラムをメンテナン
スする時代は終った．もちろんプログラミングの勉強がしたいのであればそれ
でも良い．各自がソフトウエアシステムを設計し，失敗し，改良するのも一理
あるかも知れない．しかし我々の目的はそれとは異なるのである．我々はニュー
ラルネットワーク，ひいては知的情報処理システムの研究をするべきなのであ
る．先輩が踏んで来た同じ道を後輩が辿る必要はない．先輩が築いて来た道を
延ばすのが後輩の仕事である．一方先輩は後輩が道に迷わぬよう，道標をたて
るのが仕事である．

tutorial.cとtutorial1フォーマットは，作者なりの道標のつもりで作成した
ものである．もちろんこれを固守する必要は無い．むしろそれは作者の意に反
する．このフォーマットを使用する限りはこれを発展させて頂きたい．

なお tutorial1 フォーマットに関するより詳細な仕様は，
    /home/dors/murakami/pub/nnfile/tutorial.doc (depricated)
に記述されているので，参考にされたい．


## ４．今後の発展

本プログラムは作者自身は使用していない．何故ならば任意層数，任意ユニッ
ト数，スケーリング機能など柔軟性や拡張性に欠けるからである．本プログラ
ムはあくまでも，C言語，BPNN，エディタのチュートリアル用として作成され
たものであり，より複雑な処理を行おうとした時には，機能不足であろう．利
用者はそれぞれの課題に合わせて本ソースを修正するなり，新しく作り直すな
りした方が賢明である．以下に本プログラムを修正すべき点を列挙しておくが，
これらを実装することは本来の目的に反することになるため将来tutorial.c
に組み込まれることはない．

* 学習係数などの各種パラメータのコマンドラインからの取得

* ポインタ操作による動作の高速化

* 構造体を用いたオブジェクト指向コーディング

* 柔軟性に富む層数，ユニット数等の指定

* 汎用モジュールとしてのライブラリ化


Based file : Id: README,v 1.1 1996/05/11 05:05:26 murakami Exp 
