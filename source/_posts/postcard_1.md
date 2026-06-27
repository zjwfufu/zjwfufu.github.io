---
title: Gaussian-Splatting-Gallery
mathjax: true
date: 2023-12-13 18:49:27
tags: [杂项]
categories: [杂项]
---

​	今天试了下在blog里整一个Gaussian Splatting的Viewer，所以突发奇想搞了这么个系列。

<!--more-->

<link rel="stylesheet" href="/3dgs/style.css" />

{% raw %}
<div id="viewer-container">
<select id="modelDropdown">
<option value="https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/bonsai-7k-mini.splat">Bonsai</option>
<option value="https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/fufu.splat">Hatsune Miku fufu</option>
<option value="https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/toy_trex.splat">Toy trex skeleton</option>
<option value="https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/bird.splat">Cockatiel</option>
<option value="https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/shark.splat">Blåhaj</option>
<option value="https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/park.splat">YanJiao Park</option>
<option value="https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/school.splat">My High School</option>
</select>
<canvas id="canvas"></canvas>
<div id="progress-dialog" class="gs-modal">
<p><label for="progress-indicator">Loading...</label></p>
<progress max="100" id="progress-indicator"></progress>
</div>
</div>
<script src="/3dgs/index.js" type="module"></script>
{% endraw %}

> 用鼠标可以拖拽旋转，WASD可以前进后退（某种意义上的），Q和E也可以旋转。我加了个事件监视器，所以鼠标如果移出去了画面就会暂停，某种程度上节约资源吧。
>
> **训练好的3DGS被寄存在huggingface中，所以在导入时需要科学上网。**
>
> **技术力有限，只能支持PC端。**
>
> **为了提高渲染效率，`.ply`格式被压制成了`.splat`格式，所以质量有所降低。**

​	事情是这样的，我甲流刚好，然后看不动代码论文什么的，然后就想起来整了个这个玩意。

​	起因有很多，其中一个是前几天跟一个学长聊天，他跟我说读研的那边好多吃的都没有，于是我才意识到很多东西不是理所应当的，比如益禾堂，烤冷面。以及估计马上就要离开西电，以后可能也吃不上了，萌发了下学期回去以后把他们都“录像一圈—>训出高斯点云”这样来收集起来。

​	其二是想起来之前Apple Vision Pro里说的那个空间视频，那个就跟一个4D的Gaussian Splatting很像（当然在出这个产品的宣传的时候还没有3DGS）。我当然不可能做这么好，但是用3DGS做个丐版也是不错的。人人都有一些想记录下来的场景，但无论如何你怎么改进成像本身，把拍出来的照片分辨率拉的再高再细致，洗出来裱起来等等，它终究是一“张”纸或者手机里的一个二维数组。所以做出一个可以交互的场景还是有意义的，哪怕这个场景现在仍然充满着监督不够产生的石棉一般的针刺。

​	但不管怎么说，维护回忆的最好办法还是在能创造回忆的时候尽量多创造回忆，但只不过大部分时候人们都不会意识到当下稀疏平常的东西会是以后的“回忆”。

​	在网页上实现这么一个高斯点云的viewer，依托于[gsplat.js](https://github.com/dylanebert/gsplat.js)，我只是单纯的把它缝进我的blog里了。本来我是想一个页面里放若干个这样的viewer，然后每个是不同的场景，做成一种博物馆/画廊一样的感觉，但整了一会儿没整出来，而且发现本地用浏览器渲染一个场景，电脑风扇就开始转了，所以感觉多张放一个html感觉可能要寄。

### Supplement material

​	“Cockatiel”那个模型拍摄于一只[玄凤鹦鹉](https://zh.wikipedia.org/wiki/%E9%9B%9E%E5%B0%BE%E9%B8%9A%E9%B5%A1)，但鸡不可能那么配合，所以拍摄的时候他有在左动右动，于是产生了模糊。这是他的照片：

<center>
    <img src='/images/postcard/postcard_1.jpg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

### End

​	“小巷里，让细雨，穿成线。”

​	“浸透在，海棠色，的华年。”

<center>
    <img src='/images/postcard/postcard_end.gif' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>
