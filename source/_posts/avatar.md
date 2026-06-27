---
title: Digitizing Ourselves
mathjax: true
date: 2025-04-19 21:01:27
tags: [深度学习,3DV]
categories:
  - [深度学习]
  - [3DV]
toc: 1
---

“我欲因之梦吴越，一夜飞度镜湖月。”

<!--more-->

​	相传，在传统文化中，当儿童满月时，会举行一种叫“抓周”的仪式。在抓周仪式上，家长会在孩子面前摆放许多象征不同职业或未来方向的物品，孩子会被引导去抓取这些物品，而大人会根据孩子抓到的东西预测孩子的性格，兴趣或将来职业。这更多是一种趣味性的传统仪式，不构成科学建议。

​	大约一年半之前，我的导师让我从三个方向里选一个作为毕业设计，这其实跟抓周是一样的。因为那三个方向之于我，其实和一岁的小孩之于周围一圈的物品，没有什么区别。我当时选了重建人头化身（Head Avatar），然后后来也探索了一些别的任务。现在正好是入职之前，同时实验室服务器也挂了的一个下午，想对之前看过的一些工作进行一下梳理。

​	哪怕关于avatar的工作只是计算机视觉和图形学中非常狭小的一个子课题，但将其捋清楚也不是那么容易。受自身探索范围的限制，这篇blog应该只会关注2020~2025年中的一些工作。大体上，我想先分成两个大类，无先验的（Prior-free）和有先验的（Prior-guided）。然后分别以从重建（Reconstruction）到生成（Generation）的视角来切，再将人头和人体的内容穿插其中。这里的“Reconstruction”其实事实意义上是“Novel View/Pose Synthesis”，因为我其实对重建几何没太涉猎。

## Prior-free

​	这里的有无先验的区分是在于这个工作有没有用到很强的预训练模型（DinoV2，Saipens，CLIP，GPT，Stable Diffusion等）。无先验的情况下整个工作都是在很古典的机器学习范式下：收集数据，初始化模型，训练模型，评估模型等等，所以并不是以SMPL-X和FLAME这样的几何先验作为区分的。

### Reconstruction

​	在重建部分，我们的努力是为了构造一个表示3D信息的参数化函数$\mathcal{F}_{\theta}(\cdot)$，然后根据一个投影变换$\Pi $，来得到2D图像$I=\Pi[\mathcal{F}_{\theta}]$。参数化函数可能有许多形式，例如NeRF，SDF，Occupancy field，3DGS等。

#### Head

​	我们先以单目视频中重建人头为例。对于一个单目视频，可以先用面部追踪器得到每一帧的FLAME系数等信息，继而得到FLAME mesh。这个面部追踪器可能是一个feed-forward的神经网络（[DECA](https://github.com/yfeng95/DECA)，[EMOCA](https://github.com/radekd91/emoca)），也可能是一个基于优化的追踪器（[metrical-tracker](https://github.com/Zielon/metrical-tracker)，[VHAP](https://github.com/ShenhanQian/VHAP/tree/main)），继而得到每一帧的FLAME mesh。

​	CVPR‘2022的[Neural Head Avatar](https://github.com/philgras/neural-head-avatars/tree/main)采取的是一种比较直接的办法，用MLP直接预测FLAME上的点的颜色以及偏移，然后直接光栅化：	

<center>
    <img src='/images/avatar/avatar_1.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


这样虽然很直观，但由于其表达仍然是mesh的，对于发丝等细节就会看到明显的冰片状结构。而一些基于NeRF的工作，这些工作由于基于backward-mapping式的体渲染，如果仍然想借助FLAME的几何信息，就需要构造一些特别的设计。当然，也可以不作设计，例如CVPR'2021的[NerFace](https://gafniguy.github.io/4D-Facial-Avatars/)，就是单纯的把追踪出来的系数当作某种condition，直接去训辐射场就好了：

<center>
    <img src='/images/avatar/avatar_4.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


而一些工作，例如CVPR’2023的[INSTA](https://github.com/Zielon/INSTA/tree/master)里，ray-casting是在canonical space里做的，然后将表情同时作为MLP的输入，使得在不同表情下，可以解析出不同的颜色，从而实现动起来的效果。

<center>
    <img src='/images/avatar/avatar_2.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


这样做的假设是在脸部追踪时，假定人头并不发生转动，而是相机在动。这对于人头来说是可以的，对于人体来说就比较有限了。而更符合直觉的一种做法是在deformation space下做ray-casting，即在一个数据集里寻常表情和姿态下的mesh上做，然后击中表面后，去解算击中的这个点来自于canonical space下的哪个点；再用MLP预测解算后的坐标的属性，最后作体渲染。CVPR‘2022的[IMAvatar](https://github.com/zhengyuf/IMavatar)就是围绕这一点做设计的：

<center>
    <img src='/images/avatar/avatar_3.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


这篇工作处理的对象是其作者先前探索的一种表示mesh的隐式场，所以比较复杂。如果是FLAME显式的mesh，那直接作LBS的逆过程即可，例如ToG'2023的[BakedAvatar](https://github.com/buaavrcg/BakedAvatar)，这篇工作在此基础上构造一系列等值面（Isosurfaces），最后将他们都烘培成mesh，是一个非常系统并且效果非常炫酷的工作。

​	而3DGS这种forward-mapping式的渲染在表达数字人时则具有天然的好处，因为我们可以天然的把3DGS定义在mesh上，然后高斯球就可以随着mesh的驱动而自由的移动。

> 值得指出的是，其实在3DGS流行之前，就已经有了借助点来建模人头的工作，例如CVPR’2023的[PointAvatar](https://github.com/zhengyuf/PointAvatar)和SigAsia'2023的[NPVA](https://github.com/conallwang/NPVA)。

高斯球绑定在mesh上的定义方式有很多，例如根据UV坐标的[FlashAvatar](https://ustc3dv.github.io/FlashAvatar/)（CVPR’2024），根据面片上局部坐标系的[GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)（CVPR‘2024）。一些工作出于自己的目的，可能会对高斯球做一些特殊的约束或者规定，例如绑在mesh表面，强制为球形或者圆盘形。但不管哪些，基本都会将高斯球的旋转属性定义为次级旋转（Secondary Rotation），即相对于所在的三角面片的旋转，这很大程度上是必须的。这个变换可以是按照边长计算法线方向，然后按照一条边的方向再计算垂直的一个轴。

> 但如果有涉及到法线贴图的情况，则需要一种特别的变换。这种变换考虑UV空间下的顶点坐标，记作TBN（tangent-bitangent-normal） transformation。简单来说就是要让扰动法线像纹理图到三维空间一样展开。如果不这样做，法线贴图会错位。

​	有些更细节的会让高斯球的尺度也跟三角面片的一些属性（如面积，边长）成正比。特别地，ICLR'2025的[SurFhead](https://openreview.net/forum?id=1x1gGg49jr)提出了一种特别的拉伸高斯球尺度的策略。

<center>
    <img src='/images/avatar/avatar_5.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


这种建模方式其实就是从单目视频中重建人头的版本答案了，所以后续的一些工作只能进一步探索这个任务的边界，例如ICLR'2025的[Gaussian Head & Shoulders](https://openreview.net/forum?id=HtbqsbNw9c)探索了在重建时将肩膀部分融合进去（这个事情说难不难说简单不简单，很微妙），CVPR'2025的[FATE](https://zjwfufu.github.io/FATE-page/)尝试用预训练的模型先验来补全重建人头中侧视图和后视图的缺失，以及CVPR'2025的[HRAvatar](https://arxiv.org/abs/2503.08224)应用预训练的估计albedo的模型来从单目视频中实现可重打光的人头。

​	从单目视频中重建人头，这个任务的一个特点是数据集非常容易获取，用相机，智能手机录制一段视频，甚至是互联网的一些演讲视频都可以拿来进行拟合。但其有一个很本质的问题，就是单目视频下估计出来的形状和表情是耦合的，同时FLAME模板也很难拟合特定的受试者的细节。

​	这就牵扯到一个“personalized blendshape”的问题，即在训练时优化预处理得到的表情系数和姿态系数的同时，还需要调整对应的blendshape。最简单的做法就是让FLAME的blendshape是可学习的，而Sig'2024的[GaussianBlendshape](https://gapszju.github.io/GaussianBlendshape/)进一步将高斯球的其他属性也推广成了blendshape，同时他们的后续工作[RGBAvatar](https://gapszju.github.io/RGBAvatar/)（CVPR‘2025）效果也非常的炫酷，实现了高质量的实时重建。在NeRF的时代，也有一篇在ToG'2022的工作[NeRFBlendshape](https://ustc3dv.github.io/NeRFBlendShape/)是基于这样的思想的。这种办法的角度是在给定驱动信号（例如ARKit的blendshape系数，FLAME系数下），通过用一个优化出来的新blendshape来替代传统的blendshape，继而取得更好的效果。同时，单目视频下的结果也往往在新表情和姿态下，容易出现伪影，在3DGS的表达下即是那些针状伪影。同时，由于“personalized blendshape”，单目视频条件下训出来的人头做cross-reenactment时也比较容易出现瑕疵。

​	当数据源是用专业设备拍摄的多视角数据集时，由于拍摄质量的提高，可以使用更多的手段来提高渲染质量。例如CVPR'2024的[Gaussian Head Avatar](https://yuelangx.github.io/gaussianheadavatar/)和SigAsia'2024的[NPGA](https://simongiebenhain.github.io/NPGA/)，其都会将3DGS渲染的结果再过screen-space CNN来扩大分辨率和补充细节：

<center>
    <img src='/images/avatar/avatar_6.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


由于整个网络都只针对这一个ID服务，所以其效果可以非常好。另一种方式是用属性图（Attribute Map）来表示高斯属性，就像mesh中的纹理图（Texture Map），例如CVPR'2024的[RGCA](https://shunsukesaito.github.io/rgca/)：

<center>
    <img src='/images/avatar/avatar_7.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


用这种属性图的方式，可以把无序的高斯点云给结构化。但这种结构化是要付出代价的，由于这种通过网络输出的属性图的方式，整张属性图的分辨率/细节程度是固定的，为了表示某些细节就需要增加全局的尺寸，所以会多用一些高斯点。例如RGCA里的属性图的大小其实就是$1024\times1024$，每一个texel上都表示一个高斯点，即接近一百万个高斯点。这里因为作者们重做了UV布局，使其完全铺满了UV空间，进一步提高了有效率。

#### Body

​	而在人体领域，发展轨迹是一样的。不过人体由于其更复杂的姿态空间和拓扑，神经网络的加入某种程度上就成了必然。在3DGS兴起以来，有相当多的从单目视频中重建人体的工作，他们轮番的将3DGS与先前的各种设计（多分辨率哈希储存特征，UV展开图）结合起来。例如CVPR'2024的[HUGS](https://machinelearning.apple.com/research/hugs)，[GaussianAvatar](https://huliangxiao.github.io/GaussianAvatar)，[ASH](https://vcai.mpi-inf.mpg.de/projects/ash/)等。

<center>
    <img src='/images/avatar/avatar_11.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	然而单目视频出来的人体质量确实不够高，很多时候在新的姿态序列下也会有很严重的伪影。而有些工作倾向于在重建时加非常复杂的正则项，例如ECCV'2024的[ExAvatar](https://mks0601.github.io/ExAvatar/)：

<center>
    <img src='/images/avatar/avatar_13.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


这样做可以保证在新视角和新姿态下的结果尽量合理，但也让高斯其实事实意义上非常接近mesh texture。

​	而一些非常高质量的数据集，例如[ActorHQ](https://actors-hq.com/)和一些采集的多视角数据集，能够带来更好的视觉效果。其中影响力最高的当属CVPR‘2024中[Animatable Gaussians](https://animatable-gaussians.github.io/)：

<center>
    <img src='/images/avatar/avatar_12.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

Animatable Gaussians先是从接近A-pose的多视角图片中拟合SDF，得到一个mesh。然后将SMPL-X的lbs weights扩散到这个SDF得到的模板上，可以看到这样学习到的模板天然具有一些衣服的结构。然后一个很刁钻的操作，是构造“Pose-dependent Map”，他将A-pose下前后投影得到的点，在LBS以后得到的位置，颜色映射到这个图上。然后用一个受视角调制的StyleUNet来将这个依赖于姿态的输入图迁移成高斯属性图。这高斯属性图也是前后视角的，直接贴回去就好了，不需要什么UV映射之类的，然后由于这个带衣服的参数化模板的权重也是继承自SMPL-X的，直接驱动就好了。然后由于这依赖于姿态的输入源和相机位置的调制，可以让整个avatar很好的拟合数据集里的高光和褶皱以及其动态变化。

​	最近CVPR’2025上的一篇工作[TaoAvatar](https://pixelai-team.github.io/TaoAvatar/)进一步对Animatable Gaussian做了很多工程上的优化，并且将其部署到了Apple Vision Pro里。有幸实际体验过一次，感觉效果非常好，电子女友企划或将又进一步。

<center>
    <img src='/images/avatar/avatar_28.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

而meta也进一步将人体推广到多视角的视频数据集下，他们的[relightable-full-body-gaussian-avatar](https://arxiv.org/abs/2501.14726)，搭建了一个512个相机的拍摄圆顶，配有1024个可控光源，可谓是不计成本。其建模方式和先前的RGCA是一样的。

<center>
    <img src='/images/avatar/avatar_31.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

总的来看，其实影响人体重建效果的最大因素还是数据集。采集的越精细，重建的效果越好。

### Generation

​	对于广义的生成问题，其是将数据$x$的边缘分布分解为条件概率$p(x|z)$和先验概率$p(z)$在$z$上的积分：
$$
p\left( x \right) =\int{p\left( x|z \right) p\left( z \right) \mathrm{d}z}
$$
一般来说，$z\sim p(z)$通常都是高斯先验。但对于数字人的任务，由于数据集的形式，$z$可以是更加“结构化”的变量，因为采集方式主要依靠昂贵的多视点相机以及光场：

<center>
    <img src='/images/avatar/avatar_10.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


例如，潜在分布$z$可以表述为身份编码$z_{\mathrm{id}}$，视角编码$z_{\mathrm{c}}$，表情（人头）或形状（人体）编码$z_{\mathrm{e}}$，以及姿态编码$z_{\mathrm{p}}$。我们可以自然的假设视角编码$z_{\mathrm{c}}$与身份，表情（形状），姿态都无关；表情（形状）$z_{\mathrm{e}}$和姿态$z_{\mathrm{p}}$在身份$z_{\mathrm{id}}$下是独立的。这个假设很自然，因为用什么视角拍摄确实跟受试者和其状态无关，然后同一个人可能确实存在对表情和姿态的某种偏好，以及视角编码$z_{c}$其实对于多视角数据集的采集方式，其相对简单，不需要额外建模。基于此，边缘分布可以分解为：
$$
p\left( x \right) =\int{p\left( x|z_{\mathrm{id}},z_{\mathrm{c}},z_{\mathrm{e}},z_{\mathrm{p}} \right) p\left( z_{\mathrm{e}}|z_{\mathrm{id}} \right) p\left( z_{\mathrm{p}}|z_{\mathrm{id}} \right) p\left( z_{\mathrm{c}} \right) p\left( z_{\mathrm{id}} \right) \mathrm{d}z_{\mathrm{id}}\mathrm{d}z_{\mathrm{c}}\mathrm{d}z_{\mathrm{e}}\mathrm{d}z_{\mathrm{p}}}
$$
但好在我们不需要计算这个分布本身，只需要从中采样。而如何选取怎样的策略来建模以及采样这些隐变量，就是艺术了。

> “This is art, Mr. White!”——Jesse Pinkman

#### Head

​	有很多关于人头工作在此基础上进行，例如ICCV'2023的[Preface](https://syntec-research.github.io/Preface/)和Sig’2022的[MoRF](https://dl.acm.org/doi/10.1145/3528233.3530753)，他们构造了一个只关于ID的条件NeRF，来构造人头先验，相当于只建模$z_{\mathrm{id}}$，其中的$z_{\mathrm{id}}$采样自高斯分布，这导致其不能驱动。ECCV'2022的[MoFaNeRF](https://arxiv.org/abs/2112.02308)同时建模了$z_{\mathrm{id}}$和$z_{\mathrm{e}}$，不考虑姿态，直接采样自参数化人脸张量分解得到的系数。CVPR'2022的[HeadNeRF](https://hy1995.top/HeadNeRF-Project/)中的$z_{\mathrm{id}}$和$z_{\mathrm{e}}$则采样自高斯分布。Sig'2024的[VRMM](https://arxiv.org/abs/2402.04101)在此基础上使用更复杂的设备采集了光照（可以理解为建模了$z_{\mathrm{light}}$，实现了重打光。这些工作往往都需要一个大规模的多人的多视角数据集，其中Meta的系列工作，由于其对拍摄的数据展开了材质，所以其建模$z_{\mathrm{id}}$的时候往往不是高斯分布或者形状系数之类的，而是颜色映射构成的位置图和其平均材质位置图，如其Sig‘2022的[AVA](https://research.facebook.com/publications/authentic-volumetric-avatars-from-a-phone-scan/)所示：

<center>
    <img src='/images/avatar/avatar_8.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


以上的这些工作他们都忽略了姿态$z_{\mathrm{p}}$，因为在多ID的框架上实现对姿态分布的学习，会很困难，不如把数据集的人脸都对齐好。

​	当有3DGS这种mesh友好的表达以后，一种更直接的办法应运而生。即对多视角视频数据集直接进行追踪，例如作FLAME tracking，这样其实就直接采样到了$z_{\mathrm{e}}$和$z_{\mathrm{p}}$。例如3DV’2025的[HeadGAP](https://headgap.github.io/)，而由于Meta Codec Lab团队之前的数据集（Ava256）本身就有tracking好的mesh以及UV展开，非常适合迁移到GS表达上，他们的具有重打光能力的3DGS参数化人头也很快出炉了（[URAvatar](https://arxiv.org/abs/2410.24223)，SigAsia'2024）：

<center>
    <img src='/images/avatar/avatar_9.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

由于这些数据集采集成本其实很高，这些训练出的模型并不算是那么“生成”，因为在这种结构化数据集下，整个数据集的分布其实相当的简单。他们的损失监督只是重建损失，并不需要对抗训练或者变分估计这样。这些模型更大的功用在于提供一个参数化模型，来拟合给定的单张或多张图片。

​	考虑到这些多视角数据集需要复杂的预处理以及高昂的采集成本，还有一些工作是用CG管线渲染出的人头来制作人头先验。例如CVPR'2025中的[SynShot](https://zielon.github.io/synshot/)和[GASP](https://microsoft.github.io/GASP/)。而有的工作专注于赋能CG管线，如SigAsia‘2024的[GauFace](https://dafei-qin.github.io/TransGS.github.io/)：

<center>
    <img src='/images/avatar/avatar_19.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	另一条线的工作也致力于从非结构化的数据集里构造人头分布，最经典的就是CVPR'2022的[EG3D](https://github.com/NVlabs/eg3d)，其估计了[FFHQ](https://github.com/NVlabs/ffhq-dataset)数据集里的人脸图片对应的相机位姿，相当于一起估计由$z_{\mathrm{c}},z_{\mathrm{id}},z_{\mathrm{e}},z_{\mathrm{p}}$构成的复杂分布。这是非常复杂的，所以其设计了一个精巧的GAN结构来进行估计。后来CVPR‘2023的[PanoHead](https://github.com/SizheAn/PanoHead)补全了EG3D的侧视图和后视图的效果，其一定程度上抵消了三平面的歧义，同时设计了复杂的估计侧视图和后视图下人头姿态的管线。ECCV'2024的[SphereHead](https://github.com/lhyfst/SphereHead)进一步提高了full-head的效果，有着一些深入的观察和独特的设计。

<center>
    <img src='/images/avatar/avatar_14.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	PanoHead设计了tri-grid，即构造了几组事实意义上具有不同“深度”的tri-plane，但这仍然不能完全消除较少监督时的镜像伪影。因为在tri-plane的查询时，另外两个平面的对称点，他们取到的特征总是一样的，带来了天然的耦合。而SphereHead里把卷积输出的特征图分别绕成一个半圆，一个整圆，和一个球面，让特征查询完全在极坐标下做，彻底消除了可能的耦合。

​	这些工作生成的只能是静态的人头，诚然可以应用StyleGAN里latent vector transverse的把戏让他们动起来，但其毕竟有限。而想让他们动起来其实就是如何建模$z_{\mathrm{e}}$和$z_{\mathrm{p}}$，CVPR’2023的[Next3D](https://mrtornado24.github.io/Next3D/)显式的对FFHQ里的图片进一步估计了FLAME mesh以及眼球的gaze。将mesh显式的整合进tri-plane的流程里，非常的直接。

<center>
    <img src='/images/avatar/avatar_15.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


这使得Next3D训练好以后确实可以输入一组预先追踪好的FLAME mesh来进行驱动。然而其本身只是在FFHQ上训练的，其实并没有专门处理时序或者动作上的事情。而[Portrait4D](https://github.com/YuDeng/Portrait-4D)以及其v2是将这个范式推广到了单目视频的非结构化数据集（VFHQ），其框架借鉴自ToG‘2023的[Live3DPortrait](https://research.nvidia.com/labs/nxp/lp3d/)，整体流程比较的复杂：

<center>
    <img src='/images/avatar/avatar_16.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


其第一步首先是在FFHQ上训练了一个用mesh隐式驱动的3D-aware GAN，得到了一个有效的生成器。然后用VFHQ追踪得到的FLAME系数输入生成器，得到一系列构造好的仿真数据，上图展示其实只展示后者。这个框架的一个核心设计是那两个针对图片的编码器$E$，其中$E_{detail}$只是一些卷积，而$E_{global}$是参考deeplab_v3的一个有不同尺寸的空洞卷积的编码器，这两者得到的其实都是图像的特征。而他们最后居然可以被$G_T$，其设计是一个ViT，给转换成canonical下的tri-plane，确实比较惊讶。

​	而从EG3D里进行inversion，从而实现“single-image reconstruction”等也曾被人广泛关注。其核心逻辑即PTI，如WACV’2023上一篇[工作](https://3dgan-inversion.github.io/)同时考虑了在PTI下优化相机位姿，CVPR'2023的[HFGI3D](https://github.com/jiaxinxie97/HFGI3D/)，以及NeurIPS'2024的[DualEncoder](https://berkegokmen1.github.io/dual-enc-3d-gan-inv/)。

​	然而，这种3D-aware GAN的工作其实对位姿编码$z_{\mathrm{p}}$做了一个很强的假设。因为其训练本身只能在对齐后的FFHQ下进行（不然效果会比较糟糕），而FFHQ对齐的逻辑其实是预测人脸关键点后，估计一个仿射变换将人脸“转平”，直观来说就是把图片处理成每个人眼睛都水平的状态。所以这导致EG3D这一条线下来的结果其实都比较有限。

​	以上说的3D-aware GAN的工作都是基于NeRF的，在人头里也有将其推广到3DGS版本，即SigAsia'2024的[GGHead](https://github.com/tobias-kirschstein/gghead/tree/master)，其中做了许多细节的设计来让3DGS可以在对抗训练下稳定存在。

<center>
    <img src='/images/avatar/avatar_27.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


#### Body

​	很可惜对于人体来说，构造那样有效的结构化数据集太困难了，主要是人体的多样性（服饰）太复杂了。所以人体中沿着EG3D那样走3D-aware GAN路线的，也是依靠一种类似FFHQ的数据集——[SHHQ](https://stylegan-human.github.io/data.html)。

​	由于人体比人头不管是几何还是材质上都复杂的多，所以在实现时往往要联合许多技术。例如在监督人体图片本身时还要对人脸部分单独监督，以及一些驱动变形的设计（即必须要建模$z_{\mathrm{p}}$）。这一系列的相关工作有ICCV'2023的[AG3D](https://github.com/zj-dong/AG3D)和[GETAvatar](https://getavatar.github.io/)，ICLR‘2023的[EVA3D](https://github.com/hongfz16/EVA3D)，NeurIPS’2023的[PrimDiffusion](https://frozenburning.github.io/projects/primdiffusion/)，和CVPR‘2024的[GaussianShellMaps](https://rameenabdal.github.io/GaussianShellMaps/)。

<center>
    <img src='/images/avatar/avatar_17.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


整体上这些工作输出的人都是那种很模特的人，而且质量其实仍然受限。以及由于人体的复杂性，能很难进行inversion/fitting。

​	而除此以外，也存在一些较大规模的3D人体的数据集，静态的有[THuman](https://github.com/ZhengZerong/DeepHuman/tree/master/THUmanDataset)系列，动态的有[MVHumanNet](https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet)，后者虽然有动态，但图片质量整体比较混沌，并且SMPL-X的估计好像有些问题。MM'2024上的[E3Gen](https://olivia23333.github.io/E3Gen/)确实尝试在THuman上训一个结构化的生成模型，但效果其实不是很好。

​	然而如果放宽条件，其实可以有一些非常清奇的思路，例如CVPR’2024的[GPS-Gaussian](https://github.com/aipixel/GPS-Gaussian)：

<center>
    <img src='/images/avatar/avatar_18.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

这个工作很新颖，他首先在静态数据集（THuman和Twindom）上训练人体的先验。他想训的先验其实只有几何（下面的深度估计模块以及上面的高斯协方差），而高斯的颜色完全来自输入的两个视角。这其实就直接回避了人体数据集在外貌上不够多样的困境。而高斯的位置就来自于双目深度的直接估计。当然，这种办法其实只能做free-view rendering，驱动本身还是得靠拍摄的人实时做动作。

​	但本质上，采集的人体数据集在外貌多样性上的不足，使得人们只能构造关于几何的可泛化模型。而解决外貌泛化性的有力手段，就是下面要说的引入强力的先验模型。

## Prior-guided

​	上述的工作可以说已经从已有数据集里尽数探索了效果，而后来，大规模预训练模型的出现其实重塑了人们的工作流。

### Stable Diffusion

​	其中影响最大的就是预训练的文生图模型，将其结合得分蒸馏采样损失，有许多从文本中生成avatar的工作。人头上有Sig'2024的[HeadArtist](https://arxiv.org/abs/2312.07539)和ECCV’2024的[HeadStudio](https://github.com/ZhenglinZhou/HeadStudio)，人体上有3DV‘2024的[TADA](https://github.com/TingtingLiao/TADA)，CVPR‘2024的[HumanNorm](https://humannorm.github.io/)和[HumanGaussian](https://alvinliu0.github.io/projects/HumanGaussian)。这些工作的趋势后面逐渐变成了微调SD模型，使其能监督深度，法线，语义等。例如CVPR'2024的[DreamAvatar](https://yukangcao.github.io/DreamAvatar/)，AAAI’2024的[AvatarVerse](https://avatarverse3d.github.io/)和[AvatarStudio](https://github.com/magic-research/avatarstudio/tree/main)。

<center>
    <img src='/images/avatar/avatar_20.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


但这一范式生成出来的人大多只是数据集中多次出现的著名人物和一些动漫IP，并不能生成特定个性化的人体。

​	有些工作会加入图片作为条件，如3DV'2024的[TeCH](https://github.com/huangyangyi/TeCH)，[AvatarBooth](https://zeng-yifei.github.io/avatarbooth_page/)，和SigAsia'2024的[PuzzleAvatar](https://github.com/YuliangXiu/PuzzleAvatar)。后两个工作则选择用随机采集的同一个ID的图片先去个性化一个booth，然后再做SDS。

​	但依赖SDS还是太慢了，随着后面3D生成的范式转向multi-view diffusion，继而转向large reconstruction model，avatar建模的范式也开始了转变。例如CVPR’2025的[GAF](https://tangjiapeng.github.io/projects/GAF/)，通过微调一个多视角的SD2.1，以法线图作为条件，来生成伪数据帮助单目视频重建。[CAP4D](https://felixtaubner.github.io/cap4d/)通过微调SD2.1，同时加入许多与3DMM相关的表示作为条件，实现了一个效果非常好的可变形的多视角扩散模型。通过反复推理得到的模型，可以构造一个多视角数据集，用于训练GaussianAvatars。这两个工作都需要对多视角人头的数据集进行FLAME tracking，其实tracking的过程就已经很“labor-intensive”了。

<center>
    <img src='/images/avatar/avatar_21.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	而在人体里，ECCV‘2024的[AvatarPop](https://www.nikoskolot.com/avatarpopup/)和CVPR’2025的[PSHuman](https://penghtyx.github.io/PSHuman/)，前者通过用renderpeople微调，后者用Thuman微调，取得了相当好的结果。

<center>
    <img src='/images/avatar/avatar_22.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	这些结果在没有SD2.1这样的拥有海量数据先验的模型之前，是不可能的。以及另一个很关键的区别是，SD2.1这样的先验非常的强（得益于其scaling），并不像基于StyleGAN那样需要将图片都对齐到某个规范化空间中，这大大加强了其可迁移性。

​	在另一条路线上，给Stable Diffusion进行微调，增加人体和人头相关的先验来让其驱动也如火如荼。一开始是[Animate Anyone](https://humanaigc.github.io/animate-anyone/)提出的一个非常有效的ReferenceNet的框架，将参考图片用一个对偶的网络也进行处理，然后逐层的注入进扩散模型中。

<center>
    <img src='/images/avatar/avatar_32.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

后续的一些工作如ECCV'2024的[Champ](https://github.com/fudan-generative-vision/champ)在2D-关键点的基础上继续加入了SMPL-X等控制，[hallo](https://github.com/fudan-generative-vision/hallo)系列将这种设计在人头驱动上也做了一遍。这些工作都基于扩散模型，而有一篇非常特殊的工作，其作者非常熟悉人脸关键点这一套技术。而且人头本身又没有那么多铰链的结构，拓扑相对比较固定。于是就有了一篇diffusion-free的人脸驱动的工作[LivePortrait](https://github.com/KwaiVGI/LivePortrait)。

​	值得一提的是，随着视频生成模型的兴起，也有了一些专门在数字人方面进行训练的视频生成模型，如[OmniHuman](https://omnihuman-lab.github.io/)和[DreamActor](https://grisoon.github.io/DreamActor-M1/)。这些工作和上面刚说的另一条路线的工作，虽然不是3D的，已经取得了十分高质量的结果。甚至可以说，在没有free-view rendering的需求下，2D生成确实已经是答案了（即使他们的推理速度确实堪忧）。

### Universal Encoder

​	自大规模图像编码器问世以来，利用其提取出的特征来帮助下游任务这一策略开始被广泛采用。

在NeurIPS‘2024的[GAGAvatar](https://github.com/xg-chu/GAGAvatar)中，就将DINOv2的特征进一步进行处理，然后反投影到3D空间上作为3DGS的特征：

<center>
    <img src='/images/avatar/avatar_23.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	最近的一些工作也采用类似的思路，处理人体[LHM](https://github.com/aigc3d/LHM)和处理人头[LAM](https://github.com/aigc3d/LAM)将收集的大量数据集，在进行追踪以后都用预训练的编码器打成特征，来作为条件。由于[Saipens](https://github.com/facebookresearch/sapiens)是专门在人体上进行预训练的编码器，所以LHM用Saipens处理整张人体图片，而用DINOv2额外处理头部。CVPR’2025的[IDOL](https://github.com/yiyuzhuang/IDOL)也是这样，只不过没有额外处理人脸部分。

<center>
    <img src='/images/avatar/avatar_24.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

​	一个有趣的事实是，实际上LHM比IDOL少用了许多高斯点。这其实类似于一个硬币的正反面，LHM的设计里，每一个点的位置都会位置编码以后，作为token来被transformer处理，这带来了大量的计算负担，但也允许每个位置被解析出更复杂的属性。而IDOL中的设计是用额外的卷积层来将transformer处理成的token再降通道拉分辨率的拉高到一个很高的程度，然后进行采点，这样虽然能采很多点，但属性的分布最终还是受卷积的归纳偏置影响。

> 一个事实是，IDOL的数据集来自于其用THuman微调的multi-view champ。作者先用先进的图像生成模型绘制了许多AI风格的正面图像，然后用multi-view champ得到他们不同view下的样子。所以其本身的数据集就已经蒸馏自强大的预训练模型了。然而与LAM和LHM横向比较，其泛化性貌似又和直接用这些AI生成的图像训练关系不大。

​	而[Avat3r](https://tobias-kirschstein.github.io/avat3r/)联合了dust3r的编码器，直接估计pixel-wise的3DGS，这使得其数据集并不需要做繁杂的追踪（但人头图片还是需要裁剪到中央的），也不需要依赖参数化模板：

<center>
    <img src='/images/avatar/avatar_25.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


当然，这能存在的前提，是因为meta的ava256数据集有80个相机位置，其实是相当稠密的。而不依赖于参数化模板其实也不是说一定是件好事，由于没有类似FLAME mesh那样的显式驱动，如果想让得到的3DGS做驱动，需要改变其学习到的表情变量，重新推理一遍整个网络。

​	这些工作的惊人之处在于，通用编码器其实为很多域外的数据都提供了泛化。例如对于上面的Avat3r，只用暗室下（色调偏绿）拍摄的数据集训练的模型，输入一些AIGC绘制的人头，甚至是石膏雕塑，也能有好的效果。以及像LHM和IDOL，这种方式也帮助实现了先前关于人体的3D-aware GAN系列无法完成的fitting和inversion。

<center>
    <img src='/images/avatar/avatar_29.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

<center>
    <img src='/images/avatar/avatar_30.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


> "有时候要做成一件事，还有第二条路。"——成龙历险记

### InstructPix2Pix

​	InstructPix2Pix是一个很强大的可以用文本编辑图像的预训练模型，有一些工作将其应用到编辑重建好的三维人头中。这个想法很直接，但实现起来并不容易。其一是由于扩散模型的不确定性，每一帧被编辑的结果都会有显著的不同，其二是由于3DGS并不是基于光线的，所以没法像NeRF的时候一样通过将编辑前后的像素点搅拌在一起来实现协调的更新。例如SigAsia‘2024的[Texttoon](https://songluchuan.github.io/TextToon/)，其在CLIP的特征空间上监督图片的小patch之间的相关性，而不是直接在像素空间：

<center>
    <img src='/images/avatar/avatar_26.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

而3DV'2025的[GaussianAvatarEditor](https://xiangyueliu.github.io/GaussianAvatar-Editor/)则仿照instruct-NeRF2NeRF，逐渐的用编辑之后的图像替代数据集，同时用一个鉴别器来作对抗训练，辅助时序上的连贯性。SigAsia’2024上的[PortraitGen](https://ustc3dv.github.io/PortraitGen/)，则尝试了更多的图像编辑的模型（例如neural style transfer，IC-Light）。

#### Misc

其实能用到的先验工具还有更多，例如可以用LivePortrait做表情的增强。用多模态大模型进行数据标注，用SAM来得到语义信息，用Depth-anything得到深度的良好估计，用Intrinsic-anything来估计albedo。可以说这些强力的预训练模型彻底改变了我们对数据驱动的认识。找到一些合适的角度进行微调，总能起到一些四两拨千斤的功用。

## End

​	这次的梳理其实比较仓促和随意，因为写这篇blog的动机确实很大程度上是因为服务器过载挂了。

​	自从去年8月份开始，我进组学习已经快9个月了。实验室提供了相当舒适和轻松的科研环境，在导师专业的指导下我也终于发出了一篇不带星的独立论文。虽然解决的问题并没有那么重要，但确实是一次圆满的训练。在实验室里，每次跟同门进行交流，可以说是受益匪浅。一开始工位比较紧张，就在相机阵列在的大屋子附近拉了张长桌子，在桌子和旁边光学平台边上插空坐。

<center>
    <img src='/images/avatar/206.jpg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

那时候的东西乱糟糟的放着颇有点早年间的研究院的意思，后来有工位了，就变成对着那台能当白板用的大电脑讨论了。那时候如果没有那些交流，这篇blog两天内其实也码不完。

​	这9个月是非常神奇的经历，除了能够稍微一瞥digital avatar这个方向，还有相当多值得纪念的事情：去闪击上海看全息演唱会；第一次目击一个PhD candidate摘掉candidate的过程；过年的那几天吃麦吃伤了；焦灼地等论文以及面试结果；去China3DV打野；以及那赛博版“仙人抚我顶，结发受长生。”。谁知道接下来会发生什么呢？

<center>
    <img src='/images/avatar/end.jpg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>
