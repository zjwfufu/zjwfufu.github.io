---
title: Denoising Diffusion Probabiblistic Model
mathjax: true
date: 2022-11-23 19:15:21
tags: [深度学习]
categories: 深度学习
---

​	扩散模型变得越来越流行，但它的原理并不像生成对抗网络一样那么的“明显”。于是在这篇blog里进行一下整理和学习。

<!--more-->

​	扩散模型给我的第一感觉就像，元气骑士里的许愿池……

### Forward Process

​	现在有许多介绍扩散模型的资料，他们都会说出一个形象的理解：一张图片，对它逐渐加噪。然后逐步去噪。但这直观的理解并不是“很有用”。为了更深入的理解，还是需要读懂扩散模型中的那些数学表示：

​	考虑初始的数据分布$\mathbf{x}_0\sim q\left( \mathbf{x} \right) $，那么向数据中逐步加入高斯噪声的过程，可以写作：
$$
q\left( \mathbf{x}_t|\mathbf{x}_{t-1} \right) =\mathcal{N} \left( \mathbf{x}_t;\sqrt{1-\beta _t}\mathbf{x}_{t-1},\beta _t\mathbf{I} \right)
$$
​	上式实际上给出：在已知$\mathbf{x}_{t-1}$的条件下，$\mathbf{x}_{t}$的分布服从一个均值为$\sqrt{1-\beta _t}\mathbf{x}_{t-1}$，方差为$\beta _t$的高斯分布。直观理解上，就是将一张图片，轻微的上下震荡，从而使得图片略微失真。这里的$\beta _t$可以理解为一个“调度器”，因为如果这一过程是完全线性的，研究人员指出图像的信息会消失的太快。
$$
q\left( \mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_T|\mathbf{x}_0 \right) =\frac{q\left( \mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_T,\mathbf{x}_0 \right)}{q\left( \mathbf{x}_0 \right)}
\\
=\frac{q\left( \mathbf{x}_T|\mathbf{x}_0,\mathbf{x}_1,..,\mathbf{x}_{T-1} \right) q\left( \mathbf{x}_0,\mathbf{x}_1,..,\mathbf{x}_{T-1} \right)}{q\left( \mathbf{x}_0 \right)}
$$
​	实际上这个过程是一个一阶马尔可夫链，$t$时刻只与$t-1$​时刻有关。那么我们可以使用马尔可夫假设化简分子，继续展开，得：
$$
q\left( \mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_T|\mathbf{x}_0 \right) =\frac{q\left( \mathbf{x}_T|\mathbf{x}_{T-1} \right) q\left( \mathbf{x}_{T-1}|\mathbf{x}_{T-2} \right) ...q\left( \mathbf{x}_0 \right)}{q\left( \mathbf{x}_0 \right)}
\\
=\prod_{t=1}^T{q\left( \mathbf{x}_t|\mathbf{x}_{t-1} \right)}
$$
​	加噪的过程，并不需要一步步迭代相乘。并且，注意到：如果需要从一般的高斯分布$\mathcal{N} \left( \mu ,\sigma ^2 \right) $可以由标准正态分布$\mathcal{N} \left( 0,1 \right)$作仿射变换得到。这一操作被称为“重整化”。所以，之前的加噪过程可以写为：
$$
\mathbf{x}_t=\sqrt{1-\beta _t}\mathbf{x}_{t-1}+\sqrt{\beta _t}\mathbf{\epsilon }_{t-1}
\\
\mathbf{x}_{t-1}=\sqrt{1-\beta _{t-1}}\mathbf{x}_{t-2}+\sqrt{\beta _{t-1}}\mathbf{\epsilon }_{t-2}
\\
\mathbf{x}_t=\sqrt{1-\beta _t}\left( \sqrt{1-\beta _{t-1}}\mathbf{x}_{t-2}+\sqrt{\beta _{t-1}}\mathbf{\epsilon }_{t-2} \right) +\sqrt{\beta _t}\mathbf{\epsilon }_{t-1}
$$
​	这么写的很凌乱，令$\alpha _t=1-\beta _t,\bar{\alpha}_t=\prod_{i=1}^t{\alpha _i}$。同时，根据方差的性质和线性可加性：
$$
D\left( kX \right) =k^2D\left( X \right) 
\\
\sqrt{\beta _t}\mathbf{\epsilon }_{t-1}\in \mathcal{N} \left( 0,\beta _t \right) ,
\\
\sqrt{\beta _{t-1}}\sqrt{1-\beta _t}\mathbf{\epsilon }_{t-2}\in \mathcal{N} \left( 0,\beta _{t-1}\left( 1-\beta _t \right) \right) 
$$
​	所以可以化简为：
$$
\mathbf{x}_t=\sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2}+\sqrt{\beta_t+\beta_{t-1}-\beta_t \beta_{t-1}} \boldsymbol{\epsilon}_{t-2}
\\
=\sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2}+\sqrt{1-\alpha_t+1-\alpha_{t-1}-\left(1-\alpha_t\right)\left(1-\alpha_{t-1}\right)} \boldsymbol{\epsilon}_{t-2}
\\
=\sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1} \alpha_t} \boldsymbol{\epsilon}_{t-2}
$$
​	进一步，得：
$$
\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\mathbf{\epsilon }
\\
q\left( \mathbf{x}_t|\mathbf{x}_0 \right) =\mathcal{N} \left( \mathbf{x}_t|\sqrt{\bar{\alpha}_t}\mathbf{x}_0,\left( 1-\bar{\alpha}_t \right) \mathbf{I} \right)
$$
​	即给定一个初始图片，就能算出第$t$步的$\mathbf{x}_t$。

### Reverse Process

​	加噪是简单的，问题的关键在于从噪声中恢复原图像，即反向扩散。我们在此处停顿一下：虽然印象里，我们说：给图像加噪，然后去噪。但是实际上我们现在并没有引入任何的参数。到目前为止我们只是推出了一个一阶马尔科夫链的加噪过程。

​	对于反向扩散$q\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right) $,它的分布是未知的。但理论分析证明，连续扩散过程的逆转过程当扩散率（也就是这里的$\beta_t$）很小的时候，逆转过程的分布与正向分布时同分布。（这一定程度上也解释了为什么扩散模型要“forward”很多次）。

​	$q\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right)$很难得到，但从训练的角度来看。我们总是会输入一张原始图片$\mathbf{x}_{0}$，我们对于中间的隐变量的了解完全是由后验给出的。所以后验概率$q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right)$由贝叶斯公式：
$$
q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right) =\frac{q\left( \mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_0 \right)}{q\left( \mathbf{x}_t,\mathbf{x}_0 \right)}
\\
=\frac{q\left( \mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0 \right) q\left( \mathbf{x}_{t-1},\mathbf{x}_0 \right)}{q\left( \mathbf{x}_t,\mathbf{x}_0 \right)}
\\
=\frac{q\left( \mathbf{x}_t|\mathbf{x}_{t-1} \right) q\left( \mathbf{x}_{t-1}|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_t|\mathbf{x}_0 \right)}
$$
​	最后的等号是由马尔科夫链的性质得到的。另一方面，由马尔可夫性，加入$\mathbf{x}_{0}$并不影响概率。由上述的在很小的$\beta_t$时的性质：
$$
q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right) =\mathcal{N} \left( \mathbf{x}_{t-1}|\tilde{\mu}\left( \mathbf{x}_t,\mathbf{x}_0 \right) ,\tilde{\beta}_t\mathbf{I} \right)
$$
​	所以看起来我们只需要估计逆向时每一步的均值和方差即可。我们将上面用贝叶斯公式展开的式子直接代入前向扩散时的分布，我们知道高斯分布：
$$
f\left( x \right) =\frac{1}{\sqrt{2\pi}\sigma}\exp \left( -\frac{\left( x-\mu \right) ^2}{2\sigma ^2} \right) 
$$
​	我们忽略掉前面的归一化系数，这在计算时不影响均值和指数上的方差：
$$
q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right) =\frac{q\left( \mathbf{x}_t|\mathbf{x}_{t-1} \right) q\left( \mathbf{x}_{t-1}|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_t|\mathbf{x}_0 \right)}
\\
\propto \frac{\exp \left( -\frac{\left( \mathbf{x}_t-\sqrt{1-\beta _t}\mathbf{x}_{t-1} \right) ^2}{2\beta _t} \right) \exp \left( -\frac{\left( \mathbf{x}_{t-1}-\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 \right) ^2}{2\left( 1-\bar{\alpha}_{t-1} \right)} \right)}{\exp \left( -\frac{\left( \mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0 \right) ^2}{2\left( 1-\bar{\alpha}_t \right)} \right)}
\\
=\exp \left( -\frac{\left( \mathbf{x}_t-\sqrt{1-\beta _t}\mathbf{x}_{t-1} \right) ^2}{2\beta _t}-\frac{\left( \mathbf{x}_{t-1}-\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 \right) ^2}{2\left( 1-\bar{\alpha}_{t-1} \right)}+\frac{\left( \mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0 \right) ^2}{2\left( 1-\bar{\alpha}_t \right)} \right) 
$$
​	我们整理指数上的二次型，会得到：
$$
=\exp \left( -\frac{1}{2}\left( \left( \frac{1}{1-\bar{\alpha}_{t-1}}+\frac{\alpha _t}{\beta _t} \right) \mathbf{x}_{t-1}^{2}-\left( \frac{2\sqrt{\alpha _t}}{\beta _t}\mathbf{x}_t+\frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}_0 \right) \mathbf{x}_{t-1}+C\left( \mathbf{x}_t,\mathbf{x}_0 \right) \right) \right) 
$$
​	配方后，我们会发现，待估计的均值和方差可以写作：
$$
\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta _t
\\
\tilde{\mu}\left( \mathbf{x}_t,\mathbf{x}_0 \right) =\frac{\sqrt{\alpha _t}\left( 1-\bar{\alpha}_{t-1} \right)}{1-\bar{\alpha}_t}\mathbf{x}_t+\frac{\sqrt{\bar{\alpha}_{t-1}}\beta _t}{1-\bar{\alpha}_t}\mathbf{x}_0
$$
​	这表明，方差是被调度器固定的一个值，我们不用估计它，我们只需估计均值。进一步，由上面重整化的技巧，我们可以把$\mathbf{x}_0$整理成$\mathbf{x}_t$​的形式，可以得到：
$$
\mathbf{x}_0=\frac{\left( \mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\mathbf{\epsilon } \right)}{\sqrt{\bar{\alpha}_t}}
\\
\tilde{\mu}\left( \mathbf{x}_t,\mathbf{x}_0 \right) =\frac{1}{\sqrt{\bar{\alpha}_t}}\left( \mathbf{x}_t-\frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}}\mathbf{\epsilon } \right)
$$
​	上面的一番化简，其实只传达了一个信息：当给出$\mathbf{x}_{0}$时如何估计后验概率$q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right)$。这要求的是，我得有$\mathbf{x}_{0}$，如果没有$\mathbf{x}_{0}$，我们还是不知道如何处理$q\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right)$。但一个很好的视角是：如果我们从炼丹的角度看，给出$\mathbf{x}_{0}$时相当于在训练。没有$\mathbf{x}_{0}$时表示在推断。

​	那么形式化上，我们学习一个模型$p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right) $来逼近$q\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right) $。因为$q\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right) $是高斯的，所以：
$$
p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right) =\mathcal{N} \left( \mathbf{x}_{t-1}|\tilde{\mu}_{\theta}\left( \mathbf{x}_t,t \right) ,\varSigma _{\theta}\left( \mathbf{x}_t,t \right) \mathbf{I} \right)
$$
​	那么根据全概率公式：
$$
p_{\theta}\left( \mathbf{x}_0 \right) =\int{p_{\theta}\left( \mathbf{x}_0,\mathbf{x}_1,...,\mathbf{x}_T \right) \mathrm{d}\left( \mathbf{x}_1,...,\mathbf{x}_T \right)}
$$
​	同时注意到，根据贝叶斯公式：
$$
p_{\theta}\left( \mathbf{x}_0,\mathbf{x}_1,...,\mathbf{x}_T \right) =p_{\theta}\left( \mathbf{x}_T \right) \prod_{t=1}^T{p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right)}
$$
​	以及，根据对$q\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right) $​的分析，实际上这个“逼近”，我们是希望它接近于
$$
\tilde{\mu}\left( \mathbf{x}_t,\mathbf{x}_0 \right) =\frac{1}{\sqrt{\bar{\alpha}_t}}\left( \mathbf{x}_t-\frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}}\mathbf{\epsilon } \right)
$$
​	所以，模型要学习的即是：
$$
\mu _{\theta}\left( \mathbf{x}_t,t \right) =\frac{1}{\sqrt{\bar{\alpha}_t}}\left( \mathbf{x}_t-\frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}}\mathbf{\epsilon }_{\theta}\left( \mathbf{x}_t,t \right) \right)
$$
​	来逼近$\tilde{\mu}\left( \mathbf{x}_t,\mathbf{x}_0 \right)$。

### Variational Inference

​	现在，我们需要推导损失函数。只有完全走过这一过程，我们才能比较透彻的理解为什么扩散模型是在“预测噪声”。在这之前，要先引入“变分推断”的一些知识，因为扩散模型看上去就像层次化结构的变分自编码器。如何看的更早一些，这实际上是模式识别里贝叶斯推断的内容。但很遗憾的是我没学太明白。

​	考虑构造概率分布$p(\boldsymbol{x})$的一个近似分布$q(\boldsymbol{x})$，我们可以试图最小化两者的KL散度来得到$q(\boldsymbol{x})$，KL散度即为：
$$
D_{KL}\left( p||q \right) =\int{p\left( x \right) \log \frac{p\left( x \right)}{q\left( x \right)}\mathrm{d}x}
$$
​	在变分推断中，想要近似的往往是后验概率$p(z|\boldsymbol{x})$，这里$z$称为隐变量。所以对于分布$q(z)$，要最小化的KL散度即为：
$$
D_{KL}\left( q\left( z \right) ||p\left( z|\boldsymbol{x} \right) \right) =\int{q\left( z \right) \log \frac{q\left( z \right)}{p\left( z|\boldsymbol{x} \right)}\mathrm{d}z}
\\
=\int{q\left( z \right) \log \frac{q\left( z \right) p\left( \boldsymbol{x} \right)}{p\left( z,\boldsymbol{x} \right)}\mathrm{d}z}=\int{q\left( z \right) \left( \log \frac{q\left( z \right)}{p\left( z,\boldsymbol{x} \right)}+\log p\left( \boldsymbol{x} \right) \right) \mathrm{d}z}
$$
​	$p(\boldsymbol{x})$与隐变量$z$无关，根据概率密度函数的性质，积分为1。所以上式可以变形为：
$$
\log p\left( \boldsymbol{x} \right) =D_{KL}\left( q\left( z \right) ||p\left( z|\boldsymbol{x} \right) \right) -\int{q\left( z \right) \log \frac{q\left( z \right)}{p\left( z,\boldsymbol{x} \right)}\mathrm{d}z}
$$
​	记：
$$
L\left( q\left( z \right) \right) =-\int{q\left( z \right) \log \frac{q\left( z \right)}{p\left( z,\boldsymbol{x} \right)}\mathrm{d}z}
$$
​	$L\left( q\left( z \right) \right)$​称为变分下界函数，也称为证据下界。它可以进一步拆成：
$$
L\left( q\left( z \right) \right) =-\int{q\left( z \right) \log \frac{q\left( z \right)}{p\left( z,\boldsymbol{x} \right)}\mathrm{d}z}=-\int{q\left( z \right) \log \frac{q\left( z \right)}{p\left( z \right) p\left( \boldsymbol{x}|z \right)}\mathrm{d}z}
\\
=-\int{q\left( z \right) \left( \log \frac{q\left( z \right)}{p\left( z \right)}+\log \frac{1}{p\left( \boldsymbol{x}|z \right)} \right) \mathrm{d}z}
\\
=\mathbb{E} _{q\left( z \right)}\left[ \log p\left( \boldsymbol{x}|z \right) \right] -D_{KL}\left( q\left( z \right) ||p\left( z \right) \right)
$$
​	可以看到，第二个等号里，前项是一个KL散度，后项可以写作一个期望。

### Loss Function

​	现在，我们回到推导损失函数的路上来。很自然的，我们想通过最小化$q(\mathbf{x}_0)$和$p_\theta(\mathbf{x}_0)$的交叉熵：
$$
\mathcal{L} =\mathbb{E} _{\mathbf{x}_0\sim q\left( \mathbf{x}_0 \right)}\left[ -\log p_{\theta}\left( \mathbf{x}_0 \right) \right]
$$
​	然而，我们并不知道$p_{\theta}\left( \mathbf{x}_0 \right)$的表达式，没法直接计算交叉熵。我们试着通过之前全概率公式和贝叶斯公式展开的结果，来看看能不能获得什么启发：
$$
\mathcal{L} =-\mathbb{E} _{\mathbf{x}_0\sim q\left( \mathbf{x}_0 \right)}\left[ -\log p_{\theta}\left( \mathbf{x}_0 \right) \right] 
\\
=-\mathbb{E} _{\mathbf{x}_0\sim q\left( \mathbf{x}_0 \right)}\left[ \log \int{p_{\theta}\left( \mathbf{x}_0,\mathbf{x}_1,...,\mathbf{x}_T \right) \mathrm{d}\left( \mathbf{x}_1,...,\mathbf{x}_T \right)} \right]
$$
​	为了表达的简洁，$\mathbf{x}_0,\mathbf{x}_1,...,\mathbf{x}_T$我们重新记作$\mathbf{x}_{0:T}$。

​	接下来一个在推导VAE时常见的处理作法，将损失函数转化到一个好处理的形式：
$$
\mathcal{L} =-\mathbb{E} _{\mathbf{x}_0~q\left( \mathbf{x}_0 \right)}\left[ -\log \int{p_{\theta}\left( \mathbf{x}_{0:T} \right) \mathrm{d}\left( \mathbf{x}_{1:T} \right)} \right] 
\\
=-\mathbb{E} _{\mathbf{x}_0~q\left( \mathbf{x}_0 \right)}\left[ \log \int{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right) \frac{p_{\theta}\left( \mathbf{x}_{0:T} \right)}{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right)}\mathrm{d}\left( \mathbf{x}_{1:T} \right)} \right] 
\\
=-\int{q\left( \mathbf{x}_0 \right)}\log \left( \int{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right) \frac{p_{\theta}\left( \mathbf{x}_{0:T} \right)}{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right)}\mathrm{d}\left( \mathbf{x}_{1:T} \right)} \right) \mathrm{d}\mathbf{x}_0
$$
​	然后，由琴生不等式：
$$
f\left( \frac{\sum\nolimits_{i=1}^n{x_i}}{n} \right) \geqslant \frac{\sum\nolimits_{i=1}^n{f\left( x_i \right)}}{n}
\\
f\left( E\left[ X \right] \right) \leqslant E\left( f\left( X \right) \right) 
$$
​	取$f$为$\mathrm{log}(\cdot)$，同时注意到期望的范围合并了。上面的损失函数可以化为：
$$
=-\int{q\left( \mathbf{x}_0 \right)}\log \left( \int{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right) \frac{p_{\theta}\left( \mathbf{x}_{0:T} \right)}{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right)}\mathrm{d}\left( \mathbf{x}_{1:T} \right)} \right) \mathrm{d}\mathbf{x}_0
\\
\leqslant -\int{q\left( \mathbf{x}_0 \right)}\int{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right) \log \left( \frac{p_{\theta}\left( \mathbf{x}_{0:T} \right)}{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right)} \right) \mathrm{d}\left( \mathbf{x}_{1:T} \right)}\mathrm{d}\mathbf{x}_0
\\
=\int{q\left( \mathbf{x}_{0:T} \right)}\log \left( \frac{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_{0:T} \right)} \right) \mathrm{d}\mathbf{x}_{0:T}=\mathbb{E} _{q\left( \mathbf{x}_{0:T} \right)}\left[ \log \left( \frac{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_{0:T} \right)} \right) \right] 
$$
​	最后的这一项，其实给出了一个很好的上界。这一形式见于之前变分推断的前置知识中的：
$$
L\left( q\left( z \right) \right) =\mathbb{E} _{q\left( z \right)}\left[ \log p\left( \boldsymbol{x}|z \right) \right] -D_{KL}\left( q\left( z \right) ||p\left( z \right) \right) 
$$
​	但是我们化简后的上界里，对数里面还存在不好表达的联合分布：但先前其实我们已经做足了准备工作：
$$
q\left( \mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_T|\mathbf{x}_0 \right) =\prod_{t=1}^T{q\left( \mathbf{x}_t,\mathbf{x}_{t-1} \right)}
\\
p_{\theta}\left( \mathbf{x}_0,\mathbf{x}_1,...,\mathbf{x}_T \right) =p_{\theta}\left( \mathbf{x}_T \right) \prod_{t=1}^T{p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right)}
$$
​	所以上界对数里的$\mathrm{log}$，我们可以逐渐进行处理：
$$
\log \left( \frac{q\left( \mathbf{x}_{1:T}|\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_{0:T} \right)} \right) =\log \left( \frac{\prod_{t=1}^T{q\left( \mathbf{x}_t,\mathbf{x}_{t-1} \right)}}{p_{\theta}\left( \mathbf{x}_T \right) \prod_{t=1}^T{p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right)}} \right) 
\\
=-\log p_{\theta}\left( \mathbf{x}_T \right) +\sum_{t=1}^T{\log \frac{q\left( \mathbf{x}_t,\mathbf{x}_{t-1} \right)}{p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right)}}
$$
​	我们知道$q\left( \mathbf{x}_t,\mathbf{x}_{t-1} \right) $​不好处理，还好之前我们有：
$$
q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right) 
=\frac{q\left( \mathbf{x}_t|\mathbf{x}_{t-1} \right) q\left( \mathbf{x}_{t-1}|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_t|\mathbf{x}_0 \right)}
$$
​	我们发现，后验概率不可避免的出现在了推导损失函数的过程中，注意下标的变化，我们继续展开：
$$
=-\log p_{\theta}\left( \mathbf{x}_T \right) +\sum_{t=2}^T{\log \left( \frac{q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right)} \right) \cdot \left( \frac{q\left( \mathbf{x}_t|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_{t-1}|\mathbf{x}_0 \right)} \right)}+\log \frac{q\left( \mathbf{x}_1|\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_0|\mathbf{x}_1 \right)}
\\
=-\log p_{\theta}\left( \mathbf{x}_T \right) +\sum_{t=2}^T{\log \left( \frac{q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right)} \right) +\sum_{t=2}^T{\log \left( \frac{q\left( \mathbf{x}_t|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_{t-1}|\mathbf{x}_0 \right)} \right)}}+\log \frac{q\left( \mathbf{x}_1|\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_0|\mathbf{x}_1 \right)}
$$
​	注意到第二个求和符号，根据对数的性质，它是可以约化的：
$$
\sum_{t=2}^T{\log \left( \frac{q\left( \mathbf{x}_t|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_{t-1}|\mathbf{x}_0 \right)} \right)}=\log \frac{q\left( \mathbf{x}_2|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_1|\mathbf{x}_0 \right)}+\log \frac{q\left( \mathbf{x}_3|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_2|\mathbf{x}_0 \right)}...+\log \frac{q\left( \mathbf{x}_T|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_{T-1}|\mathbf{x}_0 \right)}=\log \frac{q\left( \mathbf{x}_T|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_1|\mathbf{x}_0 \right)}
$$
​	对于现在剩下的三个落单的$\mathrm{log}$，有：
$$
-\log p_{\theta}\left( \mathbf{x}_T \right) +\log \frac{q\left( \mathbf{x}_T|\mathbf{x}_0 \right)}{q\left( \mathbf{x}_1|\mathbf{x}_0 \right)}+\log \frac{q\left( \mathbf{x}_1|\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_0|\mathbf{x}_1 \right)}
\\
=-\log p_{\theta}\left( \mathbf{x}_T \right) +\log \frac{q\left( \mathbf{x}_T|\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_0|\mathbf{x}_1 \right)}
\\
=\log \frac{q\left( \mathbf{x}_T|\mathbf{x}_0 \right)}{p_{\theta}\left( \mathbf{x}_T \right)}-\log p_{\theta}\left( \mathbf{x}_0|\mathbf{x}_1 \right)
$$
​	这样化简，会带来一个好处，因为$\mathbf{x}_T$是已知的采样出来的数据分布，$\mathbf{x}_0$是先验（输入的图片），这样当求期望时，常数的期望还是个常数。将整理后的式子关于$q$求期望，得：
$$
\mathcal{L} _{VLB}=\mathbb{E} _q\left[ -\log p_{\theta}\left( \mathbf{x}_0|\mathbf{x}_1 \right) \right] +\sum_{t=2}^T{D_{KL}\left( q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right) ||p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right) \right)}+
\\D_{KL}\left( q\left( \mathbf{x}_T|\mathbf{x}_0 \right) ||p_{\theta}\left( \mathbf{x}_T \right) \right) 
$$
​	最后一项是常数，我们不用管它。先关注最复杂的第二项。我们前面给出过：
$$
q\left( \mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0 \right) =\mathcal{N} \left( \mathbf{x}_{t-1}|\tilde{\mu}\left( \mathbf{x}_t,\mathbf{x}_0 \right) ,\tilde{\beta}_t\mathbf{I} \right)
\\
p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right) =\mathcal{N} \left( \mathbf{x}_{t-1}|\tilde{\mu}_{\theta}\left( \mathbf{x}_t,t \right) ,\varSigma _{\theta}\left( \mathbf{x}_t,t \right) \mathbf{I} \right)
$$
​	高斯分布下的KL散度存在解析解，这里不再计算了，直接给出解析式：
$$
D_{KL}\left( p_1||p_2 \right) =\frac{1}{2}\left( \ln \frac{\left| \varSigma _2 \right|}{\left| \varSigma _1 \right|}-d+\mathrm{tr}\left( \varSigma _{2}^{-1}\varSigma _1 \right) +\left( \boldsymbol{\mu }_2-\boldsymbol{\mu }_1 \right) ^T\varSigma _{2}^{-1}\left( \boldsymbol{\mu }_2-\boldsymbol{\mu }_1 \right) \right) 
$$
​	我们知道，方差是被调度器所固定的，只需关注均值构成的二次型：
$$
=\frac{1}{2\left\| \varSigma _{\theta} \right\| _{2}^{2}}\left\| \frac{1}{\sqrt{\bar{\alpha}_t}}\left( \mathbf{x}_t-\frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}}\mathbf{\epsilon } \right) -\frac{1}{\sqrt{\bar{\alpha}_t}}\left( \mathbf{x}_t-\frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}}\mathbf{\epsilon }_{\theta}\left( \mathbf{x}_t,t \right) \right) \right\| _2
\\
=\frac{\beta _{t}^{2}}{2\alpha _t\left( 1-\bar{\alpha}_t \right) \left\| \varSigma _{\theta} \right\| _{2}^{2}}\left\| \mathbf{\epsilon }-\mathbf{\epsilon }_{\theta}\left( \mathbf{x}_t,t \right) \right\| _2
$$
​	实践表明，不考虑前面的权重，生成效果更好，于是可以进一步化简为：
$$
\left\| \mathbf{\epsilon }-\mathbf{\epsilon }_{\theta}\left( \mathbf{x}_t,t \right) \right\| _2
$$
​	最后，还剩一个小问题没有被澄清，最前面那一项：
$$
\mathbb{E} _q\left[ -\log p_{\theta}\left( \mathbf{x}_0|\mathbf{x}_1 \right) \right]
$$
​	根据我们的模型$p_{\theta}\left( \mathbf{x}_{t-1}|\mathbf{x}_t \right) =\mathcal{N} \left( \mathbf{x}_{t-1},\tilde{\mu}_{\theta}\left( \mathbf{x}_t,t \right) ,\varSigma _{\theta}\left( \mathbf{x}_t,t \right) \mathbf{I} \right)$，当$t$​为0时，这是一个多元高斯分布。最前面那一项代表的其实就是这个多元高斯分布的熵，而这个熵其实只与高斯分布的协方差有关：
$$
\mathrm{H}\left[ \boldsymbol{x} \right] =\frac{d}{2}\left( \log 2\pi +1 \right) +\frac{1}{2}\ln \left| \varSigma \right|
$$
​	所以看起来，这一项也是个常数。但是DDPM的作者指出，最后一步的过程不能简单的看作从$\mathcal{N} \left( \tilde{\mu}_{\theta}\left( \mathbf{x}_1,1 \right) ,\varSigma _{\theta}\left( \mathbf{x}_1,1 \right) \mathbf{I} \right) $中采样，而是在采样的基础上再加上一步离散化。这就导致这一项并不是个常数。DDPM原文中给出了一种计算这一项的方法，后来也被改进为单独用一个离散的“解码器”来建模，总之就是不用考虑这一项了。

<center>
    <img src='/images/ddpm/DDPM_1.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	最后，训练和采样的过程，就像DDPM原文中给出的一样，非常简洁。训练时，从数据集中取出$q( \mathbf{x}_0)$，从均匀分布中抽样时间步$t$，再生成一个随机噪声$\mathbf{\epsilon }$。之后将前向扩散得到的$\mathbf{x}_t$和$t$输入模型，得到预测的结果${\epsilon }_{\theta}$，然后梯度回传，反向传播。

​	采样时，则直接采样一个高斯噪声$\mathbf{x}_t$，得到估计的噪声$\mathbf{\epsilon_\theta }$，利用估计的噪声$\mathbf{\epsilon_\theta }$和$\mathbf{x}_t$，计算估计的均值：
$$
\mu _{\theta}\left( \mathbf{x}_t,t \right) =\frac{1}{\sqrt{\bar{\alpha}_t}}\left( \mathbf{x}_t-\frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}}\mathbf{\epsilon }_{\theta}\left( \mathbf{x}_t,t \right) \right)
$$
​	再从估计出的$\mathcal{N} \left( \tilde{\mu}_{\theta},\beta _t\mathbf{I} \right) $中采样得到$\mathbf{x}_{t-1}$​，同样是用参数重整化技巧：
$$
\mathbf{x}_{t-1}=\frac{1}{\sqrt{\bar{\alpha}_t}}\left( \mathbf{x}_t-\frac{1-\bar{\alpha}_t}{\sqrt{1-\bar{\alpha}_t}}\mathbf{\epsilon }_{\theta}\left( \mathbf{x}_t,t \right) \right) +\sqrt{\beta _t}\epsilon
$$
​	当$t=1$时，不再添加噪声。因为此时预测出来的已经是$\mathbf{x}_0$了，再多加噪声只会使得生成质量变差。

### Demo

​	当有了这样的模型，第一时间我会做什么呢？显然是生成一下纸片人。之前我用Progressive Growing GAN玩过一次，这次用简单的DDPM实现了一下，发现DDPM比PGGAN实践起来简单多了，当然我还没来得及实践高分辨率的生成效果。

​	DDPM使用了加入了self-attention层的U-Net结构，将图像输入后，使用下采样和卷积，将图像投影到一个小的尺寸上。然后再使用上采样和卷积，把它再还原回原尺寸，同时下采样过程和上采样过程存在跳连。

​	由于这一过程需要向网络指出此时所处的时间戳$t$，所以引入了transformer中的位置编码：
$$
PE_{\left( pos,2i \right)}=\sin \left( \frac{pos}{10000\frac{2i}{d}} \right) 
\\
PE_{\left( pos,2i+1 \right)}=\cos \left( \frac{pos}{10000\frac{2i}{d}} \right)
$$
​	这些信息会直接”加入“图像传播的过程中。

​	基本要做的，其实就是建立一个diffusion的类，剩下的就nothing new了：

```python
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.fixed_z = torch.randn((1, 3, self.img_size, self.img_size)).to(self.device)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = self.fixed_z
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
```

​	以及一个trainer:

```python
def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=1)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
```

​	当然，如果现在去github上搜OpenAI或者别的project写的扩散模型，他们十分复杂，非常的长，他们有各种加速采样的tricks，以及更细致的损失函数。比如他们中有的忠实的实现了变分下界，而DDPM原文里实际上已经放弃了那一项。直接看他们的代码是非常不利于初学者的。

​	我最后合成了一个gif来作纪念：

<center>
    <img src='/images/ddpm/DDPM_demo.gif' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	可以看出，这看起来”come from nowhere“，在高斯噪声中突然就演化出了不同的图案，spectacular！

​	如果观察在训练时，不同epoch下，采样出的结果，会感到更加的神奇：

<center>
    <img src='/images/ddpm/DDPM_demo_2.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	图片从上到下依次是epoch从小到大的结果。最初，网络去噪之后是一团混沌，随着训练的进行逐渐开始拟合噪声。如果我不过一遍上面那些式子，然后一个人过来，告诉我说：”你可以通过预测噪声来去噪，你每次只需rand出噪声就好了“。我是断然不信的，但是随机过程和动力学给出了insight，让这个操作得以实现。

​	我把img_size改成128的，期望生成分辨率更高的图片，这次的效果就没这么好了。一方面是数据集有限，另一方面是由于分辨率变大，self-attention的一个hidden_dim跟着扩大，导致训练成本有点贵了，于是随便玩了几下就不玩了。

<center>
    <img src='/images/ddpm/ddpm_128.jpg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	有些结果还是很不错的。

### End

​	这就是对DDPM的简单介绍了，一些加速采样，余弦schedule，设计技巧之类的操作，由于时间关系和篇幅原因，就不写了。可以从Sampling看出，这就是在一团噪声中取出了什么东西，就像许愿池一样。如果不是明天考认知计算，我还没开始看PPT，我会搓个demo试一下。

​	虽然，我可能最应该做的是找github的repo开始看文件然后魔改……或者去背PPT……推一遍公式然后写下来，除了消磨了我一天时间以外，估计也就锻炼大脑一个用处了。

​	由于我水平有限，不能很高屋建瓴的说出扩散模型和之前的VAE，GAN的异同。当然这设计上差距是很大的，感觉有一种mulit-scale的设计一直在这。算辣，这个不是很重要。

<center>
    <img src='/images/ddpm/DDPM_2.jpeg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>
