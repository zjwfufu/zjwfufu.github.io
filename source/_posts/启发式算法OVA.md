---
title: 启发式算法OVA
date: 2021-08-29 14:31:03
tags: [数学建模,启发式算法]
mathjax: true
categories: 数学建模
---

​	课也选完了，明早就要跑路了，最后写个关于启发式算法的实际运用，来<\假期>

<!--more-->

​	要解决的问题选自CUMCM2020A第三小问，求解一组温度设置，使得炉温曲线从超过217°C到峰值所覆盖的面积最小。由于具体问题要求说起来比较长了，这里略去。首先建立温度场函数，即预热区，恒温区，回流区，冷却区，且其中小温区1~5的温度保持一致，小温区8~9的温度保持一致，小温区10~11的温度保持一致，同时要考虑炉前区域和炉后区域的温度（室温为25°C）。考虑一维热传导方程：
$$
\frac{\partial u\left( x,t \right)}{\partial t}=k^2\frac{\partial ^2u\left( x,t \right)}{\partial x^2}
$$
​	由于依题目要求，回焊炉启动后炉间温度快速平衡，那么温度函数关于$t$的一阶偏导为0，则关于$x$​炉间温度分布呈线性关系，对于炉前区域和炉后区域，认为从炉头到炉前末端，从炉后前端到炉尾，温度为线性变化。编写一维温度场函数（利用阶跃函数heaviside避免if语句，但是这两种做法哪个更快我还没有试过）：

```matlab
function T=TEMP(x)
global T1 T2 T3 T4
T=ceil(heaviside(x)-heaviside(x-25))*((((T1-25)/25)*(x)+25))+...
    floor(heaviside(x-25)-heaviside(x-197.5))*T1+...
    ceil(heaviside(x-197.5)-heaviside(x-202.5))*((((T2-T1)/5)*(x-197.5)+T1))+...
    floor(heaviside(x-202.5)-heaviside(x-233))*T2+...
    ceil(heaviside(x-233)-heaviside(x-238))*((((T3-T2)/5)*(x-233)+T2))+...
    floor(heaviside(x-238)-heaviside(x-268.5))*T3+...
    ceil(heaviside(x-268.5)-heaviside(x-273.5))*((((T4-T3)/5)*(x-268.5)+T3))+...
    floor(heaviside(x-273.5)-heaviside(x-339.5))*T4+...
    ceil(heaviside(x-339.5)-heaviside(x-344.5))*((((25-T4)/5)*(x-339.5)+T4))+...
    floor(heaviside(x-344.5)-heaviside(x-435.5))*25;
```

​	当给出一组温度$T$时，温度场函数的图像如图所示：

<center>
    <img src='/images/heuristic_ova/GAova_1.jpg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	该问题可以归结为一维的热传导问题，边界条件由牛顿冷却定律给出：
$$
\left\{ \begin{array}{c}
	\frac{\partial T}{\partial t}=\alpha \frac{\partial ^2T}{\partial y^2}\\
	\frac{\partial T}{\partial y}\mid_{y=|\frac{d}{2}|}^{}=h\left( T_{env}\left( x \right) -T_{surf} \right)\\
\end{array} \right. 
$$
​	其中参数$\alpha$代表热扩散率，这个物理量与温度有关，不同的温度段热扩散率可能有差异，这里按照上述分布的五个温度分区定义五个热扩散率$\alpha_1,\alpha_2,..\alpha_5$。由有限差分法，迭代格式可以写作：
$$
U^{n+1}=A^{-1}BU^n
$$
​	编程求解得到炉温曲线和电路板温度随时间的变化的图像：

<center>
    <img src='/images/heuristic_ova/GAova_2.jpg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

<center>
    <img src='/images/heuristic_ova/GAova_3.jpg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	迭代计算温度T的函数CurveCal.m过于冗长，这里略去。

​	此时还要考虑制程界限，即题目表格的限制条件：

| 界限名称                      | 最低值 | 最高值 | 单位 |
| ----------------------------- | ------ | ------ | ---- |
| 温度上升斜率                  | 0      | 3      | °C/s |
| 温度下降斜率                  | -3     | 0      | °C/s |
| 温度上升时在150°C~190°C的时间 | 60     | 120    | s    |
| 温度大于217°C的时间           | 40     | 90     | s    |
| 峰值温度                      | 240    | 250    | °C   |

​	

```matlab
function flag=limit(T,dt)
dT=T(2:end)-T(1:end-1);
k=[0 dT/dt];
Heating_Limit=0;
Extrame_Limit=0;

for i=1:length(T)
    if k(i)>0 && T(i)>150 && T(i)<190
        Heating_Limit=Heating_Limit+1;
    elseif T(i)>217
        Extrame_Limit=Extrame_Limit+1;
    end
end

Heating_time=Heating_Limit*dt;
Extrame_time=Extrame_Limit*dt;

[Highest,Highest_index]=max(T);

if max(k)>3 || min(k)<-3
    flag=0;
elseif Highest>250 || Highest<240
    flag=0;
elseif Heating_time<60 || Heating_time>120
    flag=0;
elseif Extrame_time<40 || Extrame_time>90
    flag=0;
else
    flag=1;
end
```

​	完成上述铺垫就可以调用遗传算法工具箱求解该问题，（不到万不得已不要用遗传算法这种启发式算法，没必要，本题是解空间维度很高，且根据数学试验后，并没有十分显著的规律可以得出。~~其实“充足的数值实验”可以证明此时最优的T4为265摄氏度左右~~）

```matlab
clear all
clc
dt=0.5;

tic;
opt=gaoptimset('Generations',800,'MigrationFraction',0.3);
lb=[165 185 225 245 65];
ub=[185 205 245 265 100];
[x,fval]=ga(@fitness,5,[],[],[],[],lb,ub,[],opt);
toc;

function E=fitness(x)
global T1
global T2
global T3
global T4
T1=x(1);
T2=x(2);
T3=x(3);
T4=x(4);
v=x(5);
dt=0.5;

if T1<165||T1>185||T2<185||T2>205||T3<225||T3>245||T4<245||T4>265||v<65||v>100
    E=10000;
else
    T=CurveCal(v/60,dt);
    T_curve=T(76,:);
    flag=limit(T_curve,dt);
    if flag==0
        E=10000;
        E
    else
        [~,index]=max(T_curve);
        above=find(T_curve>217);
        fit=T_curve(above(1):index)-217;
        E=sum((fit(1:end-1)+fit(2:end))*dt/2);
        E
    end
end
end
```

​	运行时间大约在十几分钟到一个小时不等。

​	在这里用matlab的ga函数是因为短时间内编写这种群优化算法比较困难，下面用模拟退火来进行“验算”。

```matlab
function [BestT1,BestT2,BestT3,BestT4,Bestv,trace,T_m]=SAA()
clear all
clc
global T1 T2 T3 T4
dt=0.5;

L=100;
K=0.98;
S=0.08;
T_m=5;
P=1;
trace=linspace(0,0,20000);
%初始化
f=1;
while(f==1)
    Bestv=rand*35+65;
    BestT1=rand*20+165;
    BestT2=rand*20+185;
    BestT3=rand*20+225;
    BestT4=rand*20+245;
    T1=BestT1;
    T2=BestT2;
    T3=BestT3;
    T4=BestT4;
    T=CurveCal(Bestv/60,dt);
    T_curve_best=T(76,:);
    flag=limit(T_curve_best,dt);
    if flag==1
        f=0;
    end
end


while (T_m>0.001) && (P<20000)
    T_m=K*T_m;
    for i=1:L
        p=0;
        while p==0
            Nextv=Bestv+T_m*(rand-rand);
            NextT1=BestT1+T_m*(rand-rand);
            NextT2=BestT2+T_m*(rand-rand);
            NextT3=BestT3+T_m*(rand-rand);
            NextT4=BestT4+T_m*(rand-rand);
            [Nextv,NextT1,NextT2,NextT3,NextT4]=Judge(Nextv,NextT1,NextT2,NextT3,NextT4);
            T1=NextT1;
            T2=NextT2;
            T3=NextT3;
            T4=NextT4;
            T=CurveCal(Nextv/60,dt);
            T_curve_next=T(76,:);
            flag=limit(T_curve_next,dt);
            if flag==1
                p=1;
            end
        end
        S_best=Cal_S(T_curve_best,dt);
        S_next=Cal_S(T_curve_next,dt);
        if S_best>S_next
            Bestv=Nextv;
            BestT1=NextT1;
            BestT2=NextT2;
            BestT3=NextT3;
            BestT4=NextT4;
            P=P+1;
            T_curve_best=T_curve_next;
            S_best=S_next;
        else
            change=-(S_next-S_best)/T_m;
            p1=exp(change);
            if p1>rand
                Bestv=Nextv;
                BestT1=NextT1;
                BestT2=NextT2;
                BestT3=NextT3;
                BestT4=NextT4;
                P=P+1;
                T_curve_best=T_curve_next;
                S_best=S_next;
            end
        end
        trace(P)=S_best;
    S_best
    T_m
    end
end
```

​	运行后发现两种算法的结果为：

|      | $T_1$    | $T_2$    | $T_3$    | $T_4$    | $v$     | $S$      |
| ---- | -------- | -------- | -------- | -------- | ------- | -------- |
| GA   | 178.8662 | 200.0697 | 226.7841 | 264.9790 | 91.1204 | 490.4073 |
| SAA  | 181.4935 | 199.7269 | 225.2310 | 264.9768 | 90.9217 | 490.6445 |

​	所以可以认为所求的最优解就在490附近。在求解解空间很高的优化问题时，一定要注意每个过程用到的子函数是否简洁，否则可能耗费更多的时间。

​	当写完这个的时候，我已经到学校了，刚把阳台收拾完，返校那一路十分折磨，仿佛是对我假期摸鱼的报复。这学期，要做个人，从不翘课开始，好好学习，天天向上。

<center>
    <img src='/images/heuristic_ova/GAova_4.jpg'  style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

​	后续：

​	当我像往常一样部署时发现展开这篇博客全文后显示为404的问题，找了好久发现是大小写的问题，因为原本我打的是'ova',后来上传上去想改成'OVA',可是改好部署后就发生了404，原因是git默认忽视文件名大小写，所以即使大小写变更，git也检测不到，解决方案是从博客项目中的.deploy_git中，修改.git下的config,将ignorecase=true改成ignorecase=false。
