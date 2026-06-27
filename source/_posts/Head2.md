---
title: Look Closely at Head 2
mathjax: true
date: 2024-01-19 22:06:11
tags: [深度学习,3DV]
categories:
  - [深度学习]
  - [3DV]
---

​	“沿着风的轨迹，乱舞蹁跹，困于死水之间。”

<!--more-->

​	这篇blog想记录一下我在“试图”（因为没做出来）做毕设的时候，关于人脸/人头数据的一些预处理。

​	预处理（preprocess）是非常重要的一个环节，在刚学习炼丹时，预处理只是对类似图片数据那样的，做均值-标准差归一化什么的，你甚至可以借助torchvision轻易的实现：

```python
transform = transforms.Compose([transforms.RandomRotation(20),
                                transforms.ColorJitter(brightness=0.1),
                                transforms.Resize([150,150]),
                                transforms.ToTensor(),
                               ])
```

​	但当接触的任务变得复杂的时候，就会发现预处理往往比炼丹本身要复杂。预处理本身是需要一些先验知识的，比如一些医学图像处理的任务里，那些CT扫描或MRI图像的格式并不是三通道的RGB，需要自行进行处理，以及他们一张图分辨率往往非常大，所以可能需要自行打patch；在有些跟信号处理，频谱挂钩的任务里，需要对时频分析，功率谱，维纳-辛钦定理有一些认识。但这些好歹都还是“theoretical”的问题，只需要“了解”这些概念，然后读一些文档，就能跟上了。

​	由于我毕设试图做关于人头相关的，预处理的步骤是为了从单目的人头视频帧中提取一些先验。这就有些一言难尽，这些预处理管线往往都基于了许多人头人脸之前工作的pretrained model，然后将许多仓库进行穿针引线，最后打包成一个shell脚本来一键启动。这个往往是非常“technical”的事情，不同作者预处理的方式，得到的数据集的内容和格式都有所不同。以及这里有许多繁杂的概念，不是那么好理解。

> 但大多数时候，把前面工作的数据集制作的管线直接拿来follow，或者直接用其制作好的数据集，也不失为一种洛可可风格的方式。

​	所以这篇blog旨在记录[INSTA](https://github.com/Zielon/INSTA)和[PointAvatar](https://github.com/zhengyuf/PointAvatar)两篇工作的预处理管线，前者是马普所的，后者是苏黎世理工的，这两篇工作都是为了从单目视频中重建出一个人头，或者叫“Avatar”。剖析这两个预处理管线主要是为了其中的细节，所以很难具体的按某个特定逻辑来撰写，不可避免的会显得生硬一些。但大概会包含以下要素：

- 如何从单目图像中提取FLAME系数。
- FLAME的灵活使用。
- landmarks（人脸关键点）之于预处理中的作用。
- 某科学の线性代数。

### INSTA

​	INSTA的数据预处理分为三步：

- 先从[MICA](https://github.com/Zielon/MICA)中运行`demo.py`，获得一个`identity.npy`。
- 然后用[Metrical-Tracker](https://github.com/Zielon/metrical-tracker)，将一段视频和刚才提取的`identity.npy`作输入，得到一系列的输出。
- 最后用[INSTA](https://github.com/Zielon/INSTA)中提供的脚本`generate.sh`来将刚才那一系列输出再处理成需要的样子。

​	我们一步一步来看，MICA也是马普所的一篇工作，目的是从一张二维的图片中恢复基于FLAME的3D topology。第一步是得：

```shell
python demo.py -i ./demo/input -o ./demo/output
```

​	这个操作是为了得到一个`identity.npy`，具体来说，这个`identity.npy`是先将输入图片，送入一个预训练好的ArcFace，ArcFace是一个很成熟的用于人脸识别的模型，ArcFace是用一种辅助度量的loss来进一步帮助以ResNet-50为backbone的分类网络来分类，它已经是2017~2018年时候的事情了，我们现在已经不用关心其本身了。不过那时候是一个很好的年代，不仅是因为炼丹的刀耕火种在如火如荼的进行，还因为那时候我还能和女同学言笑晏晏。

<center>
    <img src='/images/head2/head2_arcface.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

​	总之它是一个成熟的分类网络，那样对于任意一张“wild-image”的人脸输入，它会有不同的“响应”。MICA里继续用几层layers将ArcFace计算出的特征向量再映成FLAME里的形状系数：
$$
\boldsymbol{z}=\mathcal{M} \left( \mathrm{ArcFace}\left( I \right) \right) 
\\
\mathcal{G} _{3\mathrm{DMM}}\left( \boldsymbol{z} \right) =\mathcal{B} \odot \boldsymbol{z}+\mathcal{A}
$$
​	如果我们进入MICA中的`demo.py`，会发现`identity.npy`保存的是FLAME的形状系数，即上面的$\boldsymbol{z}\in \mathbb{R} ^{300}$。`demo.py`还会输出出`kpt7.npy`和`kpt68.npy`，这里的人脸关键点并不是估计出来的，它的来历和上一篇blog里讲的从[RingNet](https://zhangtemplar.github.io/ringnet/)里提取出的固定的embedding里抽landmark是一样的，是从FLAME中预标定好的对于不同landmark的顶点索引中查出来的。是一个象征意义的三维的landmark。而我们朴素意义上的“landmark detection”一般发生在裁剪图片的时候，例如：

```python
app = LandmarksDetector(model=detectors.RETINAFACE)
...
img = cv2.imread(image_path)
bboxes, kpss = app.detect(img)
```

​	这里的`kpss`就是我们以为的那种用NN估计出的二维的landmark，这个的功能和上文的ArcFace，已经完善的部署在了`insightface`或`face_alignment`库中。

> 在后文中会用多种方式获得，并多次使用landmark。landmark在曾经应该是为了帮助人们进行人脸识别，表情识别。在以前依赖手工特征的时候，这很有用。但现在这两个任务已经基本宣告“closed”了，那么在这两个管线里，“landmark是用来对齐不同knowledge的一个载体”，往下看就知道了。

​	接下来就到了Metrical-Tracker的环节：

```shell
python tracker.py --cfg ./configs/actors/xxx.yml
```

Metrical-Tracker和MICA其实指代的是同一篇工作，只不过前者是后者的一个扩展，他们编写了一个基于python的仓库来实现对视频中的人头进行“tracking”。在INSTA原文中的表述是：

> To this end, we use the analysis-by-synthesis-based face tracker from MICA [61], based on Face2Face [51] using a sampling-based differentiable rendering. We refer to the original paper [51] for more details. We extend the optimization with two extra blendshapes for eyelids and iris tracking using Mediapipe [34]. In contrast to MICA, we also optimize for FLAME shape parameters, with regularization towards MICA shape prediction instead of the average face shape as in Face2Face [51]

​	这里所谓的“analysis-by-synthesis-based”只是一个名字，最早是因为通过2D图像来估计3D的人脸，这个操作本身是病态（ill-conditioned）的。于是就有了将3D模型与一组用于渲染的参数一起优化，来使渲染出的照片更接近真实图片。这个思想就被称作“analysis-by-synthesis”，放在现在看来已经很平常了。

​	于是我们可以从`tracker.py`开始阅读，这是一个行数挺多的类。但在此之前我们会注意到在global里有：

```python
mediapipe_idx = np.load('flame/mediapipe/mediapipe_landmark_embedding.npz', allow_pickle=True, encoding='latin1')['landmark_indices'].astype(int)
left_iris_flame = [4597, 4542, 4510, 4603, 4570]
right_iris_flame = [4051, 3996, 3964, 3932, 4028]
left_iris_mp = [468, 469, 470, 471, 472]
right_iris_mp = [473, 474, 475, 476, 477]
```

​	这里就是为了引入Mediapipe的起手式，Mediapipe是Google做的一个非常成熟的机器学习管线，可以支持许多感知上的任务，比如在姿态估计，手部估计等等。其最后上线的模型一定是被Google的工程师精心调优过的，而且也是在Google的大规模私有数据集做出来的，所以结果是有保障的。

​	实际上，我们更习惯的`face_alignment`库也可以检测上下眼皮（eyelids），但不能检测虹膜（iris）。以及`face_alignment`可能检测上下眼皮的准确度稍逊于Mediapipe（Google力大砖飞的操作在MoveNet里我已经体会过了，他们为了让做瑜伽时候的姿态估计准一些，爬了YouTube上几乎所有的瑜伽视频），所以在整个管线里整合来自三处的landmark进行监督（Mediapipe，`face_alignment`，RingNet）可热闹极了，下面我们会提到。

​	首先，Mediapipe在dense模式下，是预测478个关键点。后10个分别就是左眼虹膜和右眼虹膜，即`left_iris_mp`和`right_iris_mp`。由于他们是Mediapipe版本更新后追加的，所以是后10个：

<center>
    <img src='/images/head2/head2_1.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

​	上图的五个绿色点，就是对虹膜的定位。

​	而`left_iris_flame`和`right_iris_flame`对应五个FLAME模板上的顶点，这十个FLAME上的顶点的位置，在拓扑上对应着上面虹膜检测的那五个点。他们的作用往后会看到。

​	我们先说回来`tracker.py`的逻辑：

```python
class Tracker(object):
	...    
    def run(self):
        self.prepare_data()
        if not self.load_checkpoint():
            self.initialize_tracking()
            self.frame = 0

        self.optimize_video()
        self.output_video()


if __name__ == '__main__':
    config = parse_args()
    ff = Tracker(config, device='cuda:0')
    ff.run()
```

​	第一步是先`self.prepare_data()`，这一步的核心基本是从视频和输入的`identity.npy`里生成每一帧的图片和landmark：

```python
class GeneratorDataset(Dataset, ABC):
    def __init__(self, source, config):
		...
        self.initialize()
        self.face_detector_mediapipe = FaceDetector('google')
        self.face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)

    def initialize(self):
        path = Path(self.source, 'source')
        if not path.exists() or len(os.listdir(str(path))) == 0:
            video_file = self.source / 'video.mp4'
            if not os.path.exists(video_file):
                logger.error(f'[ImagesDataset] Neither images nor a video was provided! Execution has stopped! {self.source}')
                exit(1)
            path.mkdir(parents=True, exist_ok=True)
            os.system(f'ffmpeg -i {video_file} -vf fps={self.config.fps} -q:v 1 {self.source}/source/%05d.png')

        self.images = sorted(glob(f'{self.source}/source/*.jpg') + glob(f'{self.source}/source/*.png'))

    def process_face(self, image):
        lmks, scores, detected_faces = self.face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True)
        if detected_faces is None:
            lmks = None
        else:
            lmks = lmks[0]
        dense_lmks = self.face_detector_mediapipe.dense(image)
        return lmks, dense_lmks

    def run(self):
        logger.info('Generating dataset...')
        bbox = None
        bbox_path = self.config.actor + "/bbox.pt"

        if os.path.exists(bbox_path):
            bbox = torch.load(bbox_path)

        for imagepath in tqdm(self.images):
            lmk_path = imagepath.replace('source', 'kpt').replace('png', 'npy').replace('jpg', 'npy')
            lmk_path_dense = imagepath.replace('source', 'kpt_dense').replace('png', 'npy').replace('jpg', 'npy')

            if not os.path.exists(lmk_path) or not os.path.exists(lmk_path_dense):
                image = cv2.imread(imagepath)
                h, w, c = image.shape

                if bbox is None and self.config.crop_image:
                    lmk, _ = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # estimate initial bbox
                    bbox = get_bbox(image, lmk, bb_scale=self.config.bbox_scale)
                    torch.save(bbox, bbox_path)

                if self.config.crop_image:
                    image = crop_image_bbox(image, bbox)
                    if self.config.image_size[0] == self.config.image_size[1]:
                        image = squarefiy(image, size=self.config.image_size[0])
                else:
                    image = cv2.resize(image, (self.config.image_size[1], self.config.image_size[0]), interpolation=cv2.INTER_CUBIC)

                lmk, dense_lmk = self.process_face(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if lmk is None:
                    logger.info(f'Empty face_alignment lmks for path: ' + imagepath)
                    lmk = np.zeros([68, 2])

                if dense_lmk is None:
                    logger.info(f'Empty mediapipe lmks for path: ' + imagepath)
                    dense_lmk = np.zeros([478, 2])

                Path(lmk_path).parent.mkdir(parents=True, exist_ok=True)
                Path(lmk_path_dense).parent.mkdir(parents=True, exist_ok=True)
                Path(imagepath.replace('source', 'images')).parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(imagepath.replace('source', 'images'), image)
                np.save(lmk_path_dense, dense_lmk)
                np.save(lmk_path, lmk)
```

​	基本逻辑是先用ffmpeg对视频进行抽帧，然后对每一帧的图片进行处理，分别用`face_alignment`和Mediapipe提取landmark和dense landmark。当然，在这之前会通过bounding box对输入图片进行裁剪（`self.config.crop_image`默认为True），bounding box当然也是用`face_alignment`库给估计的。这些数据会按如下结构存下来：

```
subject
 ├── images		# cropped and resized images
 ├── kpt		# landmarks from face_alignment detector
 ├── kpt_dense	# landmarks from Mediapipe
 └── source		# original images
```

​	有了存好了的图片和两款landmark，接下来`self.prepare_data()`会再实例化出来一个dataloader来读数据。代码里叫作`ImageDataset`，但它传出来的不止有image：

```python
def __getitem__(self, index):
	...
    shape = None

    shape_path = Path(self.source, 'identity.npy')
    if shape_path.exists():
    	shape = np.load(shape_path)
    else:
    	logger.error('[ImagesDataset] Shape (identity.npy) not found! Run MICA shape predictor from https://github.com/Zielon/MICA')
    	exit(-1)
	...
    shapes = torch.from_numpy(shape).float()

    payload = {
    'image': image,
    'lmk': lmks,
    'dense_lmk': dense_lmks,
    'shape': shapes
    }

    return payload
```

​	它会把之前在MICA里处理得到的`identity.npy`也一并取出来。

​	然后在初始化tracking的`self.initialize_tracking()`，会对所选取的“关键帧”（一般就是第一帧，或者你从这段视频中截取用来计算`identity.npy`的那一帧）进行一下“预热”。这里最关键的是要进行一次`self.optimize_camera`来大致先优化出相机系统，至少得保证人头能比例合适的打在显示屏上。同时，在`self.optimize_camera`里会调用一次`self.create_parameters`对参数进行初始化：

```python
def create_parameters(self):
    bz = 1
    R, T = look_at_view_transform(dist=1.0)
    self.R = nn.Parameter(matrix_to_rotation_6d(R).to(self.device))
    self.t = nn.Parameter(T.to(self.device))
    self.shape = nn.Parameter(self.mica_shape)	  # this is the identity.npy
    self.mica_shape = nn.Parameter(self.mica_shape)  # this is the identity.npy
    self.tex = nn.Parameter(torch.zeros(bz, self.config.tex_params).float().to(self.device))
    self.exp = nn.Parameter(torch.zeros(bz, self.config.num_exp_params).float().to(self.device))
    self.sh = nn.Parameter(torch.zeros(bz, 9, 3).float().to(self.device))
    self.focal_length = nn.Parameter(torch.tensor([[5000 / self.get_image_size()[0]]]).to(self.device))
    self.principal_point = nn.Parameter(torch.zeros(bz, 2).float().to(self.device))
    self.eyes = nn.Parameter(torch.cat([matrix_to_rotation_6d(I), matrix_to_rotation_6d(I)], dim=1))
    self.jaw = nn.Parameter(matrix_to_rotation_6d(I))
    self.eyelids = nn.Parameter(torch.zeros(bz, 2).float().to(self.device))
```

​	这里罗列的就是tracker对每一帧要优化的所有参数。其中`look_at_view_transform`是PyTorch3D提供的计算相机外参（w2c）的接口，但这里只输入了`dist`（可以理解为相机半径），没有输入角度等，所以就是初始化了一个旋转矩阵$R$和平移向量$t$：
$$
R=\left[ \begin{matrix}
	-1&		0&		0\\
	0&		1&		0\\
	0&		0&		-1\\
\end{matrix} \right] 
\\
t=\left[ 0,0,1 \right] ^T
$$
​	$R$里的负号大概是因为在PyTorch3D的坐标系规定下，世界坐标系转相机坐标系有一个沿Y轴顺时针转90°的事情，右手定则一下可知方向为负，所以有两个-1。

<center>
    <img src='/images/head2/head2_2.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

​	

​	`principal_point`是相机系统中的$(u_0,v_0)$，用于锚定图像中心。

​	这里的`matrix_to_rotation_6d`，指的是2019年的CVPR中的一篇[文章](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf)，对深度学习中如何学习“旋转”的一个结论。这个6D的说法是，考虑一个旋转矩阵，只优化它的第一列和第二列一共6个参数，所以叫6D，然后第三维用施密特正交化解算出来：
$$
\mathcal{F} \left( \left[ \begin{matrix}
	\mid&		\mid\\
	\alpha _1&		\alpha _2\\
	\mid&		\mid\\
\end{matrix} \right] \right) =\left[ \begin{matrix}
	\mid&		\mid&		\mid\\
	\beta _1&		\beta _2&		\beta _3\\
	\mid&		\mid&		\mid\\
\end{matrix} \right]
\\
\beta _1=\alpha _1
\\
\beta _2=\alpha _2-\frac{\left< \alpha _2,\beta _1 \right>}{\left< \beta _1,\beta _1 \right>}\beta _1
\\
\beta _3=\alpha _3-\frac{\left< \alpha _3,\beta _1 \right>}{\left< \beta _1,\beta _1 \right>}\beta _1-\frac{\left< \alpha _3,\beta _2 \right>}{\left< \beta _2,\beta _2 \right>}\beta _2
$$
​	但据一些网上的评论，优化这个跟优化四元数也差不多。然后我们可以看一下`self.optimize_camera`里的优化逻辑：

```python
self.cameras = PerspectiveCameras(
    device=self.device,
    principal_point=self.principal_point,
    focal_length=self.focal_length,
    R=rotation_6d_to_matrix(self.R), T=self.t,
    image_size=self.image_size
)
_, lmk68, lmkMP = self.flame(cameras=torch.inverse(self.cameras.R), shape_params=self.shape, expression_params=self.exp, eye_pose_params=self.eyes, jaw_pose_params=self.jaw)
points68 = self.cameras.transform_points_screen(lmk68)[..., :2]
pointsMP = self.cameras.transform_points_screen(lmkMP)[..., :2]

losses = {}
losses['pp_reg'] = torch.sum(self.principal_point ** 2)
losses['lmk68'] = util.lmk_loss(points68, landmarks[..., :2], [h, w], lmk_mask) * self.config.w_lmks
losses['lmkMP'] = util.lmk_loss(pointsMP, landmarks_dense[..., :2], [h, w], lmk_dense_mask) * self.config.w_lmks

all_loss = 0.
for key in losses.keys():
all_loss = all_loss + losses[key]
losses['all_loss'] = all_loss

optimizer.zero_grad()
all_loss.backward()
optimizer.step()
scheduler.step()
```

​	`PerspectiveCameras`是PyTorch3D中将相机系统进行封装，便于执行透视变换的一个类，旋转矩阵，平移向量，焦距等都作为可优化的参数。然后根据此时的flame系数（表情，形状，姿态）匹配出`lmk68`和`lmkMP`，前者还是之前RingNet里的那个，后者原理一样，但后者的那个连接Mediapipe和FLAME之间的embedding应该是他们自己做的。

> 这两个pipeline或多或少都对标准的FLAME的调用进行了修改，仔细看代码会体会到，比如有的禁用了某些参数，有的返回值有一些差异，以及不同管线里选取的基向量的数量也不同。

​	总之拿到了这两个landmark以后，调用`PerspectiveCameras`里的变换，就能计算出在screen space里的坐标了，然后再和之前在生成数据集时，从2D image里预测出的landmark作比较，然后梯度下降，借此更新FLAME的系数。这种做法本质上就是用**预训练好的landmark检测器来对齐2D image和FLAME上的关键点。**

​	而且事实上，这样做能成功的原因是FLAME当时在做这个参数化模型的时候，就用的是在$r=1$的轨迹上的相机来单位化的。所以在用`R, T = look_at_view_transform(dist=1.0)`来给定相机初始位置时，基本也能得到一个大差不差的人头。这也是FLAME里坐标的量级在：

```
v 0.065016 -0.010475 -0.049408
v 0.068556 -0.011430 -0.048024
v 0.068983 -0.009959 -0.047681
...
v 0.039229 0.045869 0.037599
v 0.046781 0.044529 0.032074
v 0.045524 0.039050 0.035560
```

这个范围的原因。

​	这里我们其实注意到了一个细节，`self.flame`在输入时：

```python
_, lmk68, lmkMP = self.flame(cameras=torch.inverse(self.cameras.R),
                             shape_params=self.shape,
                             expression_params=self.exp,
                             eye_pose_params=self.eyes,
                             jaw_pose_params=self.jaw)
```

关于FLAME的姿态，只输入了眼睛和下巴，以及`self.eyes`和`self.jaw`在初始化时实际上是：

```python
self.eyes = nn.Parameter(torch.cat([matrix_to_rotation_6d(I), matrix_to_rotation_6d(I)], dim=1))
self.jaw = nn.Parameter(matrix_to_rotation_6d(I))
```

​	所以他们用的其实是那个6D表示，并不是FLAME/SMPL中常用的轴角式。而且缺少了颈部和根结点。

> FLAME里一共应该有5个结点，一个rot，一个neck，一个jaw，一个left-eye，一个right-eye。

​	我们可以在`FLAME.py`中`FLAME`的`forward`方法里看到，neck和rot已经按缺省时初始化为单位阵处理了。同时由于这里用了6D表示，在`lbs.py`里也有相应的修改：

```python
# rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])
rot_mats = rotation_6d_to_matrix(pose.view(-1, 6)).view([batch_size, -1, 3, 3])
```

​	另一个在原版FLAME上的改动是对于`self.eyelids`，虽然在优化相机时没有输入它，但在后面正式开始优化时，眼皮的系数也会送进去。原版的FLAME并没有规定用参数来驱动眼皮，所以这里的操作是load进两个`.npy`文件，一个是`l_eyelid.npy`，另一个是`r_eyelid.npy`。这两个都是大小为[5023, 3]的numpy array，只不过在大部分的索引下值都为0，只有在对应眼皮的位置的值不为0，同时在`FLAME`的`forward`里，用：

```python
# Use linear blendskinning to model pose roations
vertices, _ = lbs(betas, full_pose, template_vertices,
    self.shapedirs, self.posedirs,
    self.J_regressor, self.parents,
    self.lbs_weights, dtype=self.dtype)

if eyelid_params is not None:
    vertices = vertices + self.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
    vertices = vertices + self.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]
```

​	来将LBS后的顶点的眼皮处进行进一步修饰。所以`eyelids`其实就是学习两个参数来放大和缩小那几个特殊位置的结点。

​	在`self.initialize_tracking`的最后，会执行一次`self.save_canonical`，会将在关键帧下“校准”后的形状系数存下来。但很烦的是它在保存时用的是`trimesh.Trimesh`，用这个库来组织`.obj`，保存纹理坐标什么的比较费劲。建议换用PyTorch3D里的`save_obj`。

​	优化相机只会被调用一次，后面反复被调用的是`self.optimize_color`，

```python
def optimize_color(self, batch, pyramid, params_func, pho_weight_func, reg_from_prev=False):
    self.update_prev_frame()
    images, landmarks, landmarks_dense, lmk_dense_mask, lmk_mask = self.parse_batch(batch)

    aspect_ratio = util.get_aspect_ratio(images)
    h, w = images.shape[2:4]
    logs = []

    for k, level in enumerate(pyramid):
        img, iters, size, image_size = level
        # Optimizer per step
        optimizer = torch.optim.Adam(params_func())
        params = optimizer.param_groups

        shape = self.get_param('shape', params)
        exp = self.get_param('exp', params)
        eyes = self.get_param('eyes', params)
        eyelids = self.get_param('eyelids', params)
        jaw = self.get_param('jaw', params)
        tex = self.get_param('tex', params)
        sh = self.get_param('sh', params)
        t = self.get_param('t', params)
        R = self.get_param('R', params)
        fl = self.get_param('focal_length', params)
        pp = self.get_param('principal_point', params)

        scale = image_size[0] / h
        self.diff_renderer.set_size(size)
        self.debug_renderer.rasterizer.raster_settings.image_size = size
        flipped = torch.flip(img, [2, 3])

        image_lmks68 = landmarks * scale
        image_lmksMP = landmarks_dense * scale
        left_iris = batch['left_iris'] * scale
        right_iris = batch['right_iris'] * scale
        mask_left_iris = batch['mask_left_iris'] * scale
        mask_right_iris = batch['mask_right_iris'] * scale

        self.diff_renderer.rasterizer.reset()

        best_loss = np.inf

        for p in range(iters):
            if p % self.config.raster_update == 0:
                self.diff_renderer.rasterizer.reset()
            losses = {}
            self.cameras = PerspectiveCameras(
                device=self.device,
                principal_point=pp,
                focal_length=fl,
                R=rotation_6d_to_matrix(R), T=t,
                image_size=(image_size,)
            )
            vertices, lmk68, lmkMP = self.flame(
                cameras=torch.inverse(self.cameras.R),
                shape_params=shape,
                expression_params=exp,
                eye_pose_params=eyes,
                jaw_pose_params=jaw,
                eyelid_params=eyelids
            )

            proj_lmksMP = self.cameras.transform_points_screen(lmkMP)[..., :2]
            proj_lmks68 = self.cameras.transform_points_screen(lmk68)[..., :2]
            proj_vertices = self.cameras.transform_points_screen(vertices)[..., :2]

            right_eye, left_eye = eyes[:, :6], eyes[:, 6:]

            # Landmarks sparse term
            losses['loss/lmk_oval'] = util.oval_lmk_loss(proj_lmks68, image_lmks68, image_size, lmk_mask) * self.config.w_lmks_oval
            losses['loss/lmk_68'] = util.lmk_loss(proj_lmks68, image_lmks68, image_size, lmk_mask) * self.config.w_lmks_68
            losses['loss/lmk_MP'] = util.face_lmk_loss(proj_lmksMP, image_lmksMP, image_size, True, lmk_dense_mask) * self.config.w_lmks
            losses['loss/lmk_eye'] = util.eye_closure_lmk_loss(proj_lmksMP, image_lmksMP, image_size, lmk_dense_mask) * self.config.w_lmks_lid
            losses['loss/lmk_mouth'] = util.mouth_lmk_loss(proj_lmksMP, image_lmksMP, image_size, True, lmk_dense_mask) * self.config.w_lmks_mouth
            losses['loss/lmk_iris_left'] = util.lmk_loss(proj_vertices[:, left_iris_flame, ...], left_iris, image_size, mask_left_iris) * self.config.w_lmks_iris
            losses['loss/lmk_iris_right'] = util.lmk_loss(proj_vertices[:, right_iris_flame, ...], right_iris, image_size, mask_right_iris) * self.config.w_lmks_iris

            # Reguralizers
            losses['reg/exp'] = torch.sum(exp ** 2) * self.config.w_exp
            losses['reg/sym'] = torch.sum((right_eye - left_eye) ** 2) * 8.0
            losses['reg/jaw'] = torch.sum((I6D - jaw) ** 2) * self.config.w_jaw
            losses['reg/eye_lids'] = torch.sum((eyelids[:, 0] - eyelids[:, 1]) ** 2)
            losses['reg/eye_left'] = torch.sum((I6D - left_eye) ** 2)
            losses['reg/eye_right'] = torch.sum((I6D - right_eye) ** 2)
            losses['reg/shape'] = torch.sum((shape - self.mica_shape) ** 2) * self.config.w_shape
            losses['reg/tex'] = torch.sum(tex ** 2) * self.config.w_tex
            losses['reg/pp'] = torch.sum(pp ** 2)

            # Dense term (look at the config pyr_levels)
            if k > 0 or self.is_initializing:
                albedos = self.flametex(tex)
                ops = self.diff_renderer(vertices, albedos, sh, self.cameras)

                # Photometric dense term
                grid = ops['position_images'].permute(0, 2, 3, 1)[:, :, :, :2]
                sampled_image = F.grid_sample(flipped, grid * aspect_ratio, align_corners=False)

                losses['loss/pho'] = util.pixel_loss(ops['images'], sampled_image, self.parse_mask(ops, batch)) * pho_weight_func(k)

            all_loss = self.reduce_loss(losses)
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            for key in losses.keys():
                self.writer.add_scalar(key, losses[key], global_step=self.global_step)

            self.global_step += 1

            if p % iters == 0:
                logs.append(f"Color loss for level {k} [frame {str(self.frame).zfill(4)}] =" + reduce(lambda a, b: a + f' {b}={round(losses[b].item(), 4)}', [""] + list(losses.keys())))

            loss_color = all_loss.item()

            if loss_color < best_loss:
                best_loss = loss_color
                self.update(optimizer.param_groups)

    for log in logs: logger.info(log)
```

​	这个函数是优化的主体，为了让tracking更准，输入的不止是当前帧的图像，其实是当前帧图像的高斯金字塔。这是一个常用的技巧，通过从多尺度下的优化来获得更精确的结果。在每个层级下，都进行一轮的优化。

​	注意这里进行了对虹膜的优化，这就是之前`left_iris_flame`，`right_iris_flame`，`left_iris_mp`和`right_iris_mp`的用处了，虹膜直接就可以从Mediapipe的dense landmark和FLAME的顶点拓扑里查表查出来了。于是预训练好的Mediapipe中的虹膜检测模型，就可以“监督”现在复原的FLAME准不准了。

​	这里有一处细节，在优化的时候，对于形状系数，我们使用的是`self.shape`，而最开始的`identity.npy`是作为`self.mica_shape`存在的，其只作为正则项：

```python
losses['reg/shape'] = torch.sum((shape - self.mica_shape) ** 2) * self.config.w_shape
```

​	也就是说每一帧的shape还是会有轻微差别的。

​	我们注意到，与原始图像做loss的“color loss”（或者叫感知损失），其预测出的图像是由：

```python
albedos = self.flametex(tex)
ops = self.diff_renderer(vertices, albedos, sh, self.cameras)

# Photometric dense term
grid = ops['position_images'].permute(0, 2, 3, 1)[:, :, :, :2]
sampled_image = F.grid_sample(flipped, grid * aspect_ratio, align_corners=False)
```

​	这样获得的，`F.grid_sample`在之前的blog里已经用过许多次了，我们这里是要探明一下这个`self.flametex(tex)`。这个机制其实是FLAME后续更新的，我们可以在FLAME官网里下到一个叫`TextureSpace.zip`的东西，里面有一个1.2GB的文件`FLAME_texture.npz`。这个文件是通过用FLAME拟合FFHQ数据集，从而得到一个纹理空间，也就是我们想要的UV map。和处理表情，形状时一样，也进行了主成分分析，得到了表示纹理空间的各个基。用Python打开这个文件，可以看到里面压缩着`['vt.npy', 'ft.npy', 'tex_dir.npy', 'mean.npy']`这四个数组，`vt`就是我们熟悉的作UV映射的纹理坐标，形状是[5118, 2]；`ft`用于描述一个面的三个点，对应着哪些纹理坐标，大小为[9976, 3]；`tex_dir`就是得到的关于纹理空间的基向量，大小为[512, 512, 3, 200]，200即是主成分的数量，在Metrical-tracker这个项目里只取了前140个。然后`mean`就是所谓平均脸，大小是[512, 512, 3]。

<center>
    <img src='/images/head2/head2_3.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	左侧就是那张平均脸的纹理图，右侧是前12个纹理空间上的基向量。

​	实际上优化当前帧下的纹理是最耗时的，如果我只是想要形状和表情，要不要这个纹理的监督其实有点“subtle”。但由于管线的这一部分本身是为了做face tracking的，所以也是很正常的。

​	这里有一个关键的事情是，第$i+1$帧的参数是从第$i$帧优化出来的参数里微调过来的，这是很自然而且方便收敛的。

​	优化完每一帧，都会把那一整套我们关心的参数存下来：

```python
def save_checkpoint(self, frame_id):
    opencv = opencv_from_cameras_projection(self.cameras, self.image_size)

    frame = {
        'flame': {
            'exp': self.exp.clone().detach().cpu().numpy(),
			...
            'jaw': self.jaw.clone().detach().cpu().numpy()
        },
        'camera': {
            'R': self.R.clone().detach().cpu().numpy(),
			...
            'pp': self.principal_point.clone().detach().cpu().numpy(),
        },
        'opencv': {
            'R': opencv[0].clone().detach().cpu().numpy(),
			...
            'K': opencv[2].clone().detach().cpu().numpy(),
        },
        'img_size': self.image_size.clone().detach().cpu().numpy()[0],
        'frame_id': frame_id,
        'global_step': self.global_step
    }

    vertices, _, _ = self.flame(
        cameras=torch.inverse(self.cameras.R),
        shape_params=self.shape,
        expression_params=self.exp,
        eye_pose_params=self.eyes,
        jaw_pose_params=self.jaw,
        eyelid_params=self.eyelids
    )

    f = self.diff_renderer.faces[0].cpu().numpy()
    v = vertices[0].cpu().numpy()

    trimesh.Trimesh(faces=f, vertices=v, process=False).export(f'{self.mesh_folder}/{frame_id}.ply')
    torch.save(frame, f'{self.checkpoint_folder}/{frame_id}.frame')
```

​	文件后缀为`.frame`，以及在该帧下的FLAME拓扑（用LBS计算出的顶点们）也会被存下来，存成`.ply`。同时还会存一些杂七杂八的东西，不过不是很重要。

​	注意在存的时候，还多存了一份opencv约定下的相机，实际上后面最后用到INSTA里的也是opencv版的，很多框架之间的相机坐标系并不一样，有时候会很烦：（[图源自](https://zhuanlan.zhihu.com/p/593204605/)）

<center>
    <img src='/images/head2/head2_5.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

​	据此，在一些民间群里延申出了一张梗图：

<center>
    <img src='/images/head2/head2_convention.jpg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

​	因为很多时候，比坐标系变换更迷的事情是：“我根本不知道我所在的仓库用的是哪个coordinate systems！”

​	至此，文件结构成了：

```
Subject
    ├── camera				# several camera projection result for debug
    ├── checkpoint			# *.frame file 
    ├── depth				# depth maps rendered by diff_render
    ├── initialization		# output of initial_tracking, for debug
    ├── input				# cropped and resized images
    ├── logs	
    ├── mesh				# .ply
    ├── pyramid				# several pyramid demo for debug
    ├── train.log
    ├── video				# tracking video frames
    ├── canonical.mtl		# if use PyTorch3D to save obj
    ├── canonical.obj
    ├── canonical.png		# if use PyTorch3D to save obj
    └── video.avi
```

​	最后的那个`video.avi`是tracking结果的可视化：

<center>
<video id="video" controls style="max-width: 800px; max-height: 400px; width: auto; height: auto;">
    <source id="mp4" src="/images/head2/head2_4.mp4" type="video/mp4" >
</video>
</center>


​	最左边是对齐后的图像，第二个是FLAME的纹理空间拟合出的结果，第三个是landmarks。其中绿色的可以理解为“gt”，因为他们是用Mediapipe和`face_alignment`估计出来的。红色的是预测出的landmarks，通过计算出顶点位置后，用一些embedding查表查出来的，然后最右边就是此时的FLAME拓扑了。

​	这里需要指出一些事情，首先，这个tracker最后输出的6D表示，不能无缝衔接到最主流的那个`FLAME`类和`lbs.py`的实现里。其次，这种tracking并不是对我们理解的FLAME的真实还原，这么说的是因为，在我们想象中的FLAME表示里，应该是人头发生旋转，比如旋转rot和neck来做出姿势，而相机始终是不变的。但这里其实我们没有建模FLAME里的rot pose，这是因为估计的结果是为了训练NeRF（instant-ngp）方便，所以这里其实是认为人头不曾转动，是相机的角度在转，变换上是等价的。另外，FLAME中的neck pose被忽略了，这个也比较微妙，因为后面在作face parsing的时候，脖子的部分其实就被截掉一块了。

​	然后就来到了INSTA下的重新组织数据集的环节，第一步是从那些`*.frame`里拆解数据，调用`dump_frame`：

```python
def dump_frame(payload):
    frame, src, output = payload
    payload = torch.load(frame)
    frame_id = payload['frame_id']
    mesh_path = frame.replace('.frame', '.ply').replace(checkpoint, 'mesh')
    if not os.path.exists(mesh_path):
        return None

    trimesh.load(mesh_path, process=False).export(f'{output}/meshes/' + frame_id + '.obj')

    depth_path = frame.replace(checkpoint, 'depth').replace('.frame', '.png')
    if os.path.exists(depth_path):
        os.system(f'cp {depth_path} {output}/depth/{frame_id}.png')

    img = f'images/{frame_id}.png'
    depth = f'depth/{frame_id}.png'

    # Flame
    dump_flame(payload['flame'], frame_id, output)

    oepncv = payload['opencv']
    R = oepncv['R'][0]
    t = oepncv['t'][0]

    # Extrinsic
    w2c = np.eye(4)
    w2c[0:3, 0:3] = R
    w2c[0:3, 3] = t

    c2w = np.linalg.inv(w2c)

    data_frame = {
        'transform_matrix': c2w,
        'file_path': img,
        'mesh_path': f'meshes/{frame_id}.obj',
        'exp_path': f'flame/exp/{frame_id}.txt',
        'depth_path': depth,
        'seg_mask_path': depth.replace('depth', 'seg_mask')
    }

    return data_frame
```

​	在dump每一帧时，也会把每一帧的FLAME系数都储存下来：

```python
def dump_flame(flame, frame_id, output):
    # all_params = ['exp', 'shape', 'tex', 'sh', 'eyes', 'eyelids', 'jaw']
    params = ['exp', 'eyes', 'eyelids', 'jaw']
    for param in params:
        coeff = flame[param]
        coeff = coeff[0].flatten('F')
        dump_text(f'{output}/flame/{param}/{frame_id}.txt', coeff)
```

​	要注意到最后存下来的是相机外参的逆（c2w），相机内参由另外一个函数来储存，用第一帧中相机内参的结果来代表所有帧下的相机内参。最后外参连同内参划分训练，验证，测试集后储存成`.json`。

​	到这一步，所有关于相机系统，FLAME系数的预处理已经完成了。接下来就是处理对齐（切割和裁剪）后的图像，一般来说是要把人头从背景里抠图（matting）出来，然后有时候想去掉脖子以下（下至肩膀）的部分，所以会用语义分割（semantic segmentation）得到哪个部位在哪，然后作“解析”（parsing）：

<center>
    <img src='/images/head2/head2_6.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

​	将人从视频的每一帧里扣出来的用的是[RobustVideoMatting(RVM)](https://github.com/PeterL1n/RobustVideoMatting)，然后语义分割是通过一个在CelebA-HQ上训练BiSeNet的[仓库](https://github.com/zllrunning/face-parsing.PyTorch)实现的。这两篇工作都来自于那个还能叠积木的时代，大概是经过了马普所人员的检验，所以就这么用了。

​	最后，INSTA的数据格式就是：

```
Subject
├── depth				# depth map rendered by diff_render
├── flame				# flame coeff
│   ├── exp				# experssion, [F, 100]
│   ├── eyelids			# eyelids coeff, [F, 2]
│   ├── eyes			# eyes pose(right and left), [F, 12], 6D representation
│   └── jaw				# jaw pose, [F, 6], 6D representation
├── images				# images after matted and parsed
├── matted				# images after matted
├── meshes				# mesh in every frame
├── seg_mask			# segmentation mask for each frame
├── canonical.obj
├── transforms.json
├── transforms_train.json
├── transforms_val.json
└── transforms_test.json
```

​	实际上最后一部分的parsing和文件结构里要存`meshes`只是INSTA的需要，从普适的完成单目视频重建来看的话，倒也不需要。在INSTA里想在“脸部区域”维持FLAME的几何先验（因为FLAME里没有头发，把全部pixel都这么做会让效果劣化），就这么一个loss，至少十年功力：
$$
\mathcal{L} _{\mathrm{gemo}}=\sum_{\mathbf{r}}{\left| \mathbb{1} _{\mathrm{face}}\left\{ \left( z\left( \mathbf{r} \right) -\hat{z}\left( \mathbf{r} \right) \right) \right\} \right|}
$$
​	因为实现这么一个loss，就需要在预处理的时候多导出每一帧时的mesh和seg_mask，将光线投射的终点与标准mesh光栅化后得到的深度作对齐。我根本不知道怎么在instant-ngp的框架下体面的实现这个功能，还有那个在canonical space下来做BVH（bounding volume hierarchy）的操作。很遗憾，以前花大量的时间炼没用的丹来着，现在已经到Gaussian Splatting的时代了，不知道什么时候能学一手这些。

### Point-Avatar

​	说实话，我应该一开始就follow这篇工作的预处理管线的，因为这个管线更注重于提取做单目视频重建的信息，运行起来没有优化一整套，以及这里是直接用DECA的结果（再稍微平滑一下），所以会快很多。

​	在Point-Avatar之前，这个组还有一篇叫[IMavatar](https://github.com/zhengyuf/IMavatar/tree/main)的工作，数据预处理的管线是继承过来的。在其仓库的`./preprocess`下写的很清楚，而且环境和一些需要下载的东西准备好了以后一键启动`preprocess.sh`就好了，写的很清楚，比INSTA要跳转好几个仓库要清晰。

​	首先，用ffmpeg抽帧，然后用一个现成的抠图网络[MODNet](https://github.com/ZHKKKe/MODNet)来分割出人头来。有趣的是MODNet实际上是刚才提到的RVM正文里的baseline。

​	其主体是用DECA来估计一整套参数。DECA在上一篇blog里提到了，用Encode-Decode的范式实现了从单张图片里估计FLAME系数和相机参数。这里我们着重剖析一些细节：

<center>
    <img src='/images/head2/head2_7.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	我们再回顾一下DECA的管线，通过$\mathrm{E}_{\mathrm{c}}$我们可以从图像中得到一些属性的*latent variables*，所以IMavatar预处理的第一步是运行`/demos/demo_reconstruct.py`来得到这些隐变量，或者说叫“code”（编码）。这里形状系数shape code $\beta\in\mathbb{R} ^{100}$，表情系数expression code $\psi\in\mathbb{R} ^{50}$，这两个是“straight-forward”的，可以直接带入FLAME。然后这里的pose code $\theta\in\mathbb{R}^{6}$，并不是$\mathbb{R} ^{15}$。因为DECA里忽略了眼球和脖子的旋转（6+3)，在DECA的`FLAME.py`的构造方法中：

```python
default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                 requires_grad=False))
default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                  requires_grad=False))
```

​	在forward的时候，对neck和eyeballs保持了缺省：

```python
batch_size = shape_params.shape[0]
if full_pose is None:
    if pose_params is None:
        pose_params = self.eye_pose.expand(batch_size, -1)
    if eye_pose_params is None:
        eye_pose_params = self.eye_pose.expand(batch_size, -1)
    if neck_pose_params is None:
        neck_pose_params = self.neck_pose.expand(batch_size, -1)
    full_pose = torch.cat([pose_params[:, :3], neck_pose_params, pose_params[:, 3:], eye_pose_params], dim=1)
```

​	以及通过考察`lbs.py`，可以知道这里pose就是用的轴角式表达。

​	这里还剩一下一个比较奇怪的camera pose $c\in\mathbb{R}^3$，如果我们检视大多数$c$，一般第一个值是9.0~10.0，然后第二个和第三个值很接近0。这个说起来比较复杂。我们注意到在前面的$\theta$里，我们是建模了根结点的旋转，或者叫全局旋转的。这是符合我们想法的，所以相机外参的旋转默认为一个单位阵。但这并不意味着$c$就是相机内参，但它确实也和相机内参有关。事情是这样的，我们输入给DECA时的图像并不是对齐的，一般情况下：

<center>
    <img src='/images/head2/head2_8.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	比如输入512×512的图像，我们会先利用人脸检测器把人脸给裁下来，得到只含人脸的子图。这张子图的大小在DECA是224×224，标定原图和子图上的三个点，在`./decalib/datasets/datasets.py`中可以导引出一个齐次变换：

```python
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
else:
    ...

DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
tform = estimate_transform('similarity', src_pts, DST_PTS)
```

​	比如在上面的图中，人头的比例基本是不变的，所以要做的其实就是一个剪切（shear）变换，这种情况下`estimate_transform`最小二乘出来的就是：

```python
tensor([[ 9.7807e-01,  3.1584e-16, -1.2275e+02],
        [-1.9529e-16,  9.7807e-01, -9.6340e+01],
        [ 0.0000e+00,  0.0000e+00,  1.0000e+00]])
```

​	然后变换和裁剪的过程会由`skimage.transform`里的`warp`完成，就得到了右图。

​	所以DECA对latent code的获得是从右图来得，而FLAME模板里的坐标，范围一般在±0.1周围。然后，$c$的目的在于对这些坐标进行变换，使其变换后可以贴到原来的那个图上。我们以`landmark2d`为例，最终的目的是让得到的`landmark2d`和原图可以像下图那样匹配：

<center>
    <img src='/images/head2/head2_9.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>


​	我们考察最左边的关键点，其$x\approx155.2$。而其本身从`verts, landmarks2d, landmarks3d = self.flame(...) `里投影出来的时候，约为$-0.0772$。在`deca.decode`里，这个`landmarks2d`在从FLAME里出来以后，会先进行`util.batch_orth_proj`，然后又有个`transform_points`。在这个函数里面有两个很奇怪的操作：

```python
def transform_points(points, tform, points_scale=None, out_scale=None):
    points_2d = points[:,:,:2]
    #'input points must use original range'
    if points_scale:
        assert points_scale[0]==points_scale[1]
        points_2d = (points_2d*0.5 + 0.5)*points_scale[0]
    # import ipdb; ipdb.set_trace()

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
                    torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), 
                    tform
                    ) 
    if out_scale: # h,w of output image size
        trans_points_2d[:,:,0] = trans_points_2d[:,:,0]/out_scale[1]*2 - 1
        trans_points_2d[:,:,1] = trans_points_2d[:,:,1]/out_scale[0]*2 - 1
    trans_points = torch.cat([trans_points_2d[:,:,:2], points[:,:,2:]], dim=-1)
    return trans_points
```

​	`point_scale`是`[224, 224]`，`h`和`w`都是512。所以整个过程是：
$$
\left( \left( \frac{\left( \left( \left( x+t \right) \cdot s\cdot 0.5+0.5 \right) \times 224 \right) \cdot \mathcal{S} +\mathcal{T}}{512} \right) \times 2-1 \right) \times 256+256\approx155.2
$$
​	为了优化上式中的$t$和$s$，其中$\mathcal{S}$和$\mathcal{C}$是刚才估计出来的那个齐次变换的逆：

```
tensor([[[ 1.0224e+00,  2.0415e-16,  0.0000e+00],
         [-3.3016e-16,  1.0224e+00,  0.0000e+00],
         [ 1.2550e+02,  9.8500e+01,  1.0000e+00]]], device='cuda:0')
```

​	可以看到$\mathcal{S}\approx1.0224, \mathcal{T}\approx125$。我们可以忽略掉$c$中的$t$，它的预测值其实是$0.004187$。那样我们直接带入$x=-0.0772$，就可以得到$s\approx9.46$，这回答了为什么$c$中的第一个元素总是接近9~10。

​	总之我们现在对`deca.encode`出的东西有了充分的认识，然后下面在这个管线里要进行的是`optimize.py`，这里我们会发现，在这个脚本的入口，手动规定了内参：

```python
parser.add_argument('--fx', type=float, default=1500)
parser.add_argument('--fy', type=float, default=1500)
parser.add_argument('--cx', type=float, default=256)
parser.add_argument('--cy', type=float, default=256)
parser.add_argument('--size', type=int, default=512)
```

​	跟这个内参对应的是代码里会将相机据原点的距离先给成`torch.tensor([0, 0, -4]).float().cuda()`，这是INSTA预处理管线里的四倍。这个有点一言难尽，这是因为在`optimize.py`里顶点什么的规模也都被乘4了。我不知道作者是不是在IMAvatar的时候就这么写了，我也不是很好说明这么写的意义是什么。只是在PointAvatar里，由于最开始是初始化一个半径为0.5的球面点云，所以将FLAME的顶点们乘四倍正好想凑个单位球也算是合理……吧？我不知道在这种优化的情景和规模下，以factor=4来scale或者不scale……变化会很大吗？

```python
# CAREFUL: FLAME head is scaled by 4 to fit unit sphere tightly
verts_p *= 4
landmarks3d_p *= 4
landmarks2d_p *= 4
```

​	由于在DECA里已经估计了人头的全局旋转，所以这里对于相机外参的处理是只学习平移向量$\mathbf{t}$，旋转一直是一个单位阵。所以在IMAvatar和PointAvatar中，其实并不存在“cam pose”这一概念。在这两篇工作中，人头的变形和旋转，都是将固定的，基于FLAME的点$x_c$输入进MLP中，来解算出实现“变形”的属性：
$$
x_d=\mathrm{LBS}\left( x_c+B_P\left( \theta ;\mathcal{P} \right) +B_E\left( \psi ;\mathcal{E} \right) ,J\left( \psi \right) ,\theta ,\mathcal{W} \right)
$$
​	$x_c$在IMAvatar中是用Ray Marching技术，找到的点；而在PointAvatar里，则就是点的坐标本身。$\theta,\psi$分别是FLAME的姿态和表情系数，这些是显式的输入，在训练时可以通过预处理后的数据中获得，在测试时可以手动设定新视角和表情。$\mathcal{P},\mathcal{E}$是用MLP计算出的blendshape，$\mathcal{W}$在这里专指对某个特定的$x_c$而言的关于不同结点的权重，也是用MLP学到的。$J(\psi)$指的是在有$B_E\left( \psi ;\mathcal{E} \right) $修正下估计出的结点位置，这一步在实现上是通过维护一个标准FLAME模板得到的。相比于INSTA的那种做法，我个人更喜欢这种方式来实现变形。虽然这种变形本质上也是炼丹炼出来，但就是个人情感上更好接受。

<center>
    <img src='/images/head2/head2_implict_morphing.png' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>

​	这个操作相比于标准FLAME的变形（Morphing），可以称其为“Implicit Morphing”，因为相比于标准的，回归出来的$\mathcal{E},\mathcal{P},\mathcal{W}$，我们在这里需要用learning-based的方法来给FLAME中没有的点“reassign”那些属性。

​	说回`optimize.py`本身，`optimize.py`的优化的目标主要是相邻帧之间的系数变化不要太大，以及通过landmarks之间的损失来对齐$\mathbf{t}$：

```python
landmark_loss2 = lossfunc.l2_distance(trans_landmarks2d[:, :len_landmark, :2], landmark[:, :len_landmark])
total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * 1e-2 + torch.mean(torch.square(exp)) * 1e-2
total_loss += torch.mean(torch.square(exp[1:] - exp[:-1])) * 1e-1
total_loss += torch.mean(torch.square(pose[1:] - pose[:-1])) * 10
```

​	然后其实就结束了，虹膜检测和人脸分割虽然在IMAvatar的预处理里实装了，但在IMAvatar和PointAvatar里其实没有用到。总之最后的数据集结构就是：

```
subject
 ├── deca
 ├── image
 ├── mask
 ├── code.json
 ├── keypoint.json
 └── flame_params.json
```

​	其中`code.json`里记录着DECA对每一帧的估计结果，`flame_params.json`是通过`optimize.py`以后进一步处理后的参数，`deca`目录下实际上是空的，大概是中间结果给删了。

### Discussion

​	这就是这两个工作的预处理管线了，我想在这儿再提及一下这两个管线里都用到的关键点损失（keypoints error）：
$$
\mathcal{L} _{kp}=\left\| K\left( \theta ,\psi ,\beta ,t \right) -K_{\mathrm{target}} \right\| _2
$$
​	我们已经知道了，$K\left( \theta ,\psi ,\beta ,t \right)$是从FLAME中“mapping”出来的关键点，然后$K_{\mathrm{target}}$是用一些预训练好的模型，不管是Mediapipe里的detector还是`face_alignment`库的landmark detectors。用这种loss来抽先验，然后进行优化是非常直接的想法，可能在之前人脸的一些领域这个loss已经被当成司空见惯的佐料了，优化的时候加点进去刷刷点。但我想说，这种思想非常的有用，有用到它直接开辟了3D生成的一种新的方法：从预训练的diffusion中抽先验。
$$
\mathcal{L} _{\mathrm{SDS}}=\hat{\epsilon}_{\phi}\left( z_t;y,t \right) -\epsilon 
$$
只不过这一子领域实在是发展太快了，他们现在大概也不是只用这个简单的loss了。

> 我推测SDS loss并不是噪声的L1或L2范数的原因，是因为在diffusion model中的噪声信号的另一个诠释就是其分布的对数概率密度函数的梯度。这个梯度的方向具有一定的指导意义，所以直接相减取noise residual就好了。不过这里说这个太跑题了。

​	所以为什么要干巴巴的写出这么一篇blog呢，因为有一天晚上下班回家，我发现GAMES Webinar有个直播，最后里面有段话：

> “所以我觉得对于刚入门的同学来说，可能一些知识上的积累，要远比于快速的卷一两篇paper可能更重要一些吧。所以当你有了这些经验和知识以后，可能现阶段你卷不过那些更senior的老师同学，但未来某一天你可能有一些新的任务新的工具出来之后，你就可以快速的应用过去。”

​	这大概就是全部了。

### End

​	“若是从未启程的故事，能够有结局的话，残留枯枝也会绽放吧。”

<center>
    <img src='/images/head2/head2_end.jpg' style="max-width: 800px; max-height: 600px; width: auto; height: auto;">
</center>
