---
title: DarkNet源码分析(#TODO)
mathjax: true
date: 2023-04-10 21:46:01
tags: [深度学习]
categories: 深度学习
---

​	自打我稀里糊涂把自己的专业变成：人工智能以来，已经过去了快三年。有很长一段时间，都是用PyTorch, matlab这种解释型语言。已经很久很久没有像我的一个朋友一样，怼着几百个头文件嗯看了。

<!--more-->

​	实际上，在很早以前。我其实巧妙地误入过一次“歧途”，就是在大一有个院里的小比赛里。要往一个小车里部署目标检测。当时由于啥也不会，各种版本的问题没有装上torch。误打误撞的装上了DarkNet框架下的YOLOV3。

​	一个好处就是，DarkNet是纯C语言编写的，而且相对简单小巧，有很好的学习意义。基本可以获得如下的收获：

- 对optim.step(), loss.backward(), compution graph有个更实际的认识
- 掌握大型项目的调试（gdb）
- 复习一下C语言

​	“*现在现在向下穿越 向下穿越* ”

### 进入项目&命令行传参

​	因为，我已经很久没有写过C语言了。所以我可能会同时记录一些很细节的事情，因为我可能确实不知道。

​	我们可以从DarkNet的[repo](https://github.com/pjreddie/darknet)直接git下来。如果你和我一样，只是想在Windows下来看源码，建议用Cygwin来编译，至少我这样就不会报错了。

​	darknet提供了Python的接口，至于，这个操作的具体原理我后面再看，现在看来就像魔法。总之我们可以一般通过的运行：

```
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
```

​	这个预测的指令，是用来作出那个经典的狗狗图的预测的。我们就拿它来一步步进入代码里吧。

​	对于每一个C语言的项目，我们都最好先找到它的入口，我们可以在`./examples/darknet.c`里找到main函数：

```c
int main(int argc, char **argv)
{
    //test_resize("data/bad.jpg");
    //test_box();
    //test_convolutional_layer();
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) {
        gpu_index = -1;
    }

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cuda_set_device(gpu_index);
    }
#endif

    if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "yolo")){
        run_yolo(argc, argv);
    } else if (0 == strcmp(argv[1], "super")){
        run_super(argc, argv);
    } else if (0 == strcmp(argv[1], "lsd")){
        run_lsd(argc, argv);
    } else if (0 == strcmp(argv[1], "detector")){
        run_detector(argc, argv);
    } else if (0 == strcmp(argv[1], "detect")){
        float thresh = find_float_arg(argc, argv, "-thresh", .5);
        char *filename = (argc > 4) ? argv[4]: 0;
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    } else if (0 == strcmp(argv[1], "cifar")){
        run_cifar(argc, argv);
    } else if (0 == strcmp(argv[1], "go")){
        run_go(argc, argv);
    } else if (0 == strcmp(argv[1], "rnn")){
        run_char_rnn(argc, argv);
    } else if (0 == strcmp(argv[1], "coco")){
        run_coco(argc, argv);
    } else if (0 == strcmp(argv[1], "classify")){
        predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    } else if (0 == strcmp(argv[1], "classifier")){
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "regressor")){
        run_regressor(argc, argv);
    } else if (0 == strcmp(argv[1], "isegmenter")){
        run_isegmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "segmenter")){
        run_segmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "art")){
        run_art(argc, argv);
    } else if (0 == strcmp(argv[1], "tag")){
        run_tag(argc, argv);
    } else if (0 == strcmp(argv[1], "3d")){
        composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
    } else if (0 == strcmp(argv[1], "test")){
        test_resize(argv[2]);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "rgbgr")){
        rgbgr_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "reset")){
        reset_normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "denormalize")){
        denormalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "statistics")){
        statistics_net(argv[2], argv[3]);
    } else if (0 == strcmp(argv[1], "normalize")){
        normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "rescale")){
        rescale_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "ops")){
        operations(argv[2]);
    } else if (0 == strcmp(argv[1], "speed")){
        speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
    } else if (0 == strcmp(argv[1], "oneoff")){
        oneoff(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "oneoff2")){
        oneoff2(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "print")){
        print_weights(argv[2], argv[3], atoi(argv[4]));
    } else if (0 == strcmp(argv[1], "partial")){
        partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "visualize")){
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else if (0 == strcmp(argv[1], "mkimg")){
        mkimg(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]);
    } else if (0 == strcmp(argv[1], "imtest")){
        test_resize(argv[2]);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
```

​	在这里（至少在我这里，Windows下），`./darknet`其实是一个可执行文件darknet.exe，所以命令行输入的第一个参数也正是程序本身的名字。`test`可以猜出来是决定模式的字符串。后面那几个路径决定了一些杂七杂八的事情（用哪个数据集cfg，用哪个权重...）。

​	我们关注argv[1] == detector时，接下来我们会进入run_detector()：(我们只保留我们需要阅读的部分了，不然太长了。)

```c
void run_detector(int argc, char **argv)
{
    ......
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    ......
	char *outfile = find_char_arg(argc, argv, "-out", 0);
    ......
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    ......

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
	......
}
```

​	在我们进一步进入test_detector()之前，我们可以停一下来看一下在run_detector()和main()中时不时出现的，帮我们解析命令行参数的辅助函数们(我来通过他们复习一下指针的知识)：

```c
void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i) {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

float find_float_arg(int argc, char **argv, char *arg, float def)
{
	......
}

char *find_char_arg(int argc, char **argv, char *arg, char *def)
{
	......
}
```

​	实际上，为了夯实一下我不怎么存在的基础，我仔细探究了一下这一过程。首先我们要仔细考察一下char* argv[]和char ** argv。如果只是作为函数传参，那么这二者是等价的。就像find_arg()和find_int_arg()的传参一样，是没有区别的。

​	但是，char* argv[]实际上是声明了一个数组argv，该数组保存多个指向char类型的指针。char **argv是声明argv是指向“指向char类型”的指针。前者会声明一个数组，所以会在内存中分配连续的一段空间。例如：

```C
#include <stdio.h>

int main() {
    char* strings[] = {"Hatsune", "Miku", "!"};
    int num_strings = sizeof(strings) / sizeof(strings[0]);
    
    for (int i = 0; i < num_strings; i++) {
        printf("strings[%d]: %p\n", i, strings[i]);
    }
    return 0;
}

# output:
# strings[0]: 0000000000404000
# strings[1]: 0000000000404008
# strings[2]: 000000000040400D
#
# &strings[0]: 000000000062FDF0
# &strings[1]: 000000000062FDF8
# &strings[2]: 000000000062FE00
```

​	可以看到，我们打印了出了strings[i]所指向的地址，这是连续的。strings[0]的首地址是4000, 而strings[1]是4008，这是因为strings[0]由7个char（1个字节1个char）组成，然后被程序自己追加了一个\Null以表分割。所以是8个字节。同理，4008+5=400D，也就得到了strings[2]的开头。

​	然而，如果我们想用char **argv来办到这点，我们需要用malloc()手动分配空间：

```C
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main() {
	char **strings = (char **)malloc(sizeof(char *) * 3);
	strings[0] = (char *)malloc(sizeof(char) * 6);
	strings[1] = (char *)malloc(sizeof(char) * 6);
	strings[2] = (char *)malloc(sizeof(char) * 2);

	strcpy(strings[0], "Hatsune");
	strcpy(strings[1], "Miku");
	strcpy(strings[2], "!");

	for (int i = 0; i < 3; i++) {
	    printf("strings[%d]: %p\n", i, strings[i]);
	}
	printf("\n");
	for (int i = 0; i < 3; i++) {
	    printf("&strings[%d]: %p\n", i, &strings[i]);
	}
	printf("\n");
	return 0;
}

# output:
# strings[0]: 00000000006D6A60
# strings[1]: 00000000006D6A80
# strings[2]: 00000000006D6AA0
#
# &strings[0]: 0000000000B86A40
# &strings[1]: 0000000000B86A48
# &strings[2]: 0000000000B86A50
```

​	此时分配的并不连续，而是隔了一个固定的间隔。这是为了*内存对齐*。所以实际上，当我们通过终端把那些字符输入进去。是操作系统帮我们创建了指针char **argv，并且根据输入给他们自动分配空间。

​	注意，这两种方式开辟的指针数组，他们本身其实都是连续的。所以我们都可以用argv[1]这样的方式来索引它。它们两两之间都隔着8位，因为char*是8位的。

​	所以，回过头来看那些解析命令行参数的工具函数，它们的原理就是把命令行参数的字符数组传入，如果匹配到需要的，就返回它，同时用后面的字符串覆盖它，最后空出来的那个置零。

​	注意那些试图匹配特殊的一些输入时用到的find_int_arg(), find_float_arg(), *find_char_arg()，我们发现它调用了两次del_arg()，这是为了同时删除选项，例如float thresh = find_float_arg(argc, argv, "-thresh", .5);中，命令行一定是以这样的形式输入的：xxx -thresh 0.8 xxxx

​	而此处预设的0.5，实际上对应find_float_arg()里默认的def参数，效果和我们在python里用argparse，设置default是一样的。

### 数据结构

​	现在我们进入test_detector()函数：

```c
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}
```

​	现在，信息量略微大了一些。第一行里突然出现了个"list"，这可不是python，list是需要自己定义了。这里的list是一个双向链表，在`darknet.h`里定义了：

```c
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;
```

​	node结构体下的数据类型是任意的，因为其是void型指针。在list.c中，定义了初始化这样双向链表的函数：

```c
list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}
```

​	我们通过读命令行，可以知道在这里read_data_cfg()传入的是字符数组`cfg/coco.data`，而这个最为我们遇到的第一个读取配置文件的操作，我们当然需要进一步探究一下：

```c
list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}
```

​	这个函数的输入大概是内容为这样的文件：

```
classes= 80
train  = /home/pjreddie/data/coco/trainvalno5k.txt
valid  = coco_testdev
#valid = data/coco_val_5k.list
names = data/coco.names
backup = /home/pjreddie/backup/
eval=coco
```

​	所以read_data_cfg()会返回一个双向链表的指针，switch-case中会检查是否是空行，注释行。如果不是，就用read_option

