import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@latest";

const canvas = document.getElementById("canvas");
const progressDialog = document.getElementById("progress-dialog");
const progressIndicator = document.getElementById("progress-indicator");

const renderer = new SPLAT.WebGLRenderer(canvas);
const scene = new SPLAT.Scene();
const camera = new SPLAT.Camera();
const controls = new SPLAT.OrbitControls(camera, canvas);

let isInSpecialArea = false;

canvas.addEventListener("mouseenter", () => {
    isInSpecialArea = true;
});

canvas.addEventListener("mouseleave", () => {
    isInSpecialArea = false;
});

async function main() {
    //const url = "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/bonsai-7k.splat";
	//const url = "/3dgs/bonsai-7k.splat";
	const url = "https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/fufu.ply";
	//const url = "https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/point_cloud.ply"
	
    //await SPLAT.Loader.LoadAsync(url, scene, (progress) => progressIndicator.value = progress * 100);
	await SPLAT.PLYLoader.LoadAsync(url, scene, (progress) => progressIndicator.value = progress * 100);
    progressDialog.close();

    const handleResize = () => {
        renderer.setSize(window.innerWidth, window.innerHeight);
    };

    const frame = () => {
        if (isInSpecialArea) {
            controls.update();
            renderer.render(scene, camera);
        }

        requestAnimationFrame(frame);
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    requestAnimationFrame(frame);
}

main();
