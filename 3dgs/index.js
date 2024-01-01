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

// 显示模态框
function showModal() {
    const modal = document.getElementById("progress-dialog");
    modal.style.display = "block";
}

// 关闭模态框
function closeModal() {
    const modal = document.getElementById("progress-dialog");
    modal.style.display = "none";
}

async function main() {
    //const url = "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/bonsai-7k.splat";
	//const url = "/3dgs/bonsai-7k.splat";
	//const url = "https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/fufu.splat";
	//const url = "https://huggingface.co/datasets/zjwfufu/gsplat_view/resolve/main/point_cloud.ply"
	
	const modelDropdown = document.getElementById("modelDropdown");

    // Set the initial model URL
    const url = modelDropdown.querySelector('.dropdown-item').getAttribute('data-model-url');	
	showModal();
    await SPLAT.Loader.LoadAsync(url, scene, (progress) => progressIndicator.value = progress * 100);
	//await SPLAT.PLYLoader.LoadAsync(url, scene, (progress) => progressIndicator.value = progress * 100);
    // progressDialog.close();
	closeModal();
	progressIndicator.value = 0;

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
	
	// Update scene when model is changed
modelDropdown.addEventListener("click", async (event) => {
    if (event.target.tagName === 'A') {
		scene.reset();
        //progressDialog.classList.add('centered-modal'); // 添加类
        // progressDialog.showModal();
		showModal();

        // Clear the scene before loading the new model
        
        const newSelectedModelUrl = event.target.getAttribute('data-model-url');
        
        await loadAndRenderModel(newSelectedModelUrl);

        // progressDialog.close();
		closeModal();
		progressIndicator.value = 0;
        //progressDialog.classList.remove('centered-modal'); // 移除类
    }
});

    requestAnimationFrame(frame);
}

async function loadAndRenderModel(modelUrl) {
    await SPLAT.Loader.LoadAsync(modelUrl, scene, (progress) => progressIndicator.value = progress * 100);
    // You can perform additional rendering or setup logic here if needed
}

main();
