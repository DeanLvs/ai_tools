document.addEventListener('DOMContentLoaded', () => {
    checkUUID(); // 确保 UUID 在页面加载时已设置
    // 为进度条添加点击事件
    const progressBar = document.querySelector('.progress-bar');

    progressBar.addEventListener('click', () => {
        const room_id = checkUUID(); // 获取 UUID
        console.log(`进度条被点击，重新连接 socket。`);
        reconnectSocket(room_id);
    });

    // 设置输入字段的默认值
//    document.getElementById('prompt').value = 'nude,authenticity,natural and plump breasts,natural and plump buttocks'//'nude,authenticity';
//    document.getElementById('reverse_prompt').value = 'no distorted body proportions, no exaggerated limbs, no multiple belly buttons, no deformed limbs, no unnaturally contorted position'; //deformed limbs,disconnected limbs,unnaturally contorted position
//    document.getElementById('prompt_2').value = 'a person with nine-heads tall proportions, balanced proportions';//naked, match the overall lighting angle of the original image,match the overall lighting intensity of the original image
//    document.getElementById('reverse_prompt_2').value = 'no elongated waists, no blurred details, no unnatural body stretching,no unrealistic body features,no overly exaggerated body parts,no incorrect body proportions' //'unrealistic features,overly exaggerated body parts,incorrect proportions';


    // 设置输入字段的默认值
    document.getElementById('prompt').value = 'nude,person,woman,the front of body,authenticity,natural asses,natural boobs';
    document.getElementById('reverse_prompt').value = 'finger,wrist,hazy,malformed,warped,hand,multiple belly buttons,multiple breasts,multiple nipples,deformed limbs,disconnected limbs,contorted position,elongated waists,overly exaggerated body parts,body incorrect proportions';
    document.getElementById('prompt_2').value = '';//naked, match the overall lighting angle of the original image,match the overall lighting intensity of the original image
    document.getElementById('reverse_prompt_2').value = '';

    document.getElementById('num_inference_steps').value = '30';
    document.getElementById('guidance_scale').value = '7';
    document.getElementById('seed').value = '128';
    document.getElementById('re_p').value = "['key_points','depth','canny']";//['key_points','depth','canny']
    document.getElementById('re_b').value = '[0.5, 0.2, 0.5, 0.5]';
    document.getElementById('ha_p').value = '0';
    document.getElementById('ga_b').value = '0.09';
    document.getElementById('re_mask').value = 'F';
    document.getElementById('strength').value = '0.8543';
    document.getElementById('def_skin').value = '2';
    document.getElementById('wei_id').value = '#unet#t1#t2#vae';
    document.getElementById('lora_id').value = 'biggunsxl_v11.safetensors#0.6#biggunii';
    previousSeeds.add(128);

    document.getElementById('fileInput').addEventListener('change', function() {
        uploadImage();
    });

    document.getElementById('fileInput_handler_face').addEventListener('change', function() {
        const fileInput = this;
        const file = fileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const uploadedFaceImage = document.getElementById('uploadedFaceImage');
                uploadedFaceImage.src = e.target.result;
                uploadedFaceImage.style.display = 'block'; // 显示图片
            };
            reader.readAsDataURL(file);
        }
        uploadImage_face();
    });

    document.getElementById('uploadedFaceImage').addEventListener('click', function() {
        const fullscreenImage = document.getElementById('fullscreenImage');
        fullscreenImage.src = this.src; // 将点击的图片路径赋给全屏图片
        document.querySelector('.fullscreen-overlay').style.display = 'flex'; // 显示全屏容器
    });

    function closeFullscreenImage() {
        document.querySelector('.fullscreen-overlay').style.display = 'none'; // 隐藏全屏容器
    }

});