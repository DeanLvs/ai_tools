let socket = io();
let room_id = checkUUID();
setupSocketListeners(socket, room_id);

let clickCount = 0;
let clickVCount = 0;
let previousSeeds = new Set();
let isDrawing = false;
let isErasing = false;
let brushSize = 20; // Default brush size set to maximum
        let ctx;
function handleClick() {
    clickCount++;
    if (clickCount === 3) {
        toggleHiddenFields();
        clickCount = 0;
    }
}

function handleVClick() {
    clickVCount++;
    if (clickVCount === 3) {
        hideOverlay();
        clickVCount = 0;
    }
}
function showOverlay() {
    document.getElementById('overlay').style.display = 'flex';
}

function hideOverlay() {
    document.getElementById('overlay').style.display = 'none';
}

function processImage_b_face() {
    showOverlay();
    const filename_handler = document.getElementById('filename').value;
    const filename_face = document.getElementById('filename_face').value;
    const room_id = checkUUID();
    checkSocket();
    socket.emit('process_replace_image_b', { room_id, filename_face: filename_face, filename_handler: filename_handler });

    setupSocketListeners(socket, room_id);
}

function uploadImage_face() {
    showOverlay();
    const fileInput_handler_face = document.getElementById('fileInput_handler_face');
    const file_img_h = fileInput_handler_face.files[0];

    const formData = new FormData();
    formData.append('file_face_img', file_img_h);
    room_id = checkUUID();
    formData.append('room_id', room_id);
    fetch('/api/rep_upload_face', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const specialImageListDiv = document.getElementById('specialImageList'); // 新的图片列表容器
        add_img_to_list('/uploads' + '/' + room_id + '/' + data.file_face_url, specialImageListDiv)
        document.getElementById('filename_face').value = data.filename_face;
        hideOverlay();
    })
    .catch(error => {
        console.error('Error:', error);
        hideOverlay();
    });
}

function add_img_to_list(url_path, specialImageListDiv){
    // 设置图片标题
    let title;
    const fileName = url_path.split('/').pop();
    if (fileName.startsWith('r_f_')) {
        title = "历史换脸图";
    } else if (fileName.startsWith('filled_use_mask_image_pil_')) {
        title = "识别处理区域";
    } else if (fileName.startsWith('clear_return_')) {
        title = "清理区域结果";
    } else if (fileName.startsWith('filled_image_pil_next_')) {
        title = "无差别填充";
    } else if (fileName.startsWith('filled_image_pil_')) {
        title = "计算区域填充";
    } else if (fileName.startsWith('fill_all_skin_')) {
        title = "无差别填充所有";
    } else if (fileName.startsWith('finaly_')) {
        title = "换脸成图";
    } else if (fileName.startsWith('control_net_')) {
        title = "姿势识别";
    } else if (fileName.startsWith('nor_control_net_')) {
        title = "光照识别";
    } else {
        title = "book成图";
    }

    // 添加新的图片元素
    const imgContainer = document.createElement('div');
    imgContainer.classList.add('image-container');

    const img = document.createElement('img');
    img.src = url_path;
    img.alt = '处理后的图片';
    img.onclick = () => showFullscreenImage(url_path);
    const caption = document.createElement('p');
    caption.innerText = title;
    caption.style.color = '#f1f1f1'; // 设置标题文字颜色
    caption.style.textAlign = 'center'; // 设置标题居中

    imgContainer.appendChild(img);
    imgContainer.appendChild(caption);

    specialImageListDiv.appendChild(imgContainer);
}


function uploadImage() {
    showOverlay();
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    room_id = checkUUID();
    formData.append('room_id', room_id);
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const uploadedImage = document.getElementById('uploadedImage');
        uploadedImage.src = '/uploads' + '/' + room_id + '/' + data.file_url;
        uploadedImage.style.display = 'block';
        const img = new Image();
        img.src = '/uploads' + '/' + room_id + '/' + data.file_url;
        document.getElementById('filename').value = data.filename;
        document.getElementById('regenerateButton_b').style.display = 'none';
        document.getElementById('processImageButton_b').style.display = 'block';
        hideOverlay();
    })
    .catch(error => {
        console.error('Error:', error);
        hideOverlay();
    });
}

function startDrawing(event) {
    isDrawing = true;
    const pos = getMousePos(event);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

function stopDrawing() {
    isDrawing = false;
}

function draw(event) {
    if (!isDrawing) return;
    const pos = getMousePos(event);
    ctx.lineTo(pos.x, pos.y);
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = isErasing ? '#1c1c1c' : '#00FF00';  // Use green for drawing, background color for erasing
    ctx.stroke();
}

function setBrushSize() {
    brushSize = document.getElementById('brushSize').value;
}

function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function refreshImageList() {
    // 实现重新获取图片列表的逻辑
    // 例如，可以重新从服务器请求数据，或者其他刷新列表的逻辑
    console.log('Image list refreshed.');
}

function setupSocketListeners(socket, room_id) {
    socket.on('connect', () => {
        const session_id = socket.id;
        console.log('Connected with session ID:', session_id);
        socket.emit('join_room', { roomId: room_id });
    });


    socket.on('disconnect', () => {
        room_id = checkUUID();
        console.log('Socket disconnected. Attempting to reconnect...');
        reconnectSocket(room_id);
    });

    socket.on('processing_step_progress', (data) => {
        document.getElementById('overlay').innerText = data.text + '-点三次蒙层关闭';
    });
    socket.on('processing_step_fin', (data) => {
        const processButton = document.getElementById('processImageButton_b');
        // 检查 processImageButton_b 是否可见
        if (processButton.style.display === 'none') {
            // 只有当 processImageButton_b 不可见时才执行
            document.getElementById('regenerateButton_b').style.display = 'block';
            processButton.style.display = 'none';
        }
    });
    socket.on('processing_progress', (data) => {
        const progressBar = document.querySelector('.progress-bar');
        progressBar.style.width = `${data.progress}%`;
        progressBar.innerText = `${data.progress}%`;
        if (data.had) {
            progressBar.innerText = progressBar.innerText + ' ' + data.had
        }
    });
    socket.on('processing_progress', (data) => {
        const progressBar = document.querySelector('.progress-bar');
        progressBar.style.width = `${data.progress}%`;
        progressBar.innerText = `${data.progress}%`;
        if (data.had) {
            progressBar.innerText = progressBar.innerText + ' ' + data.had
        }
    });
    socket.on('processing_done_key', (data) => {
        document.getElementById('handler-key').innerText = '记住这个key，等不及可以20分钟后会在用key查看: ' + data.processing_done_key;
        hideOverlay();
    });

    socket.on('processing_done_text', (data) => {
<!--                document.getElementById('overlay').innerText = data.text;-->
        hideOverlay();
    });

    // 监听服务器发送的 'online_users' 事件
    socket.on('online_users', (data) => {
        const onlineUsersElement = document.getElementById('onlineUsers');
        onlineUsersElement.innerText = `在线人数: ${data.count}`;
    });

    socket.on('processing_done', (data) => {
<!--                document.getElementById('progress').innerText = '处理结果列表';-->
        const imageListDiv = document.getElementById('imageList');
        const specialImageListDiv = document.getElementById('specialImageList'); // 新的图片列表容器
        // 检查是否已有相同的图片存在于 specialImageListDiv
        const existingSpecialImages = specialImageListDiv.querySelectorAll('img');
        if (!data.processed_image_url.startsWith('/uploads')) {
            data.processed_image_url = '/uploads' + '/' + room_id + '/' + data.processed_image_url;
        }
        existingSpecialImages.forEach(img => {
            const imgSrcPath = new URL(img.src).pathname; // 提取路径部分
            if (imgSrcPath === data.processed_image_url) {
                img.parentElement.remove(); // 移除图片所在的容器
            }
        });

        // 检查是否已有相同的图片存在
        const existingImages = imageListDiv.querySelectorAll('img');
        existingImages.forEach(img => {
            const imgSrcPath = new URL(img.src).pathname; // 提取路径部分
            if (imgSrcPath === data.processed_image_url) {
                img.parentElement.remove(); // 移除图片所在的容器
            }
        });

        let isSpecialImage = false;
        const fileName = data.processed_image_url.split('/').pop();
        if (fileName.startsWith('filled_use_mask_image_pil_')) {

            isSpecialImage = true;
        } else if (fileName.startsWith('clear_return_')) {

            isSpecialImage = true;
        } else if (fileName.startsWith('filled_image_pil_')) {

            isSpecialImage = true;
        }else if (fileName.startsWith('control_net_')) {

            isSpecialImage = true;
        }

        // 根据 isSpecialImage 将图片存储到不同的图片列表中
        if (isSpecialImage) {
           add_img_to_list(data.processed_image_url, specialImageListDiv)
        } else {
           add_img_to_list(data.processed_image_url, imageListDiv)
        }

        // 触发重新获取逻辑（例如刷新页面或其他）
        refreshImageList();

        const progressBar = document.querySelector('.progress-bar');
        progressBar.style.width = '0%';
        progressBar.innerText = '';
        if (!data.keephide) {
            hideOverlay();
        }
    });
}

function load_lora_b() {
    showOverlay();
    let roomId = checkUUID();
    let lora_id = document.getElementById('lora_id').value;
    let wei_id = document.getElementById('wei_id').value;
    checkSocket();
    socket.emit('process_set_lora', { roomId, lora_id, wei_id});
}

function processImage_b() {
    showOverlay();
    const filename = document.getElementById('filename').value;
    if (!filename || filename.trim() === '') {
        alert('你啥也没上传呢还');
        // 这里可以添加其他处理逻辑，比如阻止表单提交等
    }
    const prompt = document.getElementById('prompt').value;
    const reverse_prompt = document.getElementById('reverse_prompt').value;
    const prompt_2 = document.getElementById('prompt_2').value;
    const reverse_prompt_2 = document.getElementById('reverse_prompt_2').value;

    const num_inference_steps = document.getElementById('num_inference_steps').value;
    const guidance_scale = document.getElementById('guidance_scale').value;
    const seed = document.getElementById('seed').value;

    const re_p = document.getElementById('re_p').value;
    const re_b = document.getElementById('re_b').value;
    const ha_p = document.getElementById('ha_p').value;
    const ga_b = document.getElementById('ga_b').value;
    const re_mask = document.getElementById('re_mask').value;
    const strength = document.getElementById('strength').value;
    const def_skin = document.getElementById('def_skin').value;
    const room_id = checkUUID();
    checkSocket();
    socket.emit('process_image_b', { filename, prompt, reverse_prompt, prompt_2, reverse_prompt_2, num_inference_steps, guidance_scale, seed, re_p, re_b, ha_p, ga_b, re_mask, strength,def_skin, roomId: room_id });
    document.getElementById('processImageButton_b').style.display = 'none';
    setupSocketListeners(socket, room_id);
}

function checkSocket(){
    if (!socket || !socket.connected) {
        console.log('Attempting to reconnect...');
        initializeSocket(room_id);
    }
}

function reconnectSocket(room_id) {
    if (!socket || !socket.connected) {
        console.log('Attempting to reconnect...');
        initializeSocket(room_id);
    }
    socket.emit('re_get', { roomId:room_id });
}

function initializeSocket(room_id) {
    if (socket) {
        socket.disconnect(); // 断开现有的 socket 连接
    }
    socket = io();
    setupSocketListeners(socket, room_id);
}

async function exampleFunction() {
    console.log('开始');
    await sleep(2000);  // 暂停2秒
    console.log('2秒后');
}

function toggleHiddenFields() {
    const hiddenFields = document.querySelectorAll('#bookYesSection input, #bookYesSection button');
    hiddenFields.forEach(field => {
        if (field.type === "hidden" || field.style.display === "none") {
            field.type = "text";
            field.style.display = "block";
        } else {
            field.type = "hidden";
            field.style.display = "none";
        }
    });
    const specialImageListTitle_t = document.getElementById('specialImageListTitle');
    const specialImageList_t = document.getElementById('specialImageList');
    specialImageListTitle_t.style.display = "block";
    specialImageList_t.style.display = "block";
}

function setCookie(name, value, days) {
    const d = new Date();
    d.setTime(d.getTime() + (days*24*60*60*1000));
    const expires = "expires="+ d.toUTCString();
    document.cookie = name + "=" + value + ";" + expires + ";path=/";
}

function getCookie(name) {
    const nameEQ = name + "=";
    const ca = document.cookie.split(';');
    for(let i=0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) === ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(nameEQ) === 0) {
            return c.substring(nameEQ.length, c.length);
        }
    }
    return null;
}

function generateUniqueSeed() {
    let seed;
    do {
        seed = Math.floor(Math.random() * 500); // 生成 0 到 100 的随机数
    } while (previousSeeds.has(seed)); // 确保种子不重复
    previousSeeds.add(seed); // 记录新的种子
    return seed;
}

function regenerateImage_b() {
    // 生成新的随机种子
    const seed = generateUniqueSeed();
    document.getElementById('seed').value = seed;

    // 调用 processImage 处理图像
    processImage_b();
}

function checkUUID() {
    let uuid = getCookie("uuid");
    if (uuid === null) {
        uuid = uuidv4();
        setCookie("uuid", uuid, 365); // 存储一年
    }
    return uuid;
}

function viewImage() {
    const room_id = checkUUID();
    showOverlay();
    reconnectSocket(room_id);
}

function showFullscreenImage(url) {
    const fullscreenOverlay = document.querySelector('.fullscreen-overlay');
    const fullscreenImage = document.getElementById('fullscreenImage');
    fullscreenImage.src = url;
    fullscreenOverlay.style.display = 'flex';
}

function closeFullscreenImage() {
    const fullscreenOverlay = document.querySelector('.fullscreen-overlay');
    fullscreenOverlay.style.display = 'none';
}

function closeOverlay(){
    handleVClick();
}