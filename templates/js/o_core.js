let socket = io();
let room_id = checkUUID();
setupSocketListeners(socket, room_id);

let B = 1; // 从1开始的状态位
let oddImageList = []; // 存储奇数次选择的图片src
let evenImageList = []; // 存储偶数次选择的图片src
let selectImageList = [];
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
    const filename_handler = document.getElementById('filename').value;
    const filename_face = document.getElementById('filename_face').value;
    if (!filename_handler || filename_handler.trim() === '' || !filename_face || filename_face.trim() === '') {
        alert('你需要选择一个待处理图片，和一个有脸的图');
        // 这里可以添加其他处理逻辑，比如阻止表单提交等
        return;
    }
    showOverlay();
    const room_id = checkUUID();
    checkSocket();
    org_faces = [filename_handler]
    to_faces = [filename_face]
    filename = filename_handler
    socket.emit('process_pic_swap_face', { org_faces, to_faces, filename ,roomId: room_id});

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
        add_img_to_list('/uploads' + '/' + room_id + '/' + data.file_face_url, specialImageListDiv, '')
        document.getElementById('filename_face').value = data.filename_face;
        hideOverlay();
    })
    .catch(error => {
        console.error('Error:', error);
        hideOverlay();
    });
}

function handleSelectClick(imgSrc, button) {
    const fileName = imgSrc.split('/').pop();
    const currentBgColor = button.style.backgroundColor; // 当前按钮的背景颜色

    // 检查按钮是否已经被选择，如果是，就取消选择并从列表中移除
    if (currentBgColor === 'blue') {
        // 将按钮恢复为未选择状态
        button.style.backgroundColor = '#ccc';
        button.innerText = '选择';
        selectImageList = selectImageList.filter(item => item !== fileName);
        if(selectImageList.length > 0){
            document.getElementById('filename').value = selectImageList[selectImageList.length - 1];
        }else{
            document.getElementById('filename').value = '';
        }
        return;
    }
    button.style.backgroundColor = 'blue';
    button.innerText = '取消';
    selectImageList.push(fileName);
    if(selectImageList.length > 0){
        document.getElementById('filename').value = selectImageList[selectImageList.length - 1];
    }
}

function handleSelectListClick(imgSrc, button) {
    const fileName = imgSrc.split('/').pop();
    const currentBgColor = button.style.backgroundColor; // 当前按钮的背景颜色

    // 检查按钮是否已经被选择，如果是，就取消选择并从列表中移除
    if (currentBgColor === 'blue' || currentBgColor === 'green') {
        // 将按钮恢复为未选择状态
        button.style.backgroundColor = '#ccc';

        // 从奇数列表或偶数列表中移除 fileName
        if (currentBgColor === 'blue') {
            // 从奇数列表移除
            oddImageList = oddImageList.filter(item => item !== fileName);
        } else if (currentBgColor === 'green') {
            // 从偶数列表移除
            evenImageList = evenImageList.filter(item => item !== fileName);
        }
        console.log('取消选择:', fileName);
        console.log('奇数选择的图片:', oddImageList);
        console.log('偶数选择的图片:', evenImageList);
        return;
    }
    // 增加状态位 B
    B++;
    // 检查状态位是奇数还是偶数
    if (B % 2 === 1) {
        // 奇数，按钮颜色为蓝色
        button.style.backgroundColor = 'blue';
        // 将图片src放入奇数列表
        oddImageList.push(fileName);
    } else {
        // 偶数，按钮颜色为绿色
        button.style.backgroundColor = 'green';
        // 将图片src放入偶数列表
        evenImageList.push(fileName);
    }
    document.getElementById('filename').value = fileName;
    console.log('当前状态位 B:', B);
    console.log('奇数选择的图片:', oddImageList);
    console.log('偶数选择的图片:', evenImageList);
}

function resetSelectClick(imgSrc, button) {
    // 增加状态位 B
    B = 1;
    oddImageList = []; // 存储奇数次选择的图片src
    evenImageList = []; // 存储偶数次选择的图片src

    console.log('当前状态位 B:', B);
    console.log('奇数选择的图片:', oddImageList);
    console.log('偶数选择的图片:', evenImageList);
}

function add_img_to_list(url_path, specialImageListDiv, p_name){
    // 设置图片标题
    let title;
    const fileName = url_path.split('/').pop();
    if (p_name != '') {
        title = p_name;
    } else if (p_name = ''){
        title = "处理结果P";
    } else{
        title = "处理结果";
    }
    // 添加新的图片元素
    const imgContainer = document.createElement('div');
    imgContainer.classList.add('image-container');
    if (fileName.endsWith('.mp4')) {
        // 创建video元素
        const video = document.createElement('video');
        video.src = url_path;
        video.controls = true;
        video.alt = '处理后的视频';
        video.style.maxWidth = '100%'; // 确保视频不会超出容器宽度
        video.style.display = 'block'; // 将视频作为块级元素显示
        imgContainer.appendChild(video);
    } else {
        const img = document.createElement('img');
        img.src = url_path;
        img.alt = '处理后的图片';
        img.onclick = () => showFullscreenImage(url_path, fileName);
        imgContainer.appendChild(img);
        // 创建选择按钮
        const selectButton = document.createElement('button');
        selectButton.innerText = '选择';
        selectButton.classList.add('select-button');
        selectButton.style.backgroundColor = '#ccc'; // 初始按钮颜色
        selectButton.onclick = () => handleSelectClick(img.src, selectButton);
        imgContainer.appendChild(selectButton);

    }
    const caption = document.createElement('p');
    caption.innerText = title;
    caption.style.color = '#f1f1f1'; // 设置标题文字颜色
    caption.style.textAlign = 'center'; // 设置标题居中

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
//        const uploadedImage = document.getElementById('uploadedImage');
//        uploadedImage.src = '/uploads' + '/' + room_id + '/' + data.file_url;
//        uploadedImage.style.display = 'block';
//        const img = new Image();
//        img.src = '/uploads' + '/' + room_id + '/' + data.file_url;
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
//            document.getElementById('regenerateButton_b').style.display = 'block';
//            processButton.style.display = 'none';
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

        // 检查是否已有相同的图片存在
        const existingVideo = imageListDiv.querySelectorAll('video');
        existingVideo.forEach(video => {
            const videoSrcPath = new URL(video.src).pathname; // 提取路径部分
            if (videoSrcPath === data.processed_image_url) {
                video.parentElement.remove(); // 移除图片所在的容器
            }
        });

        let isSpecialImage = false;
        const fileName = data.processed_image_url.split('/').pop();

        // 根据 isSpecialImage 将图片存储到不同的图片列表中
        if (data.img_type == 'pre_done') {
           add_img_to_list(data.processed_image_url, specialImageListDiv, data.name)
        } else {
           add_img_to_list(data.processed_image_url, imageListDiv, data.name)
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

function process_text_gen(userInput){
    const filename = document.getElementById('filename').value;
    if (!filename || filename.trim() === '') {
        alert('先选择一个要处理的图吧');
        // 这里可以添加其他处理逻辑，比如阻止表单提交等
        return;
    }
    showOverlay();
    const room_id = checkUUID();
    checkSocket();
    prompt = userInput
    reverse_prompt = ''
    face_filename = ''
    gen_type = ''
    socket.emit('process_text_gen_pic', { filename, prompt, reverse_prompt, face_filename, gen_type, roomId: room_id });
    document.getElementById('processImageButton_b').style.display = 'none';
    setupSocketListeners(socket, room_id);
}

// 显示悬浮框
function showModal() {
    const filename = document.getElementById('filename').value;
    if (!filename || filename.trim() === '') {
        alert('先选择一个要处理的图吧');
        // 这里可以添加其他处理逻辑，比如阻止表单提交等
        return;
    }
    document.getElementById('myModal').style.display = 'flex';
}

// 关闭悬浮框
function closeModal() {
    document.getElementById('myModal').style.display = 'none';
}

// 点击确定按钮时执行的逻辑
function confirmInput() {
    const userInput = document.getElementById('userInput').value;
    if (userInput.trim() === '') {
        alert('输入不能为空');
        return;
    }
    console.log('用户输入:', userInput);

    // 在这里执行你的方法
    process_text_gen(userInput);

    // 关闭悬浮框
    closeModal();
}


function processImage_b() {
    const filename = document.getElementById('filename').value;
    if (!filename || filename.trim() === '') {
        alert('先选择一个要处理的图吧');
        // 这里可以添加其他处理逻辑，比如阻止表单提交等
        return;
    }
    showOverlay();
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

function showFullscreenImage(url, fileName) {
    const fullscreenOverlay = document.querySelector('.fullscreen-overlay');
    const fullscreenImage = document.getElementById('fullscreenImage');
    document.getElementById('v_filename').value = fileName;
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

function processImage_v() {
    const filename = document.getElementById('v_filename').value;
    if (!filename || filename.trim() === '') {
        alert('你啥也没上传呢还');
        // 这里可以添加其他处理逻辑，比如阻止表单提交等
        return;
    }
     showOverlay();
    const def_skin = '99';
    const room_id = checkUUID();
    checkSocket();
    socket.emit('process_image_b', { filename, def_skin, roomId: room_id });
    document.getElementById('processImageButton_b').style.display = 'none';
    setupSocketListeners(socket, room_id);
}