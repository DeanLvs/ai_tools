let socket = io();
let room_id = checkUUID();
setupSocketListeners(socket, room_id);

function resetSel(){
    // 重置所有按钮为未选择状态
    const allButtons = document.querySelectorAll('.select-button'); // 假设所有选择按钮都有类名 'select-button'
    allButtons.forEach(btn => {
        btn.style.backgroundColor = '#ccc';
        btn.innerText = '选择';
    });
    document.getElementById('filename').value = '';
}

function handleSelectClick(imgSrc, button) {
    const fileName = imgSrc.split('/').pop();
    const currentBgColor = button.style.backgroundColor; // 当前按钮的背景颜色

    // 检查按钮是否已经被选择，如果是，就取消选择并从列表中移除
    if (currentBgColor === 'blue') {
        // 将按钮恢复为未选择状态
        button.style.backgroundColor = '#ccc';
        button.innerText = '选择';
        document.getElementById('filename').value = '';
        return;
    }
    resetSel();
    button.style.backgroundColor = 'blue';
    button.innerText = '取消';
    document.getElementById('filename').value = fileName;
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
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
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
        //提示
        document.getElementById('overlay').innerText = data.text + '-点三次蒙层关闭';
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

    socket.on('processing_done_text', (data) => {
    });

    // 监听服务器发送的 'online_users' 事件
    socket.on('online_users', (data) => {
        const onlineUsersElement = document.getElementById('onlineUsers');
        onlineUsersElement.innerText = `在线人数: ${data.count}`;
    });

    socket.on('processing_done', (data) => {
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
        const progressBar = document.querySelector('.progress-bar');
        progressBar.style.width = '0%';
        progressBar.innerText = '';
        if (!data.keephide) {
        }
    });
}

function processImage_b() {
    const filename = document.getElementById('filename').value;
    if (!filename || filename.trim() === '') {
        alert('先选择一个要处理的图吧');
        // 这里可以添加其他处理逻辑，比如阻止表单提交等
        return;
    }
    const def_skin = document.getElementById('def_skin').value;
    const room_id = checkUUID();
    checkSocket();
    socket.emit('process_image_b', { filename, def_skin, roomId: room_id });
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

function checkUUID() {
    let uuid = getCookie("uuid");
    if (uuid === null) {
        uuid = uuidv4();
        setCookie("uuid", uuid, 365); // 存储一年
    }
    return uuid;
}