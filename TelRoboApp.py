from telegram import Update, Bot, InputMediaPhoto
from telegram import ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from book_yes_logger_config import logger
from CoreService import get_rest_key, req_get_face
from CoreDb import RoomImageManager, query_or_def, user_states_control, user_vip_control
from BookYesCommon import resize_image, add_task_list, glob_task_positions, video_task_positions, glob_task_queue, task_vide_queue, User
import threading, os, asyncio, queue, uuid, shutil
from tenacity import retry, stop_after_attempt, wait_fixed
app_path = '/nvme0n1-disk/book_yes/static/uploads'
bot = Bot(token='7467241687:AAHlU2z43Ks9f-jy8EX78AwOHPrVoO5B0kg')

# 设置重试装饰器：最多重试3次，每次等待2秒
retry_decorator = retry(
    stop=stop_after_attempt(3),  # 最多重试3次
    wait=wait_fixed(1),  # 每次重试前等待2秒
    reraise=True  # 重试失败后抛出最后一次异常
)

def notify_it(iswct='ws', event=None, js_obj=None, to=None, keyList=None, media_group=None):
    if iswct == 'ws':
        logger.info('no notify_it ws ----------')
    elif iswct == 'tel':
        if event == 'processing_progress' and int(js_obj['progress']) == 10:
            prs = js_obj['had']
            # add_task_to_telegram_notify_queue(to, f'{prs}处理中')
        elif event == 'processing_done':
            if js_obj['img_type'] == 'done':
                add_task_to_telegram_notify_queue(to, js_obj, keyList=keyList)
            elif js_obj['img_type'] == 'media_group':
                add_task_to_telegram_notify_queue(to, js_obj, keyList=keyList, media_group=media_group)
        elif event == 'processing_step_progress':
            add_task_to_telegram_notify_queue(to, js_obj['text'], keyList=keyList)
        elif event == 'processing_step_fin':
            add_task_to_telegram_notify_queue(to, '处理完成', keyList=keyList)
@retry_decorator
async def send_video(chat_id, video_url, video_name, reply_markup=None):
    try:
        # 使用 `async with` 确保在整个异步操作期间文件保持打开
        with open(video_url, 'rb') as video:
            await bot.send_video(chat_id=chat_id, video=video, caption=f"处理完成: {video_name}", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"发送视频时出错: {e}")

# 定义一个异步函数来发送图片，确保文件在异步任务执行期间保持打开状态
@retry_decorator
async def send_photo(chat_id, pic_url, pic_name, reply_markup=None):
    try:
        # 使用 `async with` 确保在整个异步操作期间文件保持打开
        with open(pic_url, 'rb') as photo:
            await bot.send_photo(chat_id=chat_id, photo=photo, caption=f"处理完成: {pic_name}", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"发送图片时出错: {e}")
@retry_decorator
async def send_group_photo(chat_id, text, media_group, reply_markup):
    try:
        # 发送图片组
        await bot.send_media_group(chat_id=chat_id, media=media_group)

        # 发送带有选择按钮的消息
        await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"发送图片时出错: {e}")
@retry_decorator
async def send_message_book(chat_id, text, reply_markup):
    try:
        await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"发送图片时出错: {e}")

# 创建一个队列用于存储 Telegram 消息通知任务
telegram_notify_queue = queue.Queue(maxsize=5000)


# 消费者函数：从队列中取出任务并发送消息
def queue_tel_consumer(loop):
    while True:
        # 从队列中获取任务
        task_data = telegram_notify_queue.get()

        # 提取任务中的信息
        chat_id = task_data.get('chat_id')
        message = task_data.get('message')
        keyList = task_data.get('keyList')
        if 'img_type' in message and message['img_type'] == 'media_group':
            media_group = task_data.get('media_group')
            text = message['text']
            future = asyncio.run_coroutine_threadsafe(
                send_group_photo(chat_id, text, media_group, keyList),
                loop
            )

        elif 'img_type' in message and  'processed_image_url' in message and message['img_type'] == 'done':
            pic_path_name = message['processed_image_url']
            pic_name = message['name']
            pic_url = os.path.join(app_path, chat_id, pic_path_name)
            # 发送图片
            if os.path.exists(pic_url):  # 确认图片路径是否存在
                future = asyncio.run_coroutine_threadsafe(
                    send_photo(chat_id, pic_url, pic_name, reply_markup=keyList),  # 调用异步任务
                    loop
                )
            else:
                logger.error(f"图片文件未找到: {pic_url}")
                future = asyncio.run_coroutine_threadsafe(
                    send_message_book(chat_id=chat_id, text=f"图片 {pic_name} 未找到"),
                    loop
                )
        elif 'img_type' in message and 'video_url' in message and message['img_type'] == 'done':
            video_name = message['filename']
            video_url = message['video_url']
            # 发送图片
            if os.path.exists(video_url):  # 确认图片路径是否存在
                future = asyncio.run_coroutine_threadsafe(
                    send_video(chat_id, video_url, video_name, reply_markup=keyList),  # 调用异步任务
                    loop
                )
            else:
                logger.error(f"视频文件未找到: {video_url}")
                future = asyncio.run_coroutine_threadsafe(
                    send_message_book(chat_id=chat_id, text=f"图片 {video_name} 未找到"),
                    loop
                )
        else:
            # 使用 run_coroutine_threadsafe 提交任务到主线程的事件循环
            future = asyncio.run_coroutine_threadsafe(
                send_message_book(chat_id=chat_id, text=message, reply_markup=keyList),
                loop
            )
        try:
            # 阻塞等待任务完成
            future.result()
        except Exception as e:
            logger.error(f"发送消息时出错: {e}")

        # 标记队列任务完成
        telegram_notify_queue.task_done()


# 向队列中添加任务
def add_task_to_telegram_notify_queue(chat_id, message, keyList=None, media_group=None):
    task_data = {
        'chat_id': chat_id,
        'message': message,
        'keyList': keyList,
        'media_group':media_group
    }
    # 将任务放入队列
    telegram_notify_queue.put(task_data)
    logger.info(f"任务已添加到 Telegram 通知队列：chat_id={chat_id}, message={message}")


async def start(message, context) -> None:
    # 发送消息，并显示自定义键盘
    await message.reply_text('功能已列在输入框下方', reply_markup=get_rest_key())


# 错误处理函数
async def error_handler(update: Update, context) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    # 通知用户发生错误
    if update:
        await update.message.reply_text("抱歉，发生了一个错误，请稍后再试。")
async def handle_video(update: Update, context):
    # Get user info and chat id
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    room_id = str(chat_id)
    user_info = query_or_def(User(user_id, room_id))# 获取用户唯一 ID
    user_channel = user_info.channel
    user_input_status = user_info.status
    if user_channel != 'swap_video':
        await update.message.reply_text(f'目前只有视频换脸模型支持处理视频内容，请先再下方切换', reply_markup=get_rest_key())
        return
    if user_input_status != 'awaiting_handler_pic':
        await update.message.reply_text(f'请先完成换脸设置，上传图片识别脸部，选择替换')
        return
    # Get the highest quality version of the uploaded video
    video_file = await update.message.video.get_file()

    # Extract file path and format
    file_path = video_file.file_path
    file_extension = os.path.splitext(file_path)[1]
    logger.info(f'now get video file is {file_path} {file_extension}')
    # Define supported formats
    if file_extension not in ['.mp4', '.avi', '.mov','.MP4', '.AVI', '.MOV']:
        await update.message.reply_text(f'目前不支持该视频格式')
        return
    # 生成 UUID 作为 `callback_data`
    unique_id = str(uuid.uuid4())
    filename = unique_id + file_extension
    logger.info(f'This room id is {room_id}, uploading file: {filename}')

    # Set up paths for saving the video
    file_path = os.path.join(app_path, room_id)
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, filename)
    logger.info(f'Saving video to: {file_path}')
    # Download the video and save to server
    await video_file.download_to_drive(file_path)
    # Notify the user that the video has been saved
    await update.message.reply_text(f'视频已保存到服务器: {filename}，正在处理该视频...')
    org_faces = user_info.org_pic_list
    to_faces = user_info.to_pic_list
    await update.message.reply_text(f'{filename} 开始换脸 {org_faces} {to_faces}')

    data = {'def_skin': 'start_swap_face_video'}
    data['notify_type'] = 'tel'
    data['filename'] = filename
    data['org_faces'] = org_faces
    data['to_faces'] = to_faces
    data['roomId'] = room_id
    logger.info(f'get req is {data}')
    room_image_manager = RoomImageManager()
    add_task_list(data, 'tel_notify_it', task_vide_queue, video_task_positions, 'task_vide_queue', notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)
    # Process the video (You can add your custom video processing logic here)
    # For example, video resizing, format conversion, etc.
    # process_video(file_path)

def trans_status(status_key):
    if status_key == 'only_face_awaiting_input':
        return '等带输入情景描述文本后，生成图片'
    if status_key == 'awaiting_handler_pic':
        return '等带输入待处理内容媒体内容'
    if status_key == 'awaiting_input':
        return '等带输入待处理内容媒体内容'
    return '暂无'
def trans_channel(channel_key):
    if channel_key=='dress_up':
        return '🥼👕重绘换装'
    if channel_key=='special_effects':
        return '🕺自定义情景'
    if channel_key=='swap_pic':
        return '🧑‍🧒📷️图片换脸'
    if channel_key=='swap_video':
        return '🧑‍🧒🎥视频换脸'
    if channel_key=='flux_txt_to_image':
        return '👑Flux模型自定义情景（更好的语意理解）'
    return '未选择模型'
async def handle_photo(update: Update, context) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    room_id = str(chat_id)
    user_info = query_or_def(User(user_id, room_id))# 获取用户唯一 ID
    # 获取图片的最高分辨率版本
    photo_file = await update.message.photo[-1].get_file()

    # 获取文件的原始路径（包括格式）
    file_path = photo_file.file_path
    # 提取文件格式（扩展名），并将其作为保存图片的格式
    file_extension = os.path.splitext(file_path)[1]
    if file_extension not in ['.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG']:
        await update.message.reply_text(f'目前不支持该类型图片')
        return

    # 生成 UUID 作为 `callback_data`
    unique_id = str(uuid.uuid4())

    filename = unique_id + file_extension
    logger.info(f'this room id is {room_id} updalte file is  {filename}')

    file_path = os.path.join(app_path, room_id)
    # 创建房间
    os.makedirs(file_path, exist_ok=True)

    logger.info(f'this room id is {room_id} updalte file_path is  {file_path}')

    file_path = os.path.join(app_path, room_id, filename)

    logger.info(f'this room id is {room_id} updalte fin_file_path is  {file_path}')
    # 下载图片并保存到服务器
    await photo_file.download_to_drive(file_path)
    # 备份原始图像
    backup_path = os.path.join(app_path, room_id, f"backup_{filename}")
    shutil.copy(file_path, backup_path)
    # 调整图像大小
    filename = resize_image(file_path, max_size=4096)
    logger.info(f'now user {user_id} in {user_info.channel}')
    if user_info.channel == 'dress_up':
        logger.info(f'answer it is {filename}')
        await update.message.reply_text(text=f"开始处理图片: {filename}")
        data = {'def_skin': 'inpaint'}
        data['prompt'] = 'nude,person,woman,the front of body,authenticity,natural asses,natural boobs'
        data['reverse_prompt'] = 'finger,wrist,hazy,malformed,warped,hand,multiple belly buttons,multiple breasts,multiple nipples,deformed limbs,disconnected limbs,contorted position,elongated waists,overly exaggerated body parts,body incorrect proportions'
        data['num_inference_steps'] = '30'
        data['guidance_scale'] = '7'
        data['seed'] = '128'
        data['strength'] = '0.8543'
        data['ha_p'] = '0'
        data['ga_b'] = '0.09'
        data['re_p'] = "['key_points','depth','canny']"
        data['re_b'] = "[0.5, 0.2, 0.5, 0.5]"
        data['notify_type'] = 'tel'
        data['filename'] = filename
        data['roomId'] = room_id
        logger.info(f'get req is {data}')
        room_image_manager = RoomImageManager()
        add_task_list(data, 'tel_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)
        return
    elif user_info.channel == 'special_effects' or user_info.channel == 'flux_txt_to_image':
        face_images = req_get_face(file_path)
        if not face_images or len(face_images) == 0:
            logger.info('not find any face')
            await update.message.reply_text(f'{filename} 未识别到人脸')
            return
        if len(face_images) == 1:
            keyboard = [
                [
                    InlineKeyboardButton("输入情景", callback_data=f"t_i_{filename}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            logger.info(f'{filename}要如何处理该图片')
            # 通知用户图片已经保存，并提供选择框
            await update.message.reply_text(f'{filename} 要如何处理该图片', reply_markup=reply_markup)
            return
        # 创建 InlineKeyboard 按钮
        keyboard = [
            [
                InlineKeyboardButton("检测到多人可选择主角", callback_data=f"f_i_{filename}"),
                InlineKeyboardButton("或直接输入情景", callback_data=f"t_i_{filename}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        # 通知用户图片已经保存，并提供选择框
        await update.message.reply_text(f'{filename} 要如何处理该图片', reply_markup=reply_markup)
        return
    elif ('swap_pic' == user_info.channel or 'swap_video' == user_info.channel):
        if user_info.status == 'awaiting_input':
            pre_face_pic_list = user_info.pre_pic_list #user_states[user_id]['pre_face_pic_list']
            pre_face_pic_list.append(filename)
            user_info.pre_pic_list = pre_face_pic_list
            user_states_control(user_info)
            keyboard = [
                [
                    InlineKeyboardButton("继续上传", callback_data=f"continue_face_pic_pre"),
                    InlineKeyboardButton("识别脸部", callback_data=f"start_face_pre")

                ],[
                    InlineKeyboardButton("切换其他模型", callback_data=f"start_command")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                f'请上传处理图片，目前您上传了{len(pre_face_pic_list)}张图片，要开始识别吗',
                reply_markup=reply_markup)
            return
        if user_info.status == 'awaiting_handler_pic':
            org_faces = user_info.org_pic_list
            to_faces = user_info.to_pic_list
            await update.message.reply_text(f'{filename} 开始换脸 {org_faces} {to_faces}')
            data = {'def_skin': 'start_swap_face'}
            data['notify_type'] = 'tel'
            data['filename'] = filename
            data['org_faces'] = org_faces
            data['to_faces'] = to_faces
            data['roomId'] = room_id
            logger.info(f'get req is {data}')
            room_image_manager = RoomImageManager()
            add_task_list(data, 'tel_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)
            return
    await update.message.reply_text(f'您还为选择处理模式请在，🤖AI功能切换 中选择一个吧', reply_markup=get_rest_key())
    return
# 定义常见的视频格式扩展名
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']

hadUser = []
async def handle_wel_user(update: Update, context) -> None:
    # 解析新用户加入时的 referral_id
    message = update.message.text
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id  # 获取用户唯一 ID
    room_id = str(chat_id)
    user_info = query_or_def(User(user_id, room_id))
    if context.args:
        # 提取参数中的 referral_id
        referral_id = context.args[0]
        # 检查 referral_id 是否符合预期格式
        if referral_id.startswith('referral_id_'):
            referrer_id = referral_id.split('referral_id_')[1]
            if referrer_id != user_id and referrer_id not in hadUser:
                # 获取分享人的信息并增加book值
                referrer_info = query_or_def(User(referrer_id))
                user_vip_control(referrer_info, 10)
                hadUser.append(referrer_id)
                # 通知分享人已成功邀请
                await context.bot.send_message(
                    chat_id=referrer_id,
                    text=f"你的好友通过你的链接加入了，book值已增加！当前book值：{referrer_info.vip_count}"
                )
    # 欢迎新用户
    await update.message.reply_text("欢迎使用，您可以在 🤖AI功能切换 中选择要使用的功能 然后按照提示操作即可，当前为试运行阶段，仅用于科研目的，请注意合法合规使用，尊重他人隐私。", reply_markup=get_rest_key())
# 处理文本消息
async def handle_message(update: Update, context) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id  # 获取用户唯一 ID
    room_id = str(chat_id)
    user_info = query_or_def(User(user_id, room_id))  # 获取用户唯一 ID
    logger.info(f'this user id and room id is {room_id}')
    text = update.message.text
    if text=='🤖AI功能切换':
        user_info.status=''
        # 创建实例
        room_image_manager = RoomImageManager()
        room_image_manager.update_user(user_info)
        # 定义带有 emoji 的按钮
        keyboard = [
            # [InlineKeyboardButton("🤖 ChatGPT4", callback_data="chatgpt4")],
            # [
            #     InlineKeyboardButton("📄 文生图", callback_data="txt_to_image"),
            #     InlineKeyboardButton("🖼️ 图生图", callback_data="img_to_img")
            # ],
            # [InlineKeyboardButton("🎨 Midjourney绘画机器人", callback_data="midjourney")],
            [
                InlineKeyboardButton("🥼👕重绘换装", callback_data="dress_up"),
                InlineKeyboardButton("🕺自定义情景", callback_data="special_effects")
            ],
            [
                InlineKeyboardButton("🧑‍👵📷️图片换脸", callback_data="swap_pic"),
                InlineKeyboardButton("🧑‍🧒🎥视频换脸", callback_data="swap_video")
            ],
            [InlineKeyboardButton("👑Flux模型自定义情景（更好的语意理解）", callback_data="flux_txt_to_image")]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        # 发送带有按钮的消息
        await update.message.reply_text(
            '点击下面 👇 的按钮选择你想要使用的功能',
            reply_markup=reply_markup
        )
        return
    if text == '👤我的':
        if room_id in glob_task_positions:
            now_pas = glob_task_positions[room_id]
            await update.message.reply_text(f'你目前排在{now_pas}, 请稍等，处理完成会通知您')
        if room_id in video_task_positions:
            now_pas = video_task_positions[room_id]
            await update.message.reply_text(f'视频处理队列你目前排在{now_pas}, 请稍等，处理完成会通知您')
        await update.message.reply_text(f'你当前正在使用 {trans_channel(user_info.channel)} ，所在状态为 {trans_status(user_info.status)}, '
                                        f'剩余book值{user_info.vip_count}', reply_markup=get_rest_key())
        return
    if text == '🗄查看历史':
        if room_id in glob_task_positions:
            now_pas = glob_task_positions[room_id]
            await update.message.reply_text(f'你目前排在{now_pas}, 请稍等，处理完成会通知您')
        if room_id in video_task_positions:
            now_pas = video_task_positions[room_id]
            await update.message.reply_text(f'视频处理队列你目前排在{now_pas}, 请稍等，处理完成会通知您')
        room_image_manager = RoomImageManager()
        pre_img_list = room_image_manager.get_pre_imgStrList(room_id, 'done')
        if len(pre_img_list) <= 0:
            return
        for imgStr, img_type, name, created_at in pre_img_list:
            done_pic_name = imgStr
            logger.info(f"Image {done_pic_name}")
            # 获取文件的扩展名
            _, file_extension = os.path.splitext(done_pic_name)
            done_pic_name_path = os.path.join(app_path, room_id, done_pic_name)
            # 判断是否是视频格式
            if file_extension.lower() in video_extensions:
                logger.info(f"{done_pic_name} is a video file.")
                await send_video(room_id, done_pic_name_path, done_pic_name)
            else:
                await send_photo(room_id, done_pic_name_path, done_pic_name)
        return
    if text == '分享增加book值':
        # 创建分享按钮
        share_link = f"https://t.me/book_yes_bot?start=referral_id_{user_id}"
        share_button = InlineKeyboardButton(text=f"分享给好友", url=share_link)
        # 创建转发按钮
        forward_button = InlineKeyboardButton(text="转发", switch_inline_query=share_link)
        # 创建键盘布局，带有分享、复制和转发按钮
        reply_markup = InlineKeyboardMarkup([[share_button], [forward_button]])
        # 发送消息并附带分享按钮
        await update.message.reply_text(
            text=f"邀请你的朋友来使用bookyes吧，使用你的链接邀请还可以获得book值哦! 或者直接复制 打开 {share_link} ",
            reply_markup=reply_markup
        )

        # 发送book值信息
        await update.message.reply_text(
            f'您的剩余book值：{user_info.vip_count}，当前暂无购买功能',
            reply_markup=get_rest_key()
        )
        return
    if text == '增加book值':
        await update.message.reply_text(
            f'您的剩余book值{user_info.vip_count}，暂无购买增加book值功能', reply_markup=get_rest_key())
        return
    if text == '❓帮助':
        await update.message.reply_text(
            f'本机器人为AI图片处理机器人，您可以在 🤖AI功能切换 中选择要使用的功能 然后按照提示操作即可，如遇到任何问题可重新选择🤖AI功能切换来重置设置，继续体验，部分时刻如因网络问题无法获取处理结果，可在 🗄查看历史 中重复获取查看最近20张处理图片', reply_markup=get_rest_key())
        return
    if text == '📢公告':
        await update.message.reply_text(
            f'本机器人为AI图片处理机器人，您可以在 🤖AI功能切换 中选择要使用的功能 然后按照提示操作即可，当前为试运行阶段，仅用于科研目的，请注意合法合规使用，尊重他人隐私。',
            reply_markup=get_rest_key())
        return
    if text == '技术交流':
        await update.message.reply_contact(
            '573244218219','Li','Dream',
            reply_markup=get_rest_key())
    if 'special_effects' == user_info.channel or 'flux_txt_to_image' == user_info.channel:
        # 检查用户状态
        if user_info.status == 'awaiting_input':
            filename = user_info.file_name
            # 处理用户输入的文本
            await update.message.reply_text(f"你输入的情景是: {text} 开始生成 {filename}")
            logger.info(f'answer it is {room_id}')
            data = {'def_skin': '909', 'prompt': text,
                    'reverse_prompt': ''}
            data['notify_type'] = 'tel'
            data['filename'] = filename
            data['face_filename'] = ''
            data['roomId'] = room_id
            if 'flux_txt_to_image' == user_info.channel:
                data['gen_type'] = 'flux'
            else:
                data['gen_type'] = ''
            logger.info(f'get req is {data}')
            # 创建实例
            room_image_manager = RoomImageManager()
            add_task_list(data, 'tel_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)
            user_info.status=''
            # 完成处理后，清除用户的状态
            user_states_control(user_info)
            return
        if user_info.status == 'only_face_awaiting_input':
            filename = user_info.file_name
            # 处理用户输入的文本
            await update.message.reply_text(f"你选择了主角，输入的情景是: {text} 开始生成 {filename}")
            logger.info(f'answer it is {room_id}')
            data = {'def_skin': '909', 'prompt': text,
                    'reverse_prompt': ''}
            data['notify_type'] = 'tel'
            data['filename'] = filename
            data['face_filename'] = filename
            data['roomId'] = room_id
            if 'flux_txt_to_image' == user_info.channel:
                data['gen_type'] = 'flux'
            logger.info(f'get req is {data}')
            # 创建实例
            room_image_manager = RoomImageManager()
            add_task_list(data, 'tel_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)
            # 完成处理后，清除用户的状态
            user_info.status=''
            user_states_control(user_info)
            return
    # 换脸
    if 'swap_pic' == user_info.channel or 'swap_video' == user_info.channel:
        if user_info.status == 'awaiting_input':
            #等待输入
            keyboard = [
                [
                    InlineKeyboardButton("继续上传", callback_data=f"continue_face_pic_pre"),
                    InlineKeyboardButton("识别脸部", callback_data=f"start_face_pre")
                ],[
                    InlineKeyboardButton("切换其他模型", callback_data=f"start_command")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            pre_face_pic_list = user_info.pre_pic_list
            await update.message.reply_text(f'请上传处理图片，目前您上传了{len(pre_face_pic_list)}张图片，要开始识别吗', reply_markup=reply_markup)
            return

    if room_id in glob_task_positions:
        now_pas = glob_task_positions[room_id]
        await update.message.reply_text(f'你目前排在{now_pas}, 请稍等，处理完成会通知您')
    if room_id in video_task_positions:
        now_pas = video_task_positions[room_id]
        await update.message.reply_text(f'视频处理队列你目前排在{now_pas}, 请稍等，处理完成会通知您')

    else:
        keyboard = [
            [InlineKeyboardButton("开启bookyes", callback_data="start_command")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        # 发送带有按钮的消息
        await update.message.reply_text(
            '点击下面 👇 的按钮启动bookyes',
            reply_markup=reply_markup
        )

def generate_reply_markup(image_paths):
    """
    根据给定的图片路径数组，生成配对选择按钮。
    :param image_paths: 图片文件路径数组
    :return: reply_markup 对象
    """
    keyboard = []
    for idx in range(len(image_paths)):
        keyboard.append([InlineKeyboardButton(f"选择图片{idx + 1}", callback_data=f'select_image_{idx + 1}')])
    return InlineKeyboardMarkup(keyboard)
def generate_media_group(image_paths):
    """
    根据给定的图片路径数组，生成 Telegram 的 media_group。
    :param image_paths: 图片文件路径数组
    :return: media_group 列表
    """
    media_group = []
    for idx, img_path in enumerate(image_paths):
        media_group.append(InputMediaPhoto(open(img_path, 'rb'), caption=f"图片{idx + 1}"))
    return media_group

async def send_images_with_options(update, context, image_paths):
    chat_id = update.effective_chat.id

    # 生成 media_group 和 reply_markup
    media_group = generate_media_group(image_paths)
    reply_markup = generate_reply_markup(image_paths)

    # 发送图片组
    await context.bot.send_media_group(chat_id=chat_id, media=media_group)

    # 发送带有选择按钮的消息
    await context.bot.send_message(chat_id=chat_id, text="请选择要配对的图片：", reply_markup=reply_markup)

async def history_pre_face(room_id):
    room_image_manager = RoomImageManager()
    pre_img_list = room_image_manager.get_pre_imgStrList(room_id, 'face_pre')
    if len(pre_img_list) <= 0:
        return
    to_keyboard_face = []
    media_group = []
    for imgStr, img_type, name, created_at in pre_img_list:
        file_face_name = imgStr
        to_keyboard_face.append([
            InlineKeyboardButton(f"选择 {file_face_name}", callback_data=f"mu_chose_{file_face_name}")
        ])
        logger.info(f"Image {file_face_name}")
        file_face_name_path = os.path.join(app_path, room_id, file_face_name)
        media_group.append(InputMediaPhoto(open(file_face_name_path, 'rb'), caption=f"图片{file_face_name}"))
    to_keyboard_face.append([
        InlineKeyboardButton(f"重新选择", callback_data=f"reset_pre_face")
    ])
    reply_markup_face = InlineKeyboardMarkup(to_keyboard_face)
    notify_it('tel', 'processing_done',
              {'img_type': 'media_group',
               'text': '以下为您的历史记录可直接选择，比如先选A在选B则为将A替换为B，你可以先后选择多组'}, to=room_id,
              keyList=reply_markup_face, media_group=media_group)

# 处理按钮点击的回调
async def button_callback(update: Update, context) -> None:
    query = update.callback_query
    chat_id = update.effective_chat.id  # 获取用户唯一 ID
    room_id = str(chat_id)
    user_id = query.from_user.id
    user_info = query_or_def(User(user_id, room_id))  # 获取用户唯一 ID
    await query.answer()
    # 处理 "是" 的点击事件
    # 处理 /start 命令的回调
    if query.data == "start_command":
        # 这里可以直接调用 /start 命令对应的处理逻辑
        await start(query.message, context)
    elif query.data == 'dress_up':
        user_info.channel='dress_up'
        user_states_control(user_info)
        await query.edit_message_text(text="上传图片后等待结果")
    elif query.data == 'special_effects':
        user_info.channel = 'special_effects'
        user_states_control(user_info)
        await query.edit_message_text(text="请先上传需要替换的人物照片，我会为你提取面部")
    elif query.data == 'flux_txt_to_image':
        user_info.channel = 'flux_txt_to_image'
        user_states_control(user_info)
        await query.edit_message_text(text="请先上传需要替换的人物照片，我会为你提取面部")
    elif query.data == 'swap_pic':
        user_info.channel = 'swap_pic'
        user_info.status = 'awaiting_input'
        user_info.pre_pic_list = []
        user_info.org_pic_list = []
        user_info.to_pic_list = []
        user_states_control(user_info)
        await history_pre_face(room_id)
        await query.edit_message_text(text="可上传需要替换的人物照片，我会为你提取面部，用于换脸处理")
    elif query.data == 'swap_video':
        await history_pre_face(room_id)
        user_info.channel = 'swap_video'
        user_info.status = 'awaiting_input'
        user_info.pre_pic_list = []
        user_info.org_pic_list = []
        user_info.to_pic_list = []
        user_states_control(user_info)
        await query.edit_message_text(text="可上传需要替换的人物照片，我会为你提取面部，用于换脸处理")
    elif query.data == 'continue_face_pic_pre':
        await query.edit_message_text(text="请继续上传图片")
    elif query.data == 'start_face_pre':
        pre_face_pic_list = user_info.pre_pic_list
        data = {'def_skin': 'swap_face'}
        data['notify_type'] = 'tel'
        data['pre_face_pic_list'] = pre_face_pic_list
        data['roomId'] = room_id
        logger.info(f'get req is {data}')
        room_image_manager = RoomImageManager()
        add_task_list(data, 'tel_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)
        # 将用户的状态设置为 'awaiting_input'，等待用户输入
        await query.edit_message_text(text=f"检测{len(pre_face_pic_list)}张图片中的人脸，稍等：")
        user_info.status=''
        user_states_control(user_info)
        return
    elif query.data == 'reset_pre_face':
        user_info.org_pic_list = []
        user_info.to_pic_list = []
        user_states_control(user_info)
        await query.message.reply_text(f"已清除配对关系")
        await history_pre_face(room_id)
    elif query.data == 'finish_pre_swap_face':
        user_info.pre_pic_list= []
        f_to_faces = user_info.to_pic_list
        user_info.status='awaiting_handler_pic'
        user_states_control(user_info)
        # 将用户的状态设置为 'awaiting_input'，等待用户输入
        if user_info.channel == 'swap_video':
            chan_name = 'mp4 avi mov 格式视频吧'
        else:
            chan_name = '图片'
        await query.message.reply_text(f"你已完成设置，我会将你选择的{len(f_to_faces)}对人脸进行替换，上传{chan_name}吧")
        return
    elif query.data.startswith('mu_chose_'):
        filename = query.data.replace("mu_chose_", "")
        if len(user_info.org_pic_list) == len(user_info.to_pic_list):
            user_info.org_pic_list.append(filename)
            user_states_control(user_info)
            await query.message.reply_text(f'已经选择原图{filename},选择下替换他的图吧')
        else:
            # 等待输入
            keyboard = [
                [
                    InlineKeyboardButton("我已完成选择", callback_data=f"finish_pre_swap_face"),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            user_info.to_pic_list.append(filename)
            user_states_control(user_info)
            await query.message.reply_text(f'选择下替换他的图{filename}，可以继续在上方选下一对或点击下边按钮后开始上传', reply_markup=reply_markup)
        return
    elif query.data.startswith("t_i_"):
        filename = query.data.replace("t_i_", "")
        # 将用户的状态设置为 'awaiting_input'，等待用户输入
        user_info.status='awaiting_input'
        user_info.file_name=filename
        user_states_control(user_info)
        await query.edit_message_text(text="请输入情景：")
    elif query.data.startswith("o_f_i_"):
        filename = query.data.replace("o_f_i_", "")
        user_info.status = 'only_face_awaiting_input'
        user_info.file_name = filename
        user_states_control(user_info)
        await query.edit_message_caption(caption="请输入情景：")
    elif query.data.startswith("f_i_"):
        filename = query.data.replace("f_i_", "")
        data = {'def_skin': 'face'}
        data['notify_type'] = 'tel'
        data['filename'] = filename
        data['roomId'] = room_id
        logger.info(f'get req is {data}')
        room_image_manager = RoomImageManager()
        add_task_list(data, 'tel_notify_it', glob_task_queue, glob_task_positions, notify_fuc=notify_it, room_image_manager=room_image_manager, user_info=user_info)
        # 将用户的状态设置为 'awaiting_input'，等待用户输入
        await query.edit_message_text(text="检测主角中稍等：")
    # 处理 "否" 的点击事件
    elif query.data == "cancel":
        await query.edit_message_text(text="已取消图片处理。")

# 启动 Telegram 机器人
def run_telegram_bot():
    # 使用你的 Bot API Token
    bot_token = '7467241687:AAHlU2z43Ks9f-jy8EX78AwOHPrVoO5B0kg'
    # 创建 Application 对象
    application = Application.builder().token(bot_token).connect_timeout(10).read_timeout(600).write_timeout(600).build()
    # 添加处理 /start 命令的处理器
    application.add_handler(CommandHandler("start", handle_wel_user))
    # 添加处理图片的处理器
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    # Add handler for handling videos
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    # 添加按钮回调处理器
    application.add_handler(CallbackQueryHandler(button_callback))
    # 添加文本消息处理器
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info(f'start listing {bot_token}')
    # 启动 Long Polling
    application.run_polling()