import os, queue, threading, json
from book_yes_logger_config import logger
from PIL import ImageDraw,Image, ImageFilter
from telegram import ReplyKeyboardMarkup
import random, requests, hashlib
# 创建队列，最大容量为30
glob_task_queue = queue.Queue(maxsize=500)

# 创建队列，最大容量为30
task_vide_queue = queue.Queue(maxsize=500)

glob_task_positions = {}  # 用来记录 room_id 和队列中的位置
video_task_positions = {}  # 用来记录 room_id 和队列中的位置

# 创建一个线程锁
lock = threading.Lock()

class User:
    def __init__(self, user_id, room_id='', channel='', status=None, pre_pic_list=None, org_pic_list=None, to_pic_list=None, vip_count=0, file_name=''):
        self.user_id = user_id
        self.room_id = room_id
        self.channel = channel
        self.status = status
        self.pre_pic_list = pre_pic_list or []  # 默认值为空列表
        self.org_pic_list = org_pic_list or []  # 默认值为空列表
        self.to_pic_list = to_pic_list or []  # 默认值为空列表
        self.vip_count = vip_count
        self.file_name = file_name

    def __repr__(self):
        return f"User(user_id={self.user_id}, room_id={self.room_id}, channel={self.channel}, status={self.status}, vip_count={self.vip_count}), file_name={self.file_name})"

class GenContextParam:
    def __init__(self, prompt, prompt_2, reverse_prompt, reverse_prompt_2,
                        next_genImage, big_mask, num_inference_steps, guidance_scale,
                        func_call_back, strength, control_img_list,
                        control_float_array, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
                        negative_pooled_prompt_embeds, original_width, original_height, processed_filename, room_id):
        self.prompt = prompt
        self.prompt_2 = prompt_2
        self.reverse_prompt = reverse_prompt
        self.reverse_prompt_2 = reverse_prompt_2
        self.next_genImage = next_genImage
        self.big_mask = big_mask
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.func_call_back = func_call_back
        self.strength = strength
        self.control_img_list = control_img_list
        self.control_float_array = control_float_array
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        self.original_width = original_width
        self.original_height = original_height
        self.save_filename = processed_filename
        self.room_id = room_id
        self.book_img_name = 'book'

class GenSavePathParam:
    def __init__(self, base_path, uq_id, filename):
        self.filled_image_pil_filename = 'filled_image_pil_' + filename
        self.filled_image_pil_path = os.path.join(base_path, uq_id, self.filled_image_pil_filename)

        self.next_filled_image_pil_filename = 'filled_image_pil_next_' + filename
        self.next_filled_image_pil_path = os.path.join(base_path, uq_id, self.next_filled_image_pil_filename)

        self.cur_res_filename = 'clear_return_' + filename
        self.cur_res_path = os.path.join(base_path, uq_id, self.cur_res_filename)

        self.mask_clear_finally_filename = 'filled_use_mask_image_pil_' + filename
        self.mask_clear_finally_path = os.path.join(base_path, uq_id, self.mask_clear_finally_filename)

        self.segmentation_image_pil_filename = 'seg_use_image_pil_' + filename
        self.segmentation_image_pil_path = os.path.join(base_path, uq_id,
                                                   self.segmentation_image_pil_filename)

        self.control_net_fname = f'control_net_' + filename
        self.control_net_fname_path = os.path.join(base_path, uq_id, self.control_net_fname)

        self.nor_control_net_fname = f'nor_control_net_' + filename
        self.nor_control_net_fname_path = os.path.join(base_path, uq_id, self.nor_control_net_fname)

        self.nor_3d_control_net_fname = f'nor_3d_control_net_' + filename
        self.nor_3d_control_net_fname_path = os.path.join(base_path, uq_id, self.nor_3d_control_net_fname)

        self.with_torso_mask_name = f'with_torso_mask_' + filename
        self.with_torso_mask_name_path = os.path.join(base_path, uq_id, self.with_torso_mask_name)

        self.line_file_name = f"line_{filename}"
        self.line_file_name_path = os.path.join(base_path, uq_id, self.line_file_name)

        self.line_file_auto_name = f"line_auto_{filename}"
        logger.info(f'save find and get {self.line_file_auto_name}')
        self.line_file_auto_name_path = os.path.join(base_path, uq_id, self.line_file_auto_name)

        self.fill_all_mask_img_fname = f'fill_all_skin_' + filename
        self.fill_all_mask_img_name_path = os.path.join(base_path, uq_id, self.fill_all_mask_img_fname)

def translate_baidu(text, from_lang='zh', to_lang='en'):
    appid = '20241003002166185'  # 你的百度翻译 AppID
    secretKey = 'JO20L2CkL8pgIeAQQ8ob'  # 你的百度翻译密钥
    salt = str(random.randint(32768, 65536))  # 随机数
    sign = appid + text + salt + secretKey  # 生成签名
    sign = hashlib.md5(sign.encode()).hexdigest()

    url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    params = {
        'q': text,
        'from': from_lang,
        'to': to_lang,
        'appid': appid,
        'salt': salt,
        'sign': sign
    }

    response = requests.get(url, params=params)
    result = response.json()

    if 'trans_result' in result:
        return result['trans_result'][0]['dst']
    else:
        return result
def extract_data(data, fields):
    """
    通用数据提取方法，从 data 中提取指定的字段并设置默认值。
    fields 是一个字典，键为字段名，值为默认值。
    """
    return {field: data.get(field, default) for field, default in fields.items()}
def process_req_data(data):
    """
    处理图片相关的数据，使用通用方法提取字段，并对特定字段进行额外处理。
    """
    # 所有需要提取的字段及其默认值
    fields = {
        'def_skin': '',
        'filename': '',
        'face_filename':'',
        'prompt': '',
        'prompt_2': '',
        're_mask': '',
        're_p': '[]',  # 默认字符串形式，后续使用 eval
        're_b': '[]',  # 默认字符串形式，后续使用 eval
        'ha_p': 0,  # 整数
        'ga_b': 0.0,  # 浮点数
        'reverse_prompt': '',
        'reverse_prompt_2': '',
        'strength': 1.0,  # 浮点数
        'num_inference_steps': 50,  # 整数
        'guidance_scale': 7.5,  # 浮点数
        'seed': 126,  # 默认值为 None
        'roomId': '',  # 房间ID
        'notify_type': '',  # 通知了类型，网页请求为ws，机器人为tel
        'pre_face_pic_list': [],
        'org_faces':[],
        'to_faces': [],
        'gen_type':''
    }

    # 提取数据
    processed_data = extract_data(data, fields)

    # 特殊字段处理
    try:
        processed_data['re_p_float_array'] = eval(processed_data['re_p'])  # 将 re_p 转换为浮点数组
    except Exception:
        processed_data['re_p_float_array'] = []  # 处理异常，确保返回空列表

    try:
        processed_data['re_b_float_array'] = eval(processed_data['re_b'])  # 将 re_b 转换为浮点数组
    except Exception:
        processed_data['re_b_float_array'] = []  # 处理异常，确保返回空列表

    # 强制转换类型，确保数据格式正确
    processed_data['ha_p'] = int(processed_data.get('ha_p', 0))  # 确保 ha_p 为整数
    processed_data['ga_b'] = float(processed_data.get('ga_b', 0.0))  # 确保 ga_b 为浮点数
    processed_data['strength'] = float(processed_data.get('strength', 1.0))  # 确保 strength 为浮点数
    processed_data['num_inference_steps'] = int(processed_data.get('num_inference_steps', 50))  # 确保为整数
    processed_data['guidance_scale'] = float(processed_data.get('guidance_scale', 7.5))  # 浮点数
    processed_data['seed'] = int(processed_data.get('seed', 126))
    if processed_data['notify_type'] == '':
        processed_data['notify_type'] = 'ws'
    logger.info(f"req map : {processed_data}")
    # 返回完整处理的数据
    return processed_data

def get_rest_key():
    final_keyboard = [
        ['🤖AI功能切换'],
        ['增加book值', '❓帮助', '📢公告'],
        ['👤我的', '🗄查看历史'],
        ['技术交流', '分享增加book值']
    ]

    # 创建自定义键盘
    final_reply_markup = ReplyKeyboardMarkup(final_keyboard, one_time_keyboard=True, resize_keyboard=True)
    return final_reply_markup

def resize_image(image_path, max_size=4096):
    # 打开图片
    image = Image.open(image_path)
    width, height = image.size

    # 获取文件的扩展名，并进行大小写敏感检查
    file_root, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()  # 转换为小写

    # 如果扩展名是 .jpg 或 .jpeg，将其转换为小写 .jpg
    if file_extension in ['.jpg', '.jpeg']:
        new_image_path = file_root + ".jpg"
    # 如果扩展名是 .png，将其转换为小写 .png
    elif file_extension == '.png':
        new_image_path = file_root + ".png"
    else:
        # 如果不是 jpg 或 png，转换为 png
        new_image_path = file_root + ".png"
        image = image.convert("RGB")  # 确保图片转为RGB模式以保存为 png

    # 如果图片尺寸大于 max_size，则调整尺寸
    if width > max_size or height > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)

    # 保存修改后的图片
    image.save(new_image_path)
    logger.info(f"Image saved at {new_image_path}")
    return os.path.basename(new_image_path)

def add_task_list(data, notify_fuc_name, task_queue, task_positions, queue_name='glob_task_queue', notify_fuc=None, room_image_manager=None, user_info = None):
    room_id = data['roomId']
    notify_type = data['notify_type']
    data['notify_fuc'] = notify_fuc_name
    if user_info is not None:
        data['user_id'] = user_info.user_id
        logger.info(f'user info is {user_info} cate zhe vip_count {user_info.vip_count}')
        if user_info.vip_count <= -11:
            notify_fuc(notify_type, 'processing_step_progress',
                       {'text': '您的积分不足，不能执行该操作了，觉得好用可以分享下赚积分'}, to=room_id, keyList=get_rest_key())
            return
    with lock:
        try:
            if room_id in task_positions:
                logger.info(f'{room_id} had no finish pic')
                now_pas = task_positions[room_id]
                notify_fuc(notify_type, 'processing_step_progress',
                          {'text': f'book yes 你还有任务在执行哈，当前排在{now_pas}位，稍等...'}, to=room_id)
            else:
                # 插入任务到数据库
                task_id = room_image_manager.insert_task(queue_name, json.dumps(data))
                data['task_id'] = task_id  # 将 task_id 添加到任务数据中
                logger.info(f'add task task_id {task_id} to {queue_name} info is {data}')
                # 尝试将任务放入队列，如果队列已满，抛出异常
                task_queue.put(data, block=False)
                position_in_queue = task_queue.qsize()
                task_positions[room_id] = position_in_queue
                if position_in_queue >= 2:
                    logger.info(f'{room_id} had no finish pic in nub 2')
                    notify_fuc(notify_type, 'processing_step_progress', {
                        'text': f'book yes 已经进入执行队列，当前排在{position_in_queue}位，已经记录您的提交，等不及稍后处理完会提示您，也可晚点点击查看历史...'},
                              to=room_id)
                else:
                    logger.info(f'{room_id} had no finish pic in last')
                    notify_fuc(notify_type, 'processing_step_progress',
                              {'text': f'book yes 已经进入执行队列，当前排在{position_in_queue}位，这就开始执行...'},
                              to=room_id)
        except queue.Full:
            if room_id in task_positions:
                logger.info(f'失败清理排队位置了')
                del task_positions[room_id]
            # 队列已满，返回错误
            notify_fuc(notify_type, 'processing_step_progress',
                      {'text': 'book yes 太火爆已满，稍后在线人数较少时重新提交可好...'}, to=room_id)

def load_unfinished_tasks(queue_name, task_queue, room_image_manager):
    # 从数据库中获取未完成的任务
    tasks = room_image_manager.get_unfinished_tasks(queue_name)
    # 按顺序将任务插入队列
    for task_data in tasks:
        logger.info(f'add task {task_data} from db')
        # 将任务数据反序列化为 Python 对象，并附加 id
        task = json.loads(task_data['task_data'])
        task['task_id'] = task_data['id']  # 将数据库中的 id 添加到任务数据中
        task_queue.put(task)  # 插入到任务队列
    logger.info(f"{len(tasks)} tasks loaded from the database into the queue.")

def read_json_file():
    # 读取json文件
    with open('/nvme0n1-disk/book_yes/static/data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_kami_file():
    # 读取json文件
    with open('/nvme0n1-disk/book_yes/static/kami.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data