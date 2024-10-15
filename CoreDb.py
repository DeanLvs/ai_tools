import sqlite3, json
from datetime import datetime
from book_yes_logger_config import logger
from BookYesCommon import User
import threading
class RoomImageManager:
    _instance_lock = threading.Lock()  # 线程安全的锁
    _instance = None  # 用于存放单例实例

    def __new__(cls, *args, **kwargs):
        # 实现单例模式，确保只有一个实例
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super(RoomImageManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_name='gen_room_t_images_ks.db'):
        # 仅当实例首次创建时初始化数据库
        if not hasattr(self, "_initialized"):
            self.db_name = db_name
            self._initialize_db()
            self._initialized = True  # 确保 `__init__` 只执行一次

    def _initialize_db(self):
        # 初始化数据库，创建表
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS room_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id TEXT NOT NULL,
                imgStr TEXT NOT NULL,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                ext_info TEXT,
                is_delete TEXT DEFAULT '0',
                created_at TEXT NOT NULL
            )
            ''')

            cursor.execute('''
                        CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT NOT NULL,
                            room_id TEXT NOT NULL,
                            channel TEXT NOT NULL,
                            status TEXT,
                            pre_pic_list TEXT,
                            org_pic_list TEXT,
                            to_pic_list TEXT,
                            file_name TEXT,
                            vip_count INTEGER
                        );
                        ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS glob_task_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_data TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_vide_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_data TEXT NOT NULL
                )
            ''')

            conn.commit()
        except sqlite3.Error as e:
            logger.info(f"SQLite error during table creation: {e}")
        finally:
            conn.close()

    def insert_task(self, queue_name, task_data):
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f'INSERT INTO {queue_name} (task_data) VALUES (?)', (task_data,))
            conn.commit()
            task_id = cursor.lastrowid  # 获取数据库生成的自增 ID
            return task_id
        except sqlite3.Error as e:
            logger.info(f"SQLite error while inserting task into {queue_name}: {e}")
            return None
        finally:
            conn.close()

    def clear_tables(self):
        conn = self._create_connection()
        try:
            cursor = conn.cursor()

            # 清空 glob_task_queue 表
            cursor.execute('DELETE FROM glob_task_queue')

            # 清空 task_vide_queue 表
            cursor.execute('DELETE FROM task_vide_queue')

            conn.commit()  # 提交更改
            print("Tables have been cleared.")
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        finally:
            conn.close()

    def remove_task_by_id(self, queue_name, task_id):
        logger.info(f'remove task {queue_name} {task_id}')
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f'DELETE FROM {queue_name} WHERE id = ?', (task_id,))
            conn.commit()
        except sqlite3.Error as e:
            logger.info(f"SQLite error while removing task from {queue_name} with task_id {task_id}: {e}")
        finally:
            conn.close()

    def get_unfinished_tasks(self, queue_name):
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            # 按 id 顺序获取未完成的任务
            cursor.execute(f'SELECT id, task_data FROM {queue_name} ORDER BY id ASC')
            tasks = cursor.fetchall()

            # 返回包含 id 和 task_data 的字典列表
            return [{'id': task[0], 'task_data': task[1]} for task in tasks]
        except sqlite3.Error as e:
            logger.info(f"SQLite error while fetching tasks from {queue_name}: {e}")
            return []
        finally:
            conn.close()

    def get_tasks(self, queue_name):
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f'SELECT task_data FROM {queue_name} ORDER BY id ASC')
            tasks = [row[0] for row in cursor.fetchall()]
            return tasks
        except sqlite3.Error as e:
            logger.info(f"SQLite error while fetching tasks from {queue_name}: {e}")
            return []
        finally:
            conn.close()

    def remove_task(self, queue_name):
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f'DELETE FROM {queue_name} WHERE id = (SELECT id FROM {queue_name} ORDER BY id ASC LIMIT 1)')
            conn.commit()
        except sqlite3.Error as e:
            logger.info(f"SQLite error while removing task from {queue_name}: {e}")
        finally:
            conn.close()

    def restore_queue(self, queue, queue_name):
        tasks = self.get_tasks(queue_name)
        for task in tasks:
            queue.put(task)

    def _create_connection(self):
        # 创建数据库连接
        try:
            conn = sqlite3.connect(self.db_name)
            return conn
        except sqlite3.Error as e:
            logger.info(f"SQLite connection error: {e}")
            return None

    def insert_user(self, user):
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            # 序列化列表到 JSON 字符串
            pre_pic_list_json = json.dumps(user.pre_pic_list)
            org_pic_list_json = json.dumps(user.org_pic_list)
            to_pic_list_json = json.dumps(user.to_pic_list)
            # 执行插入操作
            cursor.execute('''
                INSERT INTO users (user_id, room_id, channel, status, pre_pic_list, org_pic_list, to_pic_list, vip_count, file_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user.user_id, user.room_id, user.channel, user.status, pre_pic_list_json, org_pic_list_json,
                  to_pic_list_json, user.vip_count, user.file_name))

            conn.commit()
            logger.info(f"New user inserted: {user}")
        except sqlite3.Error as e:
            logger.info(f"SQLite error while inserting user: {e}")
        finally:
            conn.close()

    def get_user(self, user_id):
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            # 查询操作
            cursor.execute('''
                SELECT user_id, room_id, channel, status, pre_pic_list, org_pic_list, to_pic_list, vip_count, file_name
                FROM users
                WHERE user_id = ?
            ''', (user_id,))

            row = cursor.fetchone()
            if row:
                # 反序列化 JSON 字符串为列表
                pre_pic_list = json.loads(row[4]) if row[4] else []
                org_pic_list = json.loads(row[5]) if row[5] else []
                to_pic_list = json.loads(row[6]) if row[6] else []

                # 创建 User 对象
                user = User(row[0], row[1], row[2], row[3], pre_pic_list, org_pic_list, to_pic_list, row[7], row[8])
                logger.info(f"User found: {user}")
                return user
            else:
                logger.info(f"No user found with user_id: {user_id}")
                return None
        except sqlite3.Error as e:
            logger.info(f"SQLite error while fetching user: {e}")
            return None
        finally:
            conn.close()

    def update_user(self, user):
        conn = self._create_connection()
        try:
            cursor = conn.cursor()
            # 序列化列表到 JSON 字符串
            pre_pic_list_json = json.dumps(user.pre_pic_list)
            org_pic_list_json = json.dumps(user.org_pic_list)
            to_pic_list_json = json.dumps(user.to_pic_list)

            # 更新操作
            cursor.execute('''
                UPDATE users
                SET room_id = ?, channel = ?, status = ?, pre_pic_list = ?, org_pic_list = ?, to_pic_list = ?, vip_count = ?, file_name = ?
                WHERE user_id = ?
            ''', (user.room_id, user.channel, user.status, pre_pic_list_json, org_pic_list_json, to_pic_list_json,
                  user.vip_count, user.file_name, user.user_id))

            conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"User {user.user_id} updated successfully.")
            else:
                logger.info(f"No user found with user_id: {user.user_id}")
        except sqlite3.Error as e:
            logger.info(f"SQLite error while updating user: {e}")
        finally:
            conn.close()


    def reset_is_delete_for_room_with_limit(self, room_id):
        # 连接到数据库
        conn = self._create_connection()
        try:
            # 创建游标对象
            cursor = conn.cursor()

            # 查询 room_id 下 is_delete 为 '0' 的记录数量
            cursor.execute('''
                SELECT COUNT(*)
                FROM room_images
                WHERE room_id = ? AND is_delete = '0'
            ''', (room_id,))
            count_zeros = cursor.fetchone()[0]
            logger.info(f'get db rec is {count_zeros}')
            # 如果记录数大于 10，则更新多余的记录
            if count_zeros > 10:
                # 查询不等于 '0' 的记录，按 created_at 升序排列（最早的记录优先）
                cursor.execute('''
                    SELECT id
                    FROM room_images
                    WHERE room_id = ? AND is_delete != '0'
                    ORDER BY created_at ASC
                    LIMIT ?
                ''', (room_id, count_zeros - 10))
                # 获取需要更新的记录 ID
                rows_to_update = cursor.fetchall()
                logger.info(f'rows_to_update - {rows_to_update}')
                if rows_to_update:
                    # 更新这些记录的 is_delete 为 '1'
                    ids_to_update = [row[0] for row in rows_to_update]
                    cursor.executemany('''
                        UPDATE room_images
                        SET is_delete = '1'
                        WHERE id = ?
                    ''', [(record_id,) for record_id in ids_to_update])

                    # 提交事务
                    conn.commit()

                    # 输出受影响的行数
                    logger.info(f"{cursor.rowcount} rows were updated.")
                else:
                    logger.info("No records to update.")
            else:
                logger.info("The number of records with is_delete='0' is less than or equal to 10.")

        except sqlite3.Error as e:
            logger.info(f"SQLite error: {e}")

        finally:
            # 关闭数据库连接
            conn.close()

    # 函数：获取指定 room_id 的所有 imgStr 记录
    def get_imgStrList(self, room_id):
        try:
            conn = self._create_connection()  # 创建新的数据库连接
            t_cursor = conn.cursor()

            # 按日期排序，查询最近的 8 条记录
            t_cursor.execute(
                '''
                    SELECT imgStr, type, name, created_at FROM room_images WHERE is_delete = '0' AND room_id = ? ORDER BY ID DESC LIMIT 20 
                ''',
                (room_id,))
            # 获取所有结果
            rows = t_cursor.fetchall()

            # 检查是否获取到数据
            if rows:
                return rows
            else:
                logger.info(f"No data found for room_id: {room_id}")
                return []

        except sqlite3.Error as e:
            # 打印数据库相关错误信息
            logger.info(f"SQLite error: {e}")
            return []
        finally:
            # 确保连接在最后关闭
            conn.close()

    def get_pre_imgStrList(self, room_id, pic_type):
        try:
            conn = self._create_connection()  # 创建新的数据库连接
            t_cursor = conn.cursor()

            # 按日期排序，查询最近的 8 条记录
            t_cursor.execute(
                '''
                    SELECT imgStr, type, name, created_at FROM room_images WHERE is_delete = '0' AND room_id = ? AND type = ? ORDER BY ID DESC LIMIT 20 
                ''',
                (room_id,pic_type,))
            # 获取所有结果
            rows = t_cursor.fetchall()
            # 检查是否获取到数据
            if rows:
                return rows
            else:
                logger.info(f"No data found for room_id: {room_id}")
                return []

        except sqlite3.Error as e:
            # 打印数据库相关错误信息
            logger.info(f"SQLite error: {e}")
            return []
        finally:
            # 确保连接在最后关闭
            conn.close()

    def save_all(self, img_to_save, img_to_save_path):
        img_to_save.save(img_to_save_path)

    # 函数：为指定 room_id 插入新的一条记录
    def insert_imgStr(self, r_id=None, img_name=None, img_type=None, name=None, keephide = None , file_i = None, file_p = None, ext_info=None, notify_fuc=None, notify_type= 'ws', keyList=None):
        if file_i and file_p:
            self.save_all(file_i, file_p)
        if notify_type == 'ws':
            if keephide:
                notify_fuc(notify_type, 'processing_done',{'processed_image_url': img_name, 'img_type': img_type, 'name': name, 'keephide': keephide}, to=r_id)
            else:
                notify_fuc(notify_type, 'processing_done', {'processed_image_url': img_name, 'img_type': img_type, 'name': name}, to=r_id)
        elif notify_type == 'tel':
            notify_fuc(notify_type, 'processing_done',
                       {'processed_image_url': img_name, 'img_type': img_type, 'name': name}, to=r_id, keyList=keyList)
        # 创建数据库连接
        conn = self._create_connection()
        try:
            # 创建游标对象
            t_cursor = conn.cursor()
            # 获取当前日期时间
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if ext_info is not None:
                # 执行插入操作，包括日期
                t_cursor.execute(
                    'INSERT INTO room_images (room_id, imgStr, type, name, created_at, ext_info) VALUES (?, ?, ?, ?, ?, ?)',
                    (r_id, img_name, img_type, name, created_at, ext_info))
            else:
                # 执行插入操作，包括日期
                t_cursor.execute('INSERT INTO room_images (room_id, imgStr, type, name, created_at) VALUES (?, ?, ?, ?, ?)',
                                 (r_id, img_name, img_type, name, created_at))
            # 提交事务
            conn.commit()

        except sqlite3.Error as e:
            # 如果出现错误，打印错误信息
            logger.info(f"SQLite error: {e}")

        finally:
            # 关闭数据库连接
            conn.close()

# 全局锁
user_states_lock = threading.Lock()

def query_or_def(user: User)->User:
    with user_states_lock:
        # 创建实例
        room_image_manager = RoomImageManager()
        db_user = room_image_manager.get_user(user.user_id)
        if db_user is not None:
            return db_user
        room_image_manager.insert_user(user)
        return user
def user_states_control(user: User):
    """
    更新 user_states 字典，支持一次传入多个键值对，或删除多个键。

    参数:
    - user_id: 用户ID
    - kv_dict: 需要更新的键值对字典
    - delete_keys: 要删除的键的列表
    """
    with user_states_lock:
        # 创建实例
        room_image_manager = RoomImageManager()
        db_user = room_image_manager.get_user(user.user_id)
        if db_user is None:
            room_image_manager.insert_user(user)
        else:
            room_image_manager.update_user(user)

def user_vip_control(user: User, conut):
    """
    更新 user_states 字典，支持一次传入多个键值对，或删除多个键。

    参数:
    - user_id: 用户ID
    - kv_dict: 需要更新的键值对字典
    - delete_keys: 要删除的键的列表
    """
    with user_states_lock:
        # 创建实例
        room_image_manager = RoomImageManager()
        db_user = room_image_manager.get_user(user.user_id)
        if db_user is None:
            user.vip_count=10
            room_image_manager.insert_user(user)
        else:
            user.vip_count = db_user.vip_count + conut
            room_image_manager.update_user(user)