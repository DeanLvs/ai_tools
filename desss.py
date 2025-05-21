# -*- coding: utf-8 -*-
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_date, date_sub, current_timestamp, when, row_number, lit, desc
from pyspark.sql.window import Window
from pyspark.sql.functions import from_utc_timestamp
from datetime import datetime
import uuid
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType

# try:
#     # for python 2
#     reload(sys)
#     sys.setdefaultencoding('utf8')
# except:
#     # python 3 not needed
#     pass

PUSH_LOG_TABLE = "ods_f_push_logs_infos"
USER_PUSH_SETTING_TABLE = "ods_user_push_setting_infos"
USER_INFO_TABLE = "ods_users"
TALE_SESSION_TABLE = "ods_dialogue_infos"
QIN_MI_DU_TABLE = "ods_user_npc_infos"
NPC_INFO_TABLE = "ods_npc_infos"
RESULT_TABLE = "ods_f_push_infos"
RESULT_LOG_TABLE = "ods_f_push_logs_infos"
if __name__ == "__main__":
    print("===========")
    # 初始化Spark会话
    spark = SparkSession.builder \
        .appName("Inactive Dialogue Records with Date Range and Top 20 per User") \
        .config("spark.sql.shuffle.partitions", 200) \
        .config("spark.shuffle.compress", True) \
        .config("spark.shuffle.file.buffer", "64k") \
        .config("spark.shuffle.memoryFraction", 0.2) \
        .config("spark.task.maxFailures", 8) \
        .config("spark.executor.memoryOverhead", "1g") \
        .config("spark.sql.broadcastTimeout", 20 * 60) \
        .config("spark.sql.crossJoin.enabled", True) \
        .config("odps.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.sql.catalogImplementation", "odps") \
        .config("spark.sql.session.timeZone", "Asia/Tokyo") \
        .config("spark.sql.json.write.character.encoding", "UTF-8") \
        .getOrCreate()

    # 生成一个唯一的 UUID 并将其用作临时表名的一部分
    unique_key = str(uuid.uuid4()).replace("-", "")
    # temp_table_name = "ods_push_t_" + unique_key

    # 获取脚本执行时间并转换为字符串格式
    execution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Step: 统计 ods_user_push_log_infos 中每个 (user_id, push_npc_id) 的推送次数
    push_counts = spark.sql(
        "SELECT user_id, npc_id, COUNT(*) AS push_count FROM " + PUSH_LOG_TABLE + " WHERE push_status = 1 GROUP BY user_id, npc_id")
    print("=== Push counts per (user_id, npc_id) ===")
    push_counts.show(30)

    # Step: 读取并过滤允许推送的用户，仅保留 update_time 最大的记录
    window_push_setting = Window.partitionBy("user_id").orderBy(col("update_time").desc())
    ods_user_push_setting = spark.sql(
        "SELECT user_id, update_time FROM " + USER_PUSH_SETTING_TABLE + " WHERE enable_push = 1") \
        .withColumn("row_num", row_number().over(window_push_setting)) \
        .filter(col("row_num") == 1) \
        .drop("row_num")

    print("=== Filtered users with push enabled and latest update_time per user ===")
    ods_user_push_setting.show(30)

    # 获取启用推送的用户ID列表，供后续过滤
    push_enabled_user_ids = ods_user_push_setting.select("user_id").distinct()

    # 读取并过滤数据表
    ods_users_raw = spark.sql(
        "SELECT id as user_h_id, create_time as user_reg_time, status as user_status, languages_type FROM " + USER_INFO_TABLE)
    ods_users_raw = ods_users_raw.join(push_enabled_user_ids, ods_users_raw.user_h_id == push_enabled_user_ids.user_id,
                                       "inner").drop("user_id")

    ods_dialogue_info_raw = spark.sql(
        "SELECT id, dialogue_id, npc_id, story_id, user_id, dalogue_message_nums, real_dialogue_timestamp as modify_timestamp, update_version FROM " + TALE_SESSION_TABLE)
    ods_dialogue_info_raw = ods_dialogue_info_raw \
        .join(push_enabled_user_ids, on="user_id", how="inner") \
        .filter(col("modify_timestamp") >= date_sub(current_date(), 30))

    ods_user_npc_raw = spark.sql("SELECT id, intimacy_count, npc_id, user_id, update_version FROM " + QIN_MI_DU_TABLE)
    ods_user_npc_raw = ods_user_npc_raw.join(push_enabled_user_ids, on="user_id", how="inner")

    ods_npc_infos_raw = spark.sql(
        "SELECT id, npc_id, story_id, audit_status, is_off, update_version FROM " + NPC_INFO_TABLE) \
        .filter((col("audit_status") == 2) & (col("is_off") == 1))

    # Step 0: 获取每个表的唯一记录
    # 将 user_reg_time 转换为日本时间
    ods_users = ods_users_raw.withColumn("user_reg_time", from_utc_timestamp(col("user_reg_time"), "Asia/Tokyo"))
    #  1（即最近注册用户）0（为老客户）
    window_users = Window.partitionBy("user_h_id").orderBy(col("user_reg_time").desc())
    ods_users = ods_users.withColumn("row_num", row_number().over(window_users)) \
        .filter(col("row_num") == 1) \
        .drop("row_num") \
        .withColumn(
        "is_recent_user",
        when(col("user_reg_time") >= date_sub(current_timestamp(), 3), lit(1)).otherwise(lit(0))
    )

    window_dialogue = Window.partitionBy("id").orderBy(col("update_version").desc())
    ods_dialogue_info = ods_dialogue_info_raw.withColumn("row_num", row_number().over(window_dialogue)) \
        .filter(col("row_num") == 1) \
        .drop("row_num")

    window_user_npc = Window.partitionBy("id").orderBy(col("update_version").desc())
    ods_user_npc = ods_user_npc_raw.withColumn("row_num", row_number().over(window_user_npc)) \
        .filter(col("row_num") == 1) \
        .drop("row_num")

    # 为每个 id 选出 update_version 最新的记录
    window_npc = Window.partitionBy("id").orderBy(col("update_version").desc())
    ods_npc = ods_npc_infos_raw.withColumn("row_num", row_number().over(window_npc)) \
        .filter(col("row_num") == 1) \
        .drop("row_num")

    # Step 2: 精确计算每条记录距当前日期的天数，并添加未更新的天数范围标记列
    recent_inactive_dialogues = ods_dialogue_info \
        .filter(col("modify_timestamp") >= date_sub(current_date(), 30)) \
        .withColumn("days_ago", F.datediff(current_date(), col("modify_timestamp").cast("date"))) \
        .withColumn(
        "date_range",
        when(col("days_ago") == 0, "0")  # 当天
        .when(col("days_ago") == 1, "1")  # 1天前
        .when(col("days_ago") == 2, "2")
        .when(col("days_ago") == 3, "3")
        .when(col("days_ago") <= 6, "6")
        .when(col("days_ago") == 7, "7")
        .when(col("days_ago") == 8, "8")
        .when(col("days_ago") <= 14, "14")
        .when(col("days_ago") == 15, "15")
        .when(col("days_ago") <= 29, "29")
        .when(col("days_ago") == 30, "30")
        .otherwise("999")
    )
    print("=== Step 2: Filtered dialogues within 30 days and added date range column ===")
    recent_inactive_dialogues.show(300)

    # Step 3.1: 获取每个用户最近对话日期范围（date_range最小值）
    inactive_users = recent_inactive_dialogues.groupBy("user_id").agg(
        F.min("date_range").alias("date_range")
    ).distinct()

    # 不再使用 ods_users，所有用户统一视为 is_recent_user = 1
    # 仅保留 date_range == "1" 的记录
    inactive_users = inactive_users.filter(col("date_range") == "1")

    print("=== Step 3.1: Inactive users filtered by date_range == '1' ===")
    inactive_users.show(300)

    if inactive_users.rdd.isEmpty():
        print("inactive_users is empty so finish")
        # 清空 RESULT_TABLE
        spark.sql("TRUNCATE TABLE " + RESULT_TABLE)
        # 定义 DataFrame 的模式
        schema = StructType([
            StructField("user_id", StringType(), True),
            StructField("npc_id", StringType(), True),
            StructField("date_range", StringType(), True),
            StructField("total_dalogue_message_nums", IntegerType(), True),
            StructField("total_intimacy_count", IntegerType(), True),
            StructField("languages_type", StringType(), True),
            StructField("user_reg_time", TimestampType(), True),
            StructField("user_status", StringType(), True),
            StructField("is_recent_user", IntegerType(), True),
            StructField("latest_dalogue_message_nums", IntegerType(), True),
            StructField("max_dalogue_message_nums", IntegerType(), True),
            StructField("push_count", IntegerType(), True),
            StructField("business_time", StringType(), True)
        ])

        # 创建一个空的 DataFrame，使用指定的模式
        empty_record = spark.createDataFrame(
            [(None, None, None, None, None, None, None, None, None, None, None, None, execution_time)], schema
        ).withColumn("push_id", F.concat(F.lit(unique_key), F.lit("None"))) \
            .withColumn("create_time", F.lit(current_timestamp())) \
            .withColumn("start_gen_time", F.lit(None).cast("timestamp")) \
            .withColumn("gen_status", F.lit(0).cast("tinyint")) \
            .withColumn("gen_msg", F.lit(None).cast("string")) \
            .withColumn("finish_gen_time", F.lit(None).cast("timestamp")) \
            .withColumn("start_push_time", F.lit(None).cast("timestamp")) \
            .withColumn("finish_push_time", F.lit(None).cast("timestamp")) \
            .withColumn("push_status", F.lit(0).cast("tinyint")) \
            .withColumn("push_result", F.lit(None).cast("string")) \
            .withColumn("plan_push_time", F.lit(None).cast("timestamp"))

        # 将空记录插入到 RESULT_LOG_TABLE
        empty_record.write.mode("append").saveAsTable(RESULT_LOG_TABLE)
    else:
        # Step 3.1.1: 查找每个 user_id 在 ods_dialogue_info 中按 modify_timestamp 转换成天的分组，获取最近一天的最大 dalogue_message_nums
        window_latest_dialogue_group = Window.partitionBy("user_id", F.to_date(col("modify_timestamp")).alias(
            "modify_date")).orderBy(col("dalogue_message_nums").desc())
        latest_dalogue_message_nums = ods_dialogue_info \
            .withColumn("modify_date", F.to_date(col("modify_timestamp"))) \
            .withColumn("row_num", row_number().over(window_latest_dialogue_group)) \
            .filter(col("row_num") == 1) \
            .groupBy("user_id", "modify_date") \
            .agg(F.max("dalogue_message_nums").alias("latest_dalogue_message_nums")) \
            .orderBy(col("modify_date").desc()) \
            .groupBy("user_id") \
            .agg(F.first("latest_dalogue_message_nums").alias("latest_dalogue_message_nums"))

        print("=== Step 3.1.1: Latest non-zero dalogue_message_nums per user grouped by modify_date ===")
        latest_dalogue_message_nums.show(300)

        # Step 3.1.2: 获取所有历史对话中 dalogue_message_nums 的最大值
        max_dalogue_message_nums = ods_dialogue_info.groupBy("user_id").agg(
            F.max("dalogue_message_nums").alias("max_dalogue_message_nums")
        )

        print("=== Step 3.1.2: Maximum dalogue_message_nums per user across all history ===")
        max_dalogue_message_nums.show(300)

        # 将最新的 dalogue_message_nums 和历史最大值加入到 inactive_users 表中
        inactive_users = inactive_users \
            .join(latest_dalogue_message_nums, on="user_id", how="left") \
            .join(max_dalogue_message_nums, on="user_id", how="left")
        print("=== Step 3.1.2.1: inactive_users dalogue_message_nums per user across all history ===")
        inactive_users.show(300)

        # Step 3.2: 仅保留 inactive_users 中的 user_id，然后计算 dalogue_message_nums 和 intimacy_count 总和
        # 过滤出在 inactive_users 中的用户，避免不必要的计算
        print("=== Step 3.2: ods_user_npc all history ===")
        ods_user_npc.show(300)
        filtered_user_npc = ods_user_npc.join(inactive_users, on="user_id", how="inner")
        user_npc_stats = filtered_user_npc.groupBy("user_id", "npc_id").agg(
            F.sum("intimacy_count").alias("total_intimacy_count")
        )
        print("=== Step 3.2: user_npc_stats all history ===")
        user_npc_stats.show(300)

        filtered_user_dialogues = ods_dialogue_info.join(inactive_users, on="user_id", how="inner")
        user_dialogue_stats = filtered_user_dialogues.groupBy("user_id", "npc_id").agg(
            F.sum("dalogue_message_nums").alias("total_dalogue_message_nums")
        )

        # 合并 npc 信息和对话信息，计算每个 npc 的总 intimacy_count 和 dalogue_message_nums
        user_npc_dialogue_stats = user_npc_stats.join(
            user_dialogue_stats,
            on=["user_id", "npc_id"],
            how="inner"
        ).join(ods_npc.select("npc_id").distinct(), on="npc_id", how="inner")  # 去掉不在 ods_npc 中的 npc_id

        # Step 3.3: 获取每个用户 intimacy_count 和 dalogue_message_nums 排名前10的 npc_id
        window_spec = Window.partitionBy("user_id").orderBy(desc("total_intimacy_count"),
                                                            desc("total_dalogue_message_nums"))
        top_npc_per_user = user_npc_dialogue_stats.withColumn("row_num", row_number().over(window_spec)) \
            .drop("row_num")

        print("=== Step 3.3: Top 5 NPCs per user based on intimacy_count and dalogue_message_nums ===")
        top_npc_per_user.show(300)

        # Step 3.4: 生成最终结果表

        # 针对所有用户统一使用一个 Window 排序规则（不再区分 is_recent_user=0/1）
        window_spec = Window.partitionBy("user_id").orderBy(
            col("push_count").asc(),
            col("total_intimacy_count").desc(),
            col("total_dalogue_message_nums").desc()  # 或根据需要使用 max_dalogue_message_nums
        )

        final_result = (
            top_npc_per_user
            .join(inactive_users, on="user_id", how="inner")
            .withColumn("row_num", row_number().over(window_spec))
            .filter(col("row_num") == 1)
            .drop("row_num")
            .join(push_counts, on=["user_id", "npc_id"], how="left")
            .join(ods_user_push_setting, on="user_id", how="left")
            .select(
                "user_id",
                "npc_id",
                "date_range",
                "total_dalogue_message_nums",
                "total_intimacy_count",
                # 以下字段原本来自 ods_users，这里直接用 lit(None) 或 lit(1) 等替代
                lit(None).cast("string").alias("languages_type"),
                lit(None).cast("timestamp").alias("user_reg_time"),
                lit(None).cast("string").alias("user_status"),
                lit(1).alias("is_recent_user"),
                "latest_dalogue_message_nums",
                "max_dalogue_message_nums",
                F.coalesce("push_count", lit(0)).alias("push_count"),
                lit(execution_time).alias("business_time")
            )
        )

        print("=== Step 3.4: Final result table with unique record per user_id ===")
        final_result.show(300)

        # Step 3.5: 将结果写入事先创建好的表 ods_push_infos，使用追加模式
        final_result = final_result.withColumn("push_id", F.concat(F.lit(unique_key), col("user_id").cast("string"))) \
            .withColumn("create_time", F.lit(current_timestamp())) \
            .withColumn("start_gen_time", F.lit(None).cast("timestamp")) \
            .withColumn("gen_status", F.lit(0).cast("tinyint")) \
            .withColumn("gen_msg", F.lit(None).cast("string")) \
            .withColumn("finish_gen_time", F.lit(None).cast("timestamp")) \
            .withColumn("start_push_time", F.lit(None).cast("timestamp")) \
            .withColumn("finish_push_time", F.lit(None).cast("timestamp")) \
            .withColumn("push_status", F.lit(0).cast("tinyint")) \
            .withColumn("push_result", F.lit(None).cast("string")) \
            .withColumn("plan_push_time", F.lit(None).cast("timestamp"))
        # 增加分区数，平衡数据
        final_result = final_result.repartition(200)

        # Step 3.5: 将结果写入新的表 ods_f_push_infos
        final_result.write.mode("overwrite").saveAsTable(RESULT_TABLE)

        # 将结果写入 RESULT_LOG_TABLE 表（使用追加模式）
        final_result.write.mode("append").saveAsTable(RESULT_LOG_TABLE)