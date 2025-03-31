from selenium import webdriver
from selenium.webdriver.chrome.service import Service  # Selenium 4.6+ 使用 Service
from chromedriver_py import binary_path  # 使用 chromedriver-py 自动管理驱动路径
from bs4 import BeautifulSoup
import json
def detail(thread_uri, output_file="./nv_info.json"):
    # 设置 Chrome 浏览器选项
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 无头模式
    options.add_argument("--disable-gpu")  # 禁用 GPU
    options.add_argument("--no-sandbox")  # 适合服务器环境
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.142 Safari/537.36")

    # 启动 Selenium WebDriver
    service = Service(binary_path)
    driver = webdriver.Chrome(service=service, options=options)

    # 访问帖子页面
    thread_url = "https://xc8866.cc/"+thread_uri #thread-88392.htm"
    driver.get(thread_url)

    # 获取帖子详情页 HTML
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # 解析 message break-all 下的 blockquote
    message_block = soup.select_one("div.message.break-all blockquote")
    nv_info = {}
    if message_block:
        # 提取 <h3> 中的 <span> 内容
        h3_span = message_block.select_one("h3 span")
        if h3_span:
            nv_info['JJ'] = h3_span.text.strip()
        else:
            print(f"error h3_span {thread_uri}")
        # 提取 table.table-striped 中的内容
        table = message_block.select_one("table.table-striped")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                columns = row.find_all(["th", "td"])  # 获取表头和单元格
                if len(columns) >= 2:  # 确保有至少两列
                    key = columns[0].text.strip()  # 第一列作为键
                    value = columns[1].text.strip()  # 第二列作为值
                    nv_info[key] = value  # 存储到字典中
                else:
                    print(f"error mei liang lie {thread_uri}")
    else:
        print(f"未找到 class='message break-all' 的 blockquote {thread_uri}")

    # 解析 message break-all 下的 container mt-5
    container = soup.select_one("div.message.break-all div.container.mt-5")
    img_list = []
    if container:
        modal_images = container.select("div.modal.fade img.img-fluid")
        for img in modal_images:
            img_src = img.get("src")
            img_list.append(img_src)
    else:
        print(f"未找到 class='container mt-5' 的内容 {thread_uri}")
    nv_info['pic'] = img_list
    # 打印并保存 nv_info 为 JSON 文件
    print(nv_info)
    try:
        with open(output_file, "a", encoding="utf-8") as f:
            json.dump(nv_info, f, ensure_ascii=False, indent=4)
            f.write(",\n")  # 添加逗号分隔，用于存储多个对象
    except Exception as e:
        print(f"写入文件出错: {e}")
    # 关闭 WebDriver
    driver.quit()
if __name__ == '__main__':
    detail('thread-88392.htm')

