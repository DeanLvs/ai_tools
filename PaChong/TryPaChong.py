from selenium import webdriver
from selenium.webdriver.chrome.service import Service  # Selenium 4.6+ 使用 Service
from chromedriver_py import binary_path  # 使用 chromedriver-py 自动管理驱动路径
from bs4 import BeautifulSoup
from Detail import detail
from concurrent.futures import ThreadPoolExecutor


def page_info(page_no):
    # 设置 Chrome 浏览器选项
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 无头模式
    options.add_argument("--disable-gpu")  # 禁用 GPU
    options.add_argument("--no-sandbox")  # 适合服务器环境
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.142 Safari/537.36")

    # 启动 Selenium WebDriver
    service = Service(binary_path)
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # 访问论坛页面，解析所有 thread_url
        forum_url = f"https://xc8866.cc/forum-1-{page_no}.htm?orderby=lastpid&digest=0"
        print(f"正在解析页面: {forum_url}")
        driver.get(forum_url)

        # 解析页面 HTML
        forum_html = driver.page_source
        forum_soup = BeautifulSoup(forum_html, "html.parser")

        # 查找所有 thread_url
        thread_list = forum_soup.select(
            "div.col-lg-7.main div.card.card-threadlist div.card-body ul.list-unstyled.threadlist.mb-0 li"
        )
        data_hrefs = [li.get("data-href") for li in thread_list if li.get("data-href")]

        # 输出解析到的帖子链接
        print(f"解析到的帖子链接 (页面 {page_no}):")
        # 遍历每个帖子链接并访问详情页
        for data_href in data_hrefs:
            thread_url = f"{data_href}"
            detail(thread_url)
    except Exception as e:
        print(f"页面 {page_no} 解析出错: {e}")
    finally:
        # 关闭浏览器
        driver.quit()
if __name__ == '__main__':
    # 使用 ThreadPoolExecutor 实现多线程
    with ThreadPoolExecutor(max_workers=1) as executor:
        # 提交任务，处理页面 1 到 128
        executor.map(page_info, range(1, 129))