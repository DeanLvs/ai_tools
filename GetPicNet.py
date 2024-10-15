import requests
from bs4 import BeautifulSoup

# 目标URL
url = 'https://xc8866.cc/forum-1-1.htm?tagids=0_0_0_0'

# 自定义请求头
headers = {
    'authority': 'xc8866.cc',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Cache-Control': 'no-cache',
    'Cookie': 'bbs_sid=dqdfe9gve0fq0gtc8hifks07uh; _ga=GA1.1.305675034.1721461607; cf_clearance=kndPr3GC1N2R3N4KA3ad5GiNvAljlOQw7MYrzxtbJN4-1721461607-1.0.1-1-nmGrHtp01dZtiJf6Zpiza_7INyC1oaHpnXMIQAU3ICgJ1s28QhWda99jkvbmNVKqyAE0WKT8tIG9IcokdTKKHw; _ga_MVNT0D87ZD=GS1.1.1721461606.1.1.1721461771.0.0.0',
    'Pragma': 'no-cache',
    'Priority': 'u=0, i',
    'Sec-Ch-Ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"macOS"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
}

# 发送HTTP GET请求
response = requests.get(url, headers=headers)

# 检查请求是否成功
if response.status_code == 200:
    # 解析网页内容
    soup = BeautifulSoup(response.content, 'html.parser')
    # 提取所有 href 以 thread- 开头的链接
    links = [li['data-href'] for li in soup.find_all('li', class_='media thread tap')]


    for link in links:
        print(f'提取到的链接: {link}')
        # 访问每个链接
        response = requests.get(f'https://xc8866.cc/{link}', headers=headers)
        if response.status_code == 200:
            print(f'访问 {link} 成功')
            # 处理返回的内容
            content = response.content
            # 使用 BeautifulSoup 解析 HTML 内容
            soup = BeautifulSoup(content, 'html.parser')
            # 提取 break-all font-weight-bold 里的文字
            thread_title = soup.find('h4', class_='break-all font-weight-bold').text
            print(f'帖子标题: {thread_title}')

            # 提取 table-responsive 中每行每列的值
            table = soup.find('table', class_='table-responsive')
            rows = table.find_all('tr')
            table_data = []
            for row in rows:
                cols = row.find_all(['th', 'td'])
                table_data.append([col.text.strip() for col in cols])

            print('表格内容:')
            for row in table_data:
                print(row)

            # 提取 container mt-5 中每一行 img 的 URL
            container = soup.find('div', class_='container mt-5')
            img_tags = container.find_all('img')
            img_urls = [img['src'] for img in img_tags]

            print('图片 URLs:')
            for url in img_urls:
                print(url)

            # 进一步的处理代码
        else:
            print(f'访问 {link} 失败，状态码: {response.status_code}')
else:
    print(f'请求失败，状态码：{response.status_code}')