import requests
import json


def main():
    url = "https://pre.in2x.com/dsr1/v1/chat/completions"

    payload = json.dumps({
        "model": "in2x-llm-deploy-ds-r1-250210",
        "messages": [
            {
                "role": "user",
                "content": "你好"
            }
        ],
        "stream": False
    }, ensure_ascii=False)

    headers = {
        'Content-Type': 'application/json',
        'appid': '',
        'Authorization': ''
    }

    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))

    print(response.text)


if __name__ == '__main__':
    main()
