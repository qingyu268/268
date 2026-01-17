import http.client
import json

def ask_doubao(user_text):
    conn = http.client.HTTPSConnection("ark.cn-beijing.volces.com")

    payload = json.dumps({
        "model": "doubao-seed-1-6-lite-251015",
        "messages": [
            {
                "role": "system",
                "content": "你是一个有帮助的智能问答助手"
            },
            {
                "role": "user",
                "content": user_text
            }
        ]
    })

    headers = {
        'Authorization': 'Bearer 39f13dd2-acd2-4ff8-8238-945ba2e79131',
        'Content-Type': 'application/json'
    }

    conn.request("POST", "/api/v3/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")

    result = json.loads(data)
    return result["choices"][0]["message"]["content"]
