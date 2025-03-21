from flask import Flask, render_template, request, jsonify
import threading
# 初始化 Flask 应用
app = Flask(__name__)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from test_chat import DeepSeekModel

# 加载模型和分词器
model_name = "/home/hwtec/pythonWorkSpace/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model_chat = DeepSeekModel()

# 你需要根据实际情况实现 get_model_response 函数
def get_model_response(user_input):
    """
    调用本地部署的 deepseek-r1-qwen-1.5b 模型，获取回复
    :param user_input: 用户输入
    :return: 模型的回复
    """
    response_text = model_chat.generate_response(prompt=user_input, max_length=512, temperature=0.7)
    # 这里是你的模型调用逻辑
    # 例如：
    # response = model.generate(user_input)
    # return response
    return f"模型回复: {response_text}"  # 这里是一个占位符，替换为实际模型调用

# 控制聊天状态的标志
is_chatting = True

@app.route("/")
def home():
    """
    渲染聊天界面
    """
    return render_template("chat.html")
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data.get('input_text', '')
    response_text = model_chat.generate_response(prompt=input_text, max_length=4096, temperature=0.7)
    # 生成响应
    # inputs = tokenizer(input_text, return_tensors="pt")
    # outputs = model.generate(**inputs, max_length=50)
    # response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'response': response_text})

@app.route("/send", methods=["POST"])
def send_message():
    """
    处理用户发送的消息并返回模型回复
    """
    global is_chatting

    if not is_chatting:
        return jsonify({"response": "Chat stopped."})

    # 获取用户输入
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "No message provided."})

    # 调用模型获取回复
    response = get_model_response(user_input)

    # 返回模型回复
    return jsonify({"response": response})
@app.route("/stop", methods=["POST"])
def stop_chat():
    """
    停止聊天
    """
    global is_chatting
    is_chatting = False
    return jsonify({"response": "Chat stopped."})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9001)