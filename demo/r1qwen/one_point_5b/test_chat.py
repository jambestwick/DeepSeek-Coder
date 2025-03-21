from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class DeepSeekModel:
    def __init__(self, model_path="/home/hwtec/pythonWorkSpace/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            # 添加稳定性相关参数
            # low_cpu_mem_usage=True,
            # attn_implementation="eager"  # 禁用可能不稳定的优化
        ).to(self.device)

        # 显式设置 pad_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置pad_token为eos_token
        # 设置模型为评估模式
        self.model.eval()

    def generate_response(self, prompt, max_length, temperature=0.7):
        try:
            # 对输入进行编码
            inputs = self.tokenizer(prompt, return_tensors="pt",padding=True, truncation=True).to(self.device)
            # 添加 attention_mask
            inputs['attention_mask'] = inputs.get('attention_mask', torch.ones(inputs['input_ids'].shape))

            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=False
                )

            # 解码输出
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 去掉输入部分，只输出生成的内容
            output_text = output_text[len(prompt):].strip()  # 去除输入的重复部分

            return output_text

        except Exception as e:
            #打印报错信息
            print(f"生成回答时发生错误: {str(e)}")
            return None


def main():
    # 初始化模型
    model = DeepSeekModel()
    try:
      # 测试对话
      while True:
        #try catch捕获整段异常并打印


        user_input = input("\n请输入您的问题 (输入 'quit' 退出): ")
        if user_input.lower() == 'quit':
            break

        response = model.generate_response(user_input)
        if response:
            print("\nDeepSeek:", response)
    except Exception as e:
        print(f"DeepSeek: {str(e)}")

if __name__ == "__main__":
    main()


