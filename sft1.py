# ...existing code...

import llamafactory

def main():
    """
    一个使用 llamafactory 进行模型微调的完整示例
    """
    # 1. 加载预训练模型
    base_model = llamafactory.load_model("DeepSeek-R1-Distill-Qwen-7B")

    # 2. 准备微调数据
    training_data = [
        {"input": "示例输入1", "output": "示例输出1"},
        {"input": "示例输入2", "output": "示例输出2"},
    ]

    # 3. 设置微调参数并执行微调
    fine_tuned_model = llamafactory.fine_tune(
        model=base_model,
        train_data=training_data,
        epochs=3,
        batch_size=8,
        learning_rate=1e-4,
        distrubuted=True,
        num_gpus=2,
    )

    # 4. 保存微调后的模型
    fine_tuned_model.save("fine_tuned_llama_model")
    print("微调完成并已保存到 fine_tuned_llama_model")

if __name__ == "__main__":
    main()