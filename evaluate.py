import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sft import set_seed, get_hyperparameters, download_and_prepare_data, calculate_accuracy_batch

# Main training script
if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_seed(42)

    # Configure basic training parameters
    # Configure training parameters
    # train_url = "/data/coding/upload-data/data/math_data/V1_filtered/test_large_data_filtered.jsonl"
    test_url = "/data/coding/upload-data/data/math_data/V1_filtered/test_small_data_filtered.jsonl"
    model_name = "/data/coding/upload-data/data/DeepSeek-R1-Distill-Qwen-7B"
    peft_model = "/data/coding/upload-data/data/finetuned_model/deepseek-lora/final"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and configure padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load pre-trained model and move to appropriate device
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # model = PeftModel.from_pretrained(model, peft_model)
    model = model.to("cuda:1")

    # Get training hyperparameters
    num_epochs, batch_size, learning_rate = get_hyperparameters()

    # Load and prepare training data
    test_loader = download_and_prepare_data(test_url, tokenizer, batch_size)

    # Calculate final model performance
    test_acc = calculate_accuracy_batch(model, tokenizer, test_loader)
    print(f"Test accuracy: {test_acc:.4f}")
