import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from aimo.sft_advanced import set_seed, get_hyperparameters, download_and_prepare_data, calculate_accuracy

# Main training script
if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_seed(42)

    # Configure basic training parameters
    # Configure training parameters
    train_url = "data/tagmynews3/train.jsonl"
    test_url = "data/tagmynews3/dev.jsonl"
    model_name = "finetuned_model/gpt2_generation_full/final"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and configure padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load pre-trained model and move to appropriate device
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Get training hyperparameters
    num_epochs, batch_size, learning_rate = get_hyperparameters()

    # Load and prepare training data
    train_loader, test_loader = download_and_prepare_data(train_url, test_url, tokenizer, batch_size)

    # Calculate final model performance
    train_acc = calculate_accuracy(model, tokenizer, train_loader)
    test_acc = calculate_accuracy(model, tokenizer, test_loader)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
