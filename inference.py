from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(1234)




def infer(url, q):

    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)

    # use cuda device
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat-Int4", device_map="cuda",
                                                 trust_remote_code=True).eval()

    # 1st dialogue turn
    query = tokenizer.from_list_format([
        {'image': url},
        {'text': q},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    return response

def infer_trained(url, q):
    # Load the base model and tokenizer
    base_model_name = "Qwen/Qwen-VL-Chat-Int4"
    model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Load the adapter weights (LoRA)
    adapter_path = "/root/data/qwen/output"
    model.load_adapter(adapter_path)

    # Create input format
    query = tokenizer.from_list_format([
        {'image': url},
        {'text': q},
    ])

    # Generate response
    response, history = model.chat(tokenizer, query=query, history=None)
    return response

if __name__ == '__main__':
    url = "https://m.media-amazon.com/images/I/81+yG+f7PmL._SX679_.jpg"
    q = "This is the image of my product for listing on amazon. give me the title, description and keywords for listing"
    print(infer(url, q))
    print("===============================RESULTS OF FINETUNED MODEL=====================================")
    print(infer_trained(url, q))



