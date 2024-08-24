import os
import json


def process_subdirectory(subdirectory_path):
    json_path = os.path.join(subdirectory_path, 'product-info-chat.json')
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as file:
        product_info = json.load(file)

    conversations = product_info.get('conversations', [])
    subdirectory_image_files = os.path.join(subdirectory_path, "images")
    image_files = [f for f in os.listdir(subdirectory_image_files)
                   if os.path.isfile(os.path.join(subdirectory_image_files, f))]

    # Assuming there is only one image per conversation for simplicity
    if not image_files:
        return None

    image_reference = f"<img>{subdirectory_image_files}/{image_files[0]}</img>"

    formatted_conversations = [
        {
            "from": "user",
            "value": f"Picture 1: {image_reference}"
        },
        {
            "from": "assistant",
            "value": conversations[1]['content']
        }
    ]
    for conv in conversations:
        role = 'user' if conv['role'] == 'user' else 'assistant'
        content = conv['content'].replace('Picture 1: ', f'Picture 1: {image_reference}\n')
        formatted_conversations.append({
            "from": role,
            "value": content
        })

    return {
        "id": f"identity_{os.path.basename(subdirectory_path)}",
        "conversations": formatted_conversations
    }


def prepare_data_for_finetuning(base_path):
    all_conversations = []
    for subdir in os.listdir(base_path):
        subdirectory_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdirectory_path):
            result = process_subdirectory(subdirectory_path)
            if result:
                all_conversations.append(result)

    output_path = f'{base_path}qwen_prepared_data.json'
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(all_conversations, outfile, ensure_ascii=False, indent=4)

    print(f"Data preparation complete. Output saved to {output_path}")


if __name__ == "__main__":
    base_path = '/root/data/images/'
    prepare_data_for_finetuning(base_path)

