import openai
import requests

# Set your OpenAI API key
openai.api_key = ""

def generate_prompt_gpt4o(user_input):
    """Use GPT-4o to enhance the image description"""
    response = openai.completions.create(
        model="gpt-4o",  # GPT-4o model
        prompt=f"Describe an image for the following idea: {user_input}",
        max_tokens=150  # Limit the length of the generated text
    )
    return response["choices"][0]["text"].strip()

def generate_image(prompt, output_path="generated_image.png"):
    """Use OpenAI's DALL·E to generate an image from a detailed prompt"""
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    
    image_url = response['data'][0]['url']
    print(f"Generated Image URL: {image_url}")

    # Download and save image
    img_data = requests.get(image_url).content
    with open(output_path, "wb") as img_file:
        img_file.write(img_data)
    print(f"Image saved as {output_path}")

# Example Usage
user_text = "shakshuka"
detailed_prompt = generate_prompt_gpt4o(user_text)
print(f"Generated Prompt: {detailed_prompt}")
generate_image(detailed_prompt)
