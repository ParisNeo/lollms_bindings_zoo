from bs4 import BeautifulSoup
import re
html_content = """
<ul>
<li>license: gpl</li>
</ul>
<ul>
<li>Model creator: <a rel="noopener nofollow" href="https://huggingface.co/tiiuae">Technology Innovation Institute</a></li>
<li>Model creator: <a rel="noopener nofollow" href="https://huggingface.co/machakaka">Macha kaka</a></li>
<li>Model creator: <a rel="noopener nofollow" href="https://huggingface.co/chimachima">Chima chima</a></li>
</ul>
"""
text_to_find = "Model creator:"

soup = BeautifulSoup(html_content, 'html.parser')
# Find all <li> elements that contain the text "Model creator:"
model_creator_elements = soup.find_all('li')

# Initialize empty lists to store the extracted data
model_creator_names = []
model_creator_links = []

# Iterate through the model creator elements to extract the data
for element in model_creator_elements:
    if text_to_find in element.text:
        # Extract the model creator name (text within the <a> tag)
        name = element.find('a').text.strip()
        model_creator_names.append(name)
        
        # Extract the model creator link (href attribute of the <a> tag)
        link = element.find('a')['href']
        model_creator_links.append(link)

# Print the extracted data
for i in range(len(model_creator_names)):
    print(f"Model Creator Name: {model_creator_names[i]}")
    print(f"Model Creator Link: {model_creator_links[i]}")
    print()