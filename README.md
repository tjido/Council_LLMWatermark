# Differenciating AI content from Human Content Using UNICODE VALUES

The Idea behind this Work is Motivated by this post: https://acroll.substack.com/p/all-you-need-is-unicode

### IDEA
Mark AI-written text with different Unicode values so that we can distinguish AI-written text from human-written text even though the letters look the same. Unicode works across all devices and websites, and doesn’t need any changes to how people use the Internet. Plus, the people behind the Unicode governing body include many of the companies who are building Generative AI.

A world where we can tell the difference between human- and machine-generated content is better for humanity. We have labels for the things we put in our bodies; we should care as much about what we put in our minds. 

In this project, we are distinguishing between text from a human (PDF documents) and text from the LLM.


### Step to reproduce the code.

1. cd to chatbot repo `cd chatbot`
2. install requirements `pip install -r requirements.txt`
3. run `python main.py` on the terminal
4. After running the `main.py` file, the input will pop up.
    - enter pdf file name. eg: `msft-10K-2022.pdf`
    - enter company  name. eg: `Microsoft`
    - Query the pdf file: `How is the financial performance of Microsoft?`
5. The Output displays on the command line for both 
    - Hidden Unicode WaterMark AI Text
    - AI Output
    - Human Output

### Logic for the Unicode AI WaterMark 
Watermarking text using Unicode characters to differentiate between AI-generated and human-written content.

1. Embedding the Watermark: The embed_watermark function adds the hidden watermark to the original text. It works like this:
    - This function takes the original text and the watermark you want to hide. It goes through each character in the text:

    If the character is a letter, it adds the next character from the watermark followed by the letter itself. This hides the watermark within the text. If the character is not a letter (like a space or punctuation), it just adds the character as it is.

2. Revealing the Original text: The goal is to reveal the text from the watermarked text. Here's how the reveal_watermark_text function does it:

- This function goes through the watermarked text and checks each character:

    If the character matches the next watermark character, it means we've found a watermark character. So, we move to the next watermark character.
    If the character doesn't match, it's a regular letter, so we add it to the revealed text.
    
    This code uses the reveal_watermark function to extract the hidden watermark and prints the revealed text.

#### Chatbot Output

![Chatbot Output](chatbot\watermark_ai.png)



