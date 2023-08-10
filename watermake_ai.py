def embed_watermark(text, watermark_key=None):
    if watermark_key:
        watermarked_text = ""
        for i in range(len(text)):
            watermark_char = watermark_key[i % len(watermark_key)]
            watermarked_char = chr(ord(text[i]) ^ ord(watermark_char))
            watermarked_text += watermarked_char
        return watermarked_text, watermark_key
    else:
        return embed_watermark(text, watermark_key)  # To remove watermark, simply apply the same function

def extract_watermark(watermarked_text, watermark_key):
    extracted_watermark = ""
    for i in range(len(watermarked_text)):
        watermark_char = watermark_key[i % len(watermark_key)]
        extracted_char = chr(ord(watermarked_text[i]) ^ ord(watermark_char))
        extracted_watermark += extracted_char
    return extracted_watermark

def human_text(text):
    return text