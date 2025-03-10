import gdown
url = 'https://drive.google.com/uc?id=1MJ7AHiumIpTMJZAqJKyj8eCXeGhHr4-o'
output = 'multiencoder_model.pt'
gdown.download(url, output, quiet=False)