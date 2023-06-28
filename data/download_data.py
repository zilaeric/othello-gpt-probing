import gdown

url = 'https://drive.google.com/file/d/1A3XFmZ2YdgbRzqXfk9qIHDIyUCfa96FS/view?usp=drive_link'
output_path = 'data/othello_championship.zip'
gdown.download(url, output_path, quiet=False, fuzzy=True)

url = 'https://drive.google.com/file/d/1YGDX3vZh5Hk3QnVL1NpMrTIRtzwwQNUP/view?usp=drive_link'
output_path = 'data/othello_synthetic.zip'
gdown.download(url, output_path, quiet=False, fuzzy=True)
