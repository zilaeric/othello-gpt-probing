import gdown

url = 'https://drive.google.com/drive/folders/1KFtP7gfrjmaoCV-WFC4XrdVeOxy1KmXe'
output = 'data/othello_championship/'
gdown.download_folder(url, output=output)

url = 'https://drive.google.com/drive/folders/1pDMdMrnxMRiDnUd-CNfRNvZCi7VXFRtv'
output = 'data/othello_synthetic/'
gdown.download_folder(url, output=output)