import urllib.request
import re


with open('input', 'r') as img_list:
  for img_url in img_list:
    img_url = img_url.replace('\n', '')
    found = re.search('sat//(.+?)$', img_url).group(1)
    urllib.request.urlretrieve(img_url, 'input_images/' + found)

with open('labels', 'r') as img_list:
  for img_url in img_list:
    img_url = img_url.replace('\n', '')
    found = re.search('map//(.+?)$', img_url).group(1)
    urllib.request.urlretrieve(img_url, 'label_images/' + found)