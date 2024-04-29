import requests
from bs4 import BeautifulSoup
import os
import time

def fetch_image_urls(query:str, max_links_to_fetch:int, wd:requests.Session, sleep_between_interactions:int=1):
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = wd.get(search_url, headers=headers)  # User-Agent 헤더를 사용합니다.

    image_urls = set()
    image_count = 0
    while image_count < max_links_to_fetch:
        soup = BeautifulSoup(response.text, 'html.parser')
        for img in soup.find_all('img'):
            src = img.attrs.get('src')  # 'src' 속성을 확인합니다.
            if src:
                image_urls.add(src)
                image_count += 1
                print(image_count)
                if image_count >= max_links_to_fetch:
                    break
        time.sleep(sleep_between_interactions)
        response = wd.get(search_url + f"&start={image_count}", headers=headers)  # 페이지를 넘기면서 계속 검색합니다.

    return image_urls

def download_images(folder_path:str, query:str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with requests.Session() as wd:
        images = fetch_image_urls(query, 470, wd)  # 5개 이미지를 검색
        for i, url in enumerate(images):
            img_data = wd.get(url).content
            img_file = os.path.join(folder_path, f"{query}_{i + 1}.jpg")
            with open(img_file, 'wb') as f:
                f.write(img_data)
            print(f"Downloaded {img_file}")

download_images("downloaded_images", "gun")