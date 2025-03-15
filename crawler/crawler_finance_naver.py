from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
from datetime import datetime, timedelta

# Selenium Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")  
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# ChromeDriver 자동 다운로드 및 실행
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# 날짜 범위 설정 (2020-01-01 ~ 2024-12-31)
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)

# 전체 뉴스 데이터를 저장할 리스트
all_news_data = []

# 날짜별 뉴스 크롤링 실행
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y%m%d")  
    print(f"{date_str} 날짜의 뉴스 크롤링 중...")

    # 네이버 금융 뉴스 URL 
    url = f"https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=429&date={date_str}"
    driver.get(url)
    time.sleep(3)  # 로딩 대기

    # 뉴스 목록 크롤링
    soup = BeautifulSoup(driver.page_source, "html.parser")
    news_list = soup.select("ul.realtimeNewsList li.newsList dl dd.articleSubject a")

    for news in news_list:
        title = news.get_text(strip=True)  # 뉴스 제목
        link = news["href"]  # 뉴스 링크

        # 개별 뉴스 페이지로 이동하여 본문 크롤링
        driver.get(link)
        time.sleep(2)  # 페이지 로딩 대기

        news_soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # 뉴스 본문 가져오기
        content = news_soup.select_one("#dic_area, .content, .articleCont")
        content = content.get_text(strip=True) if content else "본문 없음"

        # 뉴스 데이터 저장
        all_news_data.append({"date": date_str, "title": title, "url": link, "content": content})

    # 다음 날짜로 변경
    current_date += timedelta(days=1)

# 데이터프레임 변환 및 저장
df_news = pd.DataFrame(all_news_data)
df_news.to_csv("naver_finance_news_2020_2024.csv", index=False, encoding="utf-8-sig")

print(f"크롤링 완료! 총 {len(df_news)}개의 뉴스가 저장됨")

# 드라이버 종료
driver.quit()
