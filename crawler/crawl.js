// 실행 코드 : node crawl.js

const puppeteer = require("puppeteer");
const fs = require("fs");
const path = require("path");

(async () => {
  const browser = await puppeteer.launch({
    headless: true, 
    defaultViewport: null,
  });

  const page = await browser.newPage();
  const startId = 374; // to 2024
  const endId = 12375; // from 2020
  const results = [];

  for (let postId = startId; postId <= endId; postId++) {
    const url = `https://kr.investing.com/currencies/usd-krw-commentary/${postId}`;
    console.log(` ${postId}번 페이지 크롤링 중...`);

    try {
      await page.goto(url, {
        waitUntil: "domcontentloaded",
        timeout: 15000,
      });

      // 대기
      await page.waitForSelector("#comments_new > div", { timeout: 5000 });

      const commentDataList = await page.evaluate(() => {
        const outerDivs = document.querySelectorAll("#comments_new > div");
        const results = [];

        outerDivs.forEach((outerDiv, index) => {
          if (index === 0) return; // 광고/제목 스킵

          const innerDiv = outerDiv.querySelector("div");
          if (!innerDiv) return;

          const contentEl = innerDiv.querySelector("div.break-words.leading-5");
          const nicknameEl = innerDiv.querySelector("a");
          const dateEl = innerDiv.querySelector("span");

          if (contentEl && nicknameEl && dateEl) {
            results.push({
              nickname: nicknameEl.innerText.trim(),
              date: dateEl.innerText.trim(),
              content: contentEl.innerText.trim(),
            });
          }
        });

        return results;
      });

      if (commentDataList.length > 0) {
        commentDataList.forEach(comment => {
          results.push({
            post_id: postId,
            ...comment,
          });
        });
      } else {
        console.log(`${postId}번: 댓글 없음`);
      }
    } catch (err) {
      console.log(`${postId}번 오류: ${err.message}`);
    }

    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  await browser.close();

  const saveDir = path.join(__dirname, "investing_results");
  if (!fs.existsSync(saveDir)) fs.mkdirSync(saveDir);

  const finalPath = path.join(saveDir, "investing_comments_final.json");
  fs.writeFileSync(finalPath, JSON.stringify(results, null, 2), "utf-8");

  console.log(`\n크롤링 완료! 총 ${results.length}개 댓글 저장됨 → ${finalPath}`);
})();
