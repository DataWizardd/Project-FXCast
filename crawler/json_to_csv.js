// 실행 코드 : node json_to_csv.js


const fs = require("fs");
const path = require("path");
const { parse } = require("json2csv");

// 파일 경로
const inputPath = path.join(__dirname, "investing_results", "investing_comments_final.json");
const outputPath = path.join(__dirname, "investing_results", "investing_comments_final.csv");

try {
  const rawData = fs.readFileSync(inputPath, "utf-8");
  const jsonData = JSON.parse(rawData);

  // 변환할 필드 정의
  const fields = ["post_id", "nickname", "date", "content"];
  const opts = { fields };

  const csv = parse(jsonData, opts);
  fs.writeFileSync(outputPath, csv, "utf-8");

  console.log(`CSV 변환 완료! → ${outputPath}`);
} catch (err) {
  console.error("변환 오류:", err.message);
}
