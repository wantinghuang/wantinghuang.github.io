layout:post
title:微軟資料科學家面試
---
# Microsoft Data Science Interview
微軟資料科學家面試總共分四個部分
1. Team Intro
  > SwiftKey is an Keyboard APP，是華為、小米內建的鍵盤。
  > 本次在台灣招人的目標：中文的language model與使用者資料開發。
  
2. Questions to them
* 這份職位要做什麼？
    * 要分析User behavioral資料，告訴PM分析結果、跟PM溝通，建構Language Model開發產品
* 當嘗試過許多model、參數調整，結果卻不夠完美，如何突破？從何開始？
    * 先質疑feature好不好-->改進方法：domain knowledge很重要，不同管道收集user behavior
    * side by side跟同業比較，去看到功能以外的東西
3. About me and my project
* 實習：資料前處理、資料視覺化、建立類神經網路模型、調整清洗機參數使得清洗結果最好。
  面試官提問：
  * * 建立什麼模型？
  * * 怎麼做feature selection？
  * * LASSO為什麼可以做feature selection？
  * * L1 norm可以使得部分權重變為0，L2 norm為什麼不行？
  * * 怎麼處理missing value？有沒有其他非regression的方法可以做feature selection？
  * * 如果今天建立了兩個模型，要怎麼拿到真實世界（生產線上）測試哪一個好？
* 自然語言處理期末專題：給予電影台詞文本，生成電影名稱。先斷詞，然後找搭配字(2-word) collocation作為電影名稱。如果沒有搭配字，則去標記詞性，找最常出現的時間副詞，和最常出現的名詞或動詞搭配，成為電影名稱。（因為時間副詞帶來空間感，例如明天過後、明日世界、昨日盛開的花朵）在班上贏得『最符合電影內容』的獎項。
  面試官提問：
  * * 要如何用supervised learning的方法去做這個題目、建模？
4. Coding
```Given a list of string, find the index with the longest string.
# ex: ['aa','bbss','ddd','wwww','sss','t']```
