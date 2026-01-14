from tavily_search.tavily_search_results_with_images import (
    TavilySearchResultsWithImages,
)
import json
from openai import OpenAI
from crawler import Crawler
import tiktoken

search_tool = TavilySearchResultsWithImages(
    name="web_search",
    max_results=10,
    include_raw_content=False,
    include_images=False,
    include_image_descriptions=False,
)

query = '2024年 APEC 部长级 会议 结束 后 紧接 召开 哪 一 场 会议'
from serpapi import GoogleSearch

serpapi_key = 'fc2e7a9c0fdcbbf92b3f3edc006355c1262e0ab5d2433b96923d1861f60350bf'

params = {
    "engine": "google",
    "q": query,
    'num': 10,
    'page': 1,
    'gl': 'us',
    'hl': 'en',
    'location': 'California, United States',
    'tbs': 'qdr:y',
    "api_key": serpapi_key,
}

crawler = Crawler()

parameters = {"url":"https://cs.zut.edu.cn/info/1021/2319.htm","visit_goal":"find host of seminar statement"}

url = parameters.get("url", "")
visit_goal = parameters.get("visit_goal", "")
print(url, visit_goal)

# article = crawler.crawl(url)
# article_content = article.to_markdown()

# print(article_content)

# summarize_model = 'Qwen/Qwen3-32B'
# client = OpenAI(base_url='http://fs-mbz-gpu-397:8888/v1', 
#                 api_key='EMPTY')
# max_tokens = 28000
# max_context_length = 32768

# enc = tiktoken.encoding_for_model("gpt-4o")

# def summarize_article(article_content: str, visit_goal: str) -> str:
#     # Check if the article content is too long
#     if len(enc.encode(article_content)) > max_tokens:
#         # Truncate the article content
#         article_content = enc.decode(enc.encode(article_content)[:max_tokens])
        
# #         prompt = f"""Please summarize the following webpage content. The user's goal is: **{visit_goal}**. \
# # Present the summary in markdown format, using bullet points or short paragraphs. Be clear and concise.

# # Webpage content:

# # {article_content}"""

#     prompt = f"""Please process the following webpage content and user goal to extract relevant information:

# ## **Webpage Content** 
# {article_content}

# ## **User Goal**
# {visit_goal}

# ## **Task Guidelines**
# 1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
# 2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
# 3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

# **Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
# """
#     # system_prompt = "You are an expert in summarizing web pages"
#     # prompt_length = len(self.enc.encode(prompt)) + 2 + len(self.enc.encode(system_prompt))
#     prompt_length = len(enc.encode(prompt)) + 1
    
#     if summarize_model.startswith('Qwen/Qwen3'):
#         response = client.chat.completions.create(
#             model=summarize_model,
#             messages=[{"role": "user",  
#                         "content": prompt}],
#             max_completion_tokens=min(max_context_length - prompt_length, 2048),
#             temperature=0.7,
#             top_p=0.8,
#             presence_penalty=1.5,
#             extra_body={
#                 "top_k": 20, 
#                 "chat_template_kwargs": {"enable_thinking": False},
#             },
#         )
#         # print(f'Qwen/Qwen3 Response: {response.choices[0]}\n\n')
#         # print('-' * 100)
#         # return response.choices[0].message.reasoning_content
#         return response.choices[0].message.content
        
#     else:
#         response = client.chat.completions.create(
#             model=summarize_model,
#             messages=[{"role": "user",  
#                     "content": prompt}],
#             max_completion_tokens=min(max_context_length - prompt_length, 2048),
#         )
#         return response.choices[0].message.content

# summary = summarize_article(article_content, visit_goal)
# print(summary)

# if summary.startswith('```json'):
#     summary = summary[7:]
# if summary.endswith('```'):
#     summary = summary[:-3]

# raw = json.loads(summary)
# print(raw)
# useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=visit_goal)
# useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
# useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"
# print(useful_information)

search = GoogleSearch(params)
results = search.get_dict()
# print(json.dumps(results, indent=2))

try:
    if "organic_results" not in results:
        raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

    web_snippets = list()
    idx = 0
    if "organic_results" in results:
        for page in results["organic_results"]:
            idx += 1
            date_published = ""
            if "date" in page:
                date_published = "\nDate published: " + page["date"]

            source = ""
            if "source" in page:
                source = "\nSource: " + page["source"]

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + page["snippet"]

            redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"

            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)

    content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
except:
    content = f"No results found for '{query}'. Try with a more general query, or remove the year filter."

print(content)
# A Google search for '2024年 APEC 部长级 会议 结束 后 紧接 召开 哪 一 场 会议' found 10 results:

# ## Web Results
# 1. [习近平主席出席亚太经合组织第三十一次领导人非正式会议 ...](http://www.cppcc.gov.cn/zxww/2024/11/25/ARTI1732496815134159.shtml)
# Date published: Nov 25, 2024
# Source: 中国政协网

# 此次利马会议，在中方积极推动下，各方发表了《2024年亚太经合组织领导人马丘比丘宣言》《关于亚太自由贸易区议程新展望的声明》《关于推动向正规和全球经 ...

# 2. [习近平出席亚太经合组织第三十一次领导人非正式会议并 ...](https://www.mfa.gov.cn/zyxw/202411/t20241117_11527668.shtml)
# Date published: Nov 17, 2024
# Source: 中华人民共和国外交部

# 当地时间2024年11月16日上午，亚太经合组织第三十一次领导人非正式会议在秘鲁利马会议中心举行。国家主席习近平出席会议并发表题为《共担时代责任共促 ...

# 3. [APEC與G20前瞻：習近平出訪拉美各國將共同面對「房間裡的 ...](https://www.bbc.com/zhongwen/articles/c8jywpkz7x7o/trad)
# Date published: Nov 13, 2024
# Source: BBC

# 兩場備受矚目的國際峰會將相繼在南美洲舉行——11月15日至16日，在秘魯首都利馬舉行亞太經合組織（APEC）峰會；11月18日至19日則在里約熱內盧舉行二十國 ...

# 4. [2024年秘鲁APEC峰会- 维基百科，自由的百科全书](https://zh.wikipedia.org/zh-hans/2024%E5%B9%B4%E7%A7%98%E9%AD%AFAPEC%E5%B3%B0%E6%9C%83)
# Source: 维基百科

# 2024年秘鲁APEC峰会又称亚太经济合作组织第31次领导人非正式会议，是亚太经济合作组织的年度会议，于2024年11月15日至16日在秘鲁举行。这也是秘鲁第三次举办APEC峰会。

# 5. [2024年APEC重要會議日程](https://www.trade.gov.tw/Pages/Detail.aspx?nodeid=1582&pid=776484)
# Source: 經濟部國際貿易署經貿資訊網

# 2024年APEC重要會議 ; 8/18. 糧食安全部長會議 (FSMM) ; 9/9-9/13, 中小企業部長會議 (SMEMM) ; 10/1-10/4. 財政部長會議 (FMM) ; 11/10當週. 總結資深官員會議 (CSOM) 年度 ...

# 6. [王毅谈习近平主席出席亚太经合组织第三十一次领导人非 ...](https://www.gov.cn/yaowen/liebiao/202411/content_6989057.htm)
# Date published: Nov 23, 2024
# Source: 中国政府网

# 2024年11月13日至23日，国家主席习近平应邀赴秘鲁出席亚太经合组织（APEC）第三十一次领导人非正式会议并对秘鲁进行国事访问，赴巴西出席二十国集团（G20） ...

# 7. [亚太经济合作组织 - 维基百科](https://zh.wikipedia.org/zh-hans/%E4%BA%9A%E5%A4%AA%E7%BB%8F%E6%B5%8E%E5%90%88%E4%BD%9C%E7%BB%84%E7%BB%87)
# Source: 维基百科

# 亚太经济合作组织（英语：Asia-Pacific Economic Cooperation，缩写：APEC），简称亚太经合组织，是亚太区内各地区之间促进经济成长、合作、贸易、投资的论坛。

# 8. [美国2024年亚太经济合作组织会议成果简报](https://china.usembassy-china.org.cn/zh/u-s-2024-apec-outcomes/)
# Date published: Nov 19, 2024
# Source: U.S. Embassy & Consulates in China

# APEC经济体领导人会议（APEC Economic Leaders）于11月15日至16日在利马举行会议，由秘鲁提出的会议主题为“赋权、包容、增长”（Empower. Include. Grow.）。本 ...

# 9. [活动会议 - 亚太经合组织中小企业信息化促进中心](http://apecsmei.org/eventmeeting.html)
# Source: apecsmei.org

# 2024-09-14. 第30次APEC中小企业部长会议在秘鲁普卡尔帕召开. 当地时间2024年9月13日，第30次亚太经合组织（APEC）中小企业部长会议在秘鲁普卡尔帕召开。会议主题是“赋能 ...

# 10. [专题报道：APEC峰会](https://www.voachinese.com/z/4297)
# Source: 美国之音

# 第30次亚太经合组织(APEC)领导人非正式会议11月15-17日在旧金山举行，对美国来说，这个成员国贸易占全球贸易一半的区域性经济论坛相当重要。然而，这次峰会可能被美国总统 ...

# result = search_tool.invoke({"args": {"query": query}, "type": "tool_call", "id": "foo", "name": "tavily"})

# print(result.content)
# [{"type": "page", "title": "2024 年亚太经合组织第35 届部长级会议联合声明（摘要）", "url": "https://gjs.mofcom.gov.cn/api-gateway/jpaas-web-server/front/document/download?fileUrl=YW5UzzlvCwcM%2FNHHX%2FtT6O3sd%2BAp1PLMt6L1NvhKfXVvsPzbLLKURhVByCKkrx%2ByXX2oZYaikulwWYWx8glPKMZu8Het9M%2FN9nfaRNeTtu9iBy2fe55XoqfN7W%2FYxd56mXGYe%2BLC7HeVlIMch6PjOptiOwvXlPUqLUpZA7WfzKE%3D&fileName=2024%E5%B9%B4%E4%BA%9A%E5%A4%AA%E7%BB%8F%E5%90%88%E7%BB%84%E7%BB%87%E7%AC%AC35%E5%B1%8A%E9%83%A8%E9%95%BF%E7%BA%A7%E4%BC%9A%E8%AE%AE%E8%81%94%E5%90%88%E5%A3%B0%E6%98%8E%E4%B8%AD%E8%AF%91%E6%96%87%EF%BC%88%E6%91%98%E8%A6%81%EF%BC%89.pdf", "content": "2024 年亚太经合组织第35 届部长级会议 联合声明（摘要） 秘鲁利马，2024 年11 月14 日 （中译文仅供参考） 我们，亚太经合组织（APEC）部长们，于2024 年11 月14 日在秘鲁利马举行会议，由外交部长埃尔默·希亚勒阁下和外贸 与旅游部长德西卢·莱昂女士共同主持。我们欢迎亚太经合组织 工商咨询理事会（ABAC）、东南亚国家联盟（ASEAN）、太 平洋经济合作理事会（PECC）以及联合国粮食及农业组织 （FAO）、美洲开发银行（IDB）、经济合作与发展组织 （OECD）、世界银行（WB）和世界贸易组织（WTO）的代表 参与会议。 我们重申对《APEC 2040 年布特拉加亚愿景》的承诺，即 到2040 年建立一个开放、活力、韧性和和平的亚太共同体，造 福所有人民和子孙后代。我们继续实施《奥特奥罗亚行动计 划》（APA），并在2024 年APEC 主题“赋能、包容、增长”的 指导下，通过三大优先主题推进了APEC 合作议程：促进包容 和互联增长的贸易投资；以创新和数字化推动向正规经济和全 球经济转型；推动可持续增长实现韧性发展。这些努力建立在 APEC [...] 我们承认结构性改革对包容性经济增长至关重要。（略） 我们赞赏第31 届APEC 财长会议在“可持续+数字+韧性 =APEC”主题下达成的共识成果。（略） 我们赞赏第14 届能源部长会议达成的共识成果。（略） 我们赞赏第9 届APEC 粮食安全部长会议达成的共识成 果。（略） 我们赞赏APEC 高级别矿业对话，认可矿业在亚太地区向 清洁、可持续、公正、可负担和包容的能源转型以及可持续经 济增长方面发挥的重要作用。 我们庆祝反腐败和透明度专家工作组（ACTWG）成立20 周年。（略） 加强APEC 我们重申APEC 作为亚太地区重要且有效的经济合作论坛 的地位，并重申其自愿、非约束和协商一致的原则。 我们对所有APEC 各委员会、工作组和分论坛以及APEC 秘书处和政策支持单位在2024 年的工作和贡献表示赞赏。我们 将努力确保APEC 秘书处和政策支持小组的财务可持续性。我 们感谢ABAC、PECC 和APEC 研究中心联盟提供的合作和贡 献，帮助我们取得全面成果，期待2025 年进一步与其深化合 作。 我们欢迎并注意到2024 年APEC 高官会主席报告和2024 [...] 年高官会年度经济技术合作报告。我们批准《本年度贸易和投 资委员会向部长提交的报告》。我们注意到ABAC 主席报告。 我们批准2025 年APEC 秘书处账户预算及2025 年各成员相应 分额。我们感谢成员对一般和专项子基金的贡献以及设立促进 绿色转型数字化子基金。 我们欢迎APEC 秘书处新任执行主任爱德华多·佩德罗萨成 功当选。同时，我们感谢即将卸任的执行主任丹斯里·丽贝卡·斯 达·玛丽亚博士在任期内所做出的杰出贡献。 我们鼓励与私营部门、非政府组织（NGOs）、民间社会和 青年等经济利益相关方扩大合作。为此，我们赞赏秘鲁发起的 国内倡议“面向民众的亚太经合组织”（APEC Ciudadano），这 代表了一种推动APEC 更贴近人民的务实方式，欢迎APEC 经 济体为此做出贡献。 我们感谢秘鲁对APEC 的坚定支持，感谢秘鲁主办2024 年 APEC 会议。我们再次欢迎并支持韩国主办2025 年APEC 会 议。我们高度重视APEC 在协商一致和多边主义基础上，依据 全体成员平等参与原则，按照《APEC 会议主办指南》及APEC 惯例，在包括领导人周在内的所有活动中继续开展合作。", "score": 0.61668247}, {"type": "page", "title": "2024年APEC年度部長會議聯合聲明", "url": "https://www.trade.gov.tw/Pages/Detail.aspx?nodeID=1585&pid=793555", "content": ":::\n   Image 33: 首頁圖標\n   國際組織協定\n   WTO、APEC等國際經貿組織\n   APEC(亞太經濟合作)\n   重要宣言及聲明\n\n2024年APEC年度部長會議聯合聲明\n-------------------\n\n 2024-11-21 多邊組經貿合作科\n\n2024年APEC年度部長會議(AMM)於11月14日在秘魯利馬舉行，我國由行政院楊政務委員珍妮及國家發展委員會劉主委鏡清代表出席。APEC會後發布「部長聯合聲明」，重點如下：\n\n在秘魯本年主題「賦權、包容、成長」下，透過三大優先領域「貿易及投資促進包容及連結成長」、「創新及數位化促進轉型至正式及全球連結成長」、「永續成長達致韌性發展」，持續強化亞太區域經濟合作，並鼓勵擴大利害關係人參與以回應人民需求，及透過執行「奧特亞羅瓦行動計畫」(APA)及「生物、循環與綠色(BCG)曼谷目標」、「舊金山原則」等，實現APEC 2040太子城願景承諾。 [...] 經濟部國際貿易署 ::: 重要宣言及聲明 \n\n===============\nSkip to main content block\n\n關閉\n\n### Image 1將這篇文章推薦給好友Image 2\n\n標題： 2024年APEC年度部長會議聯合聲明\n\n內容： 2024年APEC年度部長會議(AMM)於11月14日在秘魯利馬舉行，我國由行政院楊政務委員珍妮及國家發展委員會劉主委鏡清代表出席。APEC會後發布「部長聯合聲明」，重點如下：在秘魯本年主題「賦權、包...\n\nImage 3\n\n好友Email \n\nImage 4\n\n推薦人 \n\nImage 5\n\n推薦原因 \n\nImage 6\n\n驗證碼 \n\nImage 7: 驗證碼\n\nImage 8 更新驗證碼 \n\n取消 寄出\n\n跳到主要內容區塊 [...] APEC網站新聞稿及相關文件：APEC Ministers Issue Joint Statement\n\n點擊率：802\n\n2025年APEC貿易部長聯合聲明\n\n2024年APEC經濟領袖宣言\n\n返回前一頁\n\n:::\n\n網站導覽展開\n\n協助廠商出口\n   出口拓銷資源\n   補助資源\n   金融協助\n\n經貿往來\n   全球商機資訊\n   美國對等關稅\n   雙邊合作\n   國際情勢分析\n   貿易統計\n   貿易救濟與障礙\n   我國採行之貿易救濟措施\n   新南向政策專網\n   雙邊貿易協定\n   經貿簡報\n\n會展產業發展\n   推動會展產業發展\n   會展中心營運及管理\n   輔導資源\n\n貿易法規與管理\n   貿易法規\n   貿易管理\n   高科技貨品管理\n   線上聲明異議系統\n\n國際組織協定\n   WTO、APEC等國際經貿組織\n   CPTPP等區域經濟整合\n\n關於貿易署\n   位置簡圖\n   沿革\n   組織架構與職掌\n   署長簡介\n   副署長簡介\n   經濟部駐外單位\n   聯絡我們", "score": 0.57293844}, {"type": "page", "title": "美国2024年亚太经济合作组织会议成果简报 - State Department", "url": "https://2021-2025.state.gov/translations/chinese/20241118-u-s-2024-apec-outcomes-chinese/", "content": "APEC经济体领导人会议（APEC Economic Leaders）于 11 月 15 日至 16 日在利马举行会议，由秘鲁提出的会议主题为“赋权、包容、增长”（Empower. Include. Grow.）。本届会议旨在推进 APEC 在贸易和投资、数字化和创新以及可持续和包容性增长方面的重点要务。2024 年，美国与秘鲁及 APEC 伙伴合作，基于往年做出的承诺，其中包括在美国 APEC 主办年2023年APEC领导人金门宣言（2023 APEC Leaders’ Golden Gate Declaration）中列出的成果，以加强经济合作，促进可持续和包容性经济增长，应对区域挑战。除了拜登总统（President Biden）参加 APEC 经济体领导人会议外，国务卿安东尼·布林肯（Antony Blinken）和美国贸易代表戴琪（Katherine Tai）也参加了 11 月 14 日举行的 APEC 部长级会议。APEC 经济体领导人会议周还包括 APEC 工商领导人峰会（APEC CEO Summit）。在会议上，APEC [...] – 赞助在 APEC 经济体领导人会议周期间举办的“Hackathon Amazonia Impacta”，该活动旨在开发创新和实用的技术解决方案，以加强来自秘鲁亚马逊地区（Peruvian Amazon）的原住民女性与本地和全球市场接轨。Hackathon（黑客松）汇集了来自洛雷托（Loreto）、乌卡亚利（Ucayali）、圣马丁（San Mation）、马德雷德迪奥斯（Madre de Dios）和亚马逊（Amazonas）的大学生以及利马的学生。\n\n– 支持NCAPEC的可持续未来论坛（Sustainable Future Forum）和 11 月 13 日在利马举行的年度午餐会（Annual Luncheon），该活动在 APEC CEO 峰会之前举行。这项活动汇集了NCAPEC成员、美国政府和其他 APEC 经济体的高级别官员以及私营部门的代表，讨论应对该地区面临的种种挑战的策略。可持续未来论坛召集了当地和地区的利益相关者，以分享在农业综合企业和采矿业中推进关键的可持续增长目标的看法。\n\n_支持 APEC 作为一个机构的发展_ [...] – 通过APEC供应链互联互通联盟 （A2C2）开发应对当前供应链挑战的创新方法，该联盟将区域利益相关方聚集在一起，讨论旨在增强亚太地区供应链韧性的最佳规范和潜在解决方案。\n\n– 扩大APEC 服务贸易监管环境指数（APEC Index for Measuring the Regulatory Environment of Services Trade），以涵盖更多经济体和经济部门，同时召开多方利益相关者研讨会并发布关于制定服务技术标准的研究报告（study），为政策制定者、学者和行业领导者提供关键数据和最佳规范。\n\n– 通过共同发起APEC 部长们批准的《性别与结构性改革原则》（Gender and Structural Reform Principles），将性别平等制度化并提高其重要性。这些原则强调了结构性改革可以通过消除教育系统和工作场所中的障碍来促进妇女经济赋权的关键领域，包括对妇女和女童有重大影响的领导和决策职位。\n\n– 促进国际合作和制定标准，支持关键技术领域的创新和经济增长，包括促进人工智能、区块链和云计算等新兴技术在数字化和国际监管合作中的应用。", "score": 0.49802306}, {"type": "page", "title": "美国2024年亚太经济合作组织会议成果简报", "url": "https://china.usembassy-china.org.cn/zh/u-s-2024-apec-outcomes/", "content": "– 赞助在APEC经济体领导人会议周期间举办的“Hackathon Amazonia Impacta”，该活动旨在开发创新和实用的技术解决方案，以加强来自秘鲁亚马逊地区（Peruvian Amazon）的原住民女性与本地和全球市场接轨。Hackathon（黑客松）汇集了来自洛雷托（Loreto）、乌卡亚利（Ucayali）、圣马丁（San Mation）、马德雷德迪奥斯（Madre de Dios）和亚马逊（Amazonas）的大学生以及利马的学生。\n\n– 支持NCAPEC的可持续未来论坛（Sustainable Future Forum）和11月13日在利马举行的年度午餐会（Annual Luncheon），该活动在APEC CEO峰会之前举行。这项活动汇集了NCAPEC成员、美国政府和其他APEC经济体的高级别官员以及私营部门的代表，讨论应对该地区面临的种种挑战的策略。可持续未来论坛召集了当地和地区的利益相关者，以分享在农业综合企业和采矿业中推进关键的可持续增长目标的看法。\n\n_支持APEC作为一个机构的发展_ [...] APEC经济体领导人会议（APEC Economic Leaders）于11月15日至16日在利马举行会议，由秘鲁提出的会议主题为“赋权、包容、增长”（Empower. Include. Grow.）。本届会议旨在推进APEC在贸易和投资、数字化和创新以及可持续和包容性增长方面的重点要务。2024年，美国与秘鲁及APEC伙伴合作，基于往年做出的承诺，其中包括在美国APEC主办年2023年APEC领导人金门宣言（2023 APEC Leaders’ Golden Gate Declaration）中列出的成果，以加强经济合作，促进可持续和包容性经济增长，应对区域挑战。除了拜登总统（President Biden）参加APEC经济体领导人会议外，国务卿安东尼·布林肯（Antony Blinken）和美国贸易代表戴琪（Katherine Tai）也参加了11月14日举行的APEC部长级会议。APEC经济体领导人会议周还包括APEC工商领导人峰会（APEC CEO [...] – 商定一项由美国主导的多年期公正能源转型融资（Financing a Just Energy Transition）工作流程，该工作流程强调了与各经济体的净零/碳中和目标相一致的包容性和整体能源转型的可用政策工具。\n\n– 2024年粮食安全部长级会议批准了_《_防止和减少粮食损失和浪费的特鲁希略原则》（Trujillo Principles for Preventing and Reducing Food Loss and Waste），凸显了美国在《2023年通过可持续农业粮食系统实现粮食安全原则》（2023 Principles for Achieving Food Security Through Sustainable Agri-food Systems）中继续发挥的领导作用。这些原则旨在减少粮食损失和浪费，同时增强供应链韧性，刺激基础设施投资，并促进多个经济部门的参与。", "score": 0.47767657}, {"type": "page", "title": "2024 年亚太经合组织第30 届贸易部长会议联合声明", "url": "https://gjs.mofcom.gov.cn/api-gateway/jpaas-web-server/front/document/download?fileUrl=YW5UzzlvCwcM%2FNHHX%2FtT6O3sd%2BAp1PLMt6L1NvhKfXX9A2UD88aK66Rlyu%2FhVljEQFMyIgsRD2yErpQS7gKsxtlt%2FnewQ555VsxxKlpYIDHUQR03sWIirxr%2FStd8AtRj4%2F5wCe2%2F62w3x6W3oFN70sqOk2pFJhFN9KF%2FQcO8liA%3D&fileName=2024%E5%B9%B4%E4%BA%9A%E5%A4%AA%E7%BB%8F%E5%90%88%E7%BB%84%E7%BB%87%E7%AC%AC30%E5%B1%8A%E8%B4%B8%E6%98%93%E9%83%A8%E9%95%BF%E4%BC%9A%E8%AE%AE%E8%81%94%E5%90%88%E5%A3%B0%E6%98%8E%E4%B8%AD%E8%AF%91%E6%96%87.pdf", "content": "2024 年亚太经合组织第30 届贸易部长会议联合声明 秘鲁阿雷基帕，2024 年5 月18 日 （中译文仅供参考） 我们，亚太经合组织（APEC）贸易部长，于2024 年5 月 17 日至18 日在秘鲁阿雷基帕举行会议，由秘鲁外贸和旅游部 长伊丽莎白·加尔多阁下主持。我们欢迎世界贸易组织（WTO） 副总干事、APEC 工商咨询理事会（ABAC） 、东南亚国家联盟 （ASEAN） 、太平洋经济合作理事会（PECC）以及APEC 秘书 处参与会议。我们对阿雷基帕市的热情欢迎和接待表示衷心感 谢。 我们重申对《2040 年APEC 布特拉加亚愿景》的承诺，通 过实施《奥特奥罗亚行动计划》 ，以期到2040 年构建一个开 放、活力、韧性和和平的亚太共同体，实现全体人民和子孙后 代的繁荣。我们认识到自由、开放、公平、非歧视、透明、包 容和可预测的贸易投资环境的重要性，并将继续共同努力实现 该目标。 近年来，贸易持续面临强劲阻力。随着需求增长放缓， 2023 年全球商品贸易出现收缩，服务贸易增速显著下滑。全球 经济挑战及环境相关在内的其他因素扰乱了航运活动，导致贸 [...] 可使用的、完整的和运转良好的争端解决机制。 我们将通过对话协商、发挥领导力以及APEC 作为理念孵 化器的作用，与WTO 开展协作并提供支持，确保各经济体内 部和相互间共享贸易红利。为此，我们欢迎ABAC 代表团近期 对WTO 的访问，此举有效促进了公私部门对话。 我们欢迎2024 年2 月26 日至3 月2 日在阿联酋阿布扎比 举行的WTO 第十三届部长级会议（MC13）取得的成果，承诺 建设性推进会议成果全面落实。 我们认识到WTO 框架下诸边谈判对推进重点议题的积极 作用，欢迎《促进发展的投资便利化协定》完成文本磋商，呼 吁尽早将其纳入WTO 法律框架。我们对服务贸易国内规制联 合声明倡议下的纪律生效表示赞赏。 我们欢迎继续重振《电子商务工作计划》 ，注意到MC13 延 长电子传输暂免关税期限的决定。 我们敦促WTO 电子商务联合声明倡议参加方，加快达成 首套具有全球性影响力的电子商务承诺，为数字经济中的消费 者、劳动者和企业提供更明确预期。 我们赞赏WTO 就当代贸易议题深化讨论的努力。 我们欢迎15 个APEC 经济体核准《WTO 渔业补贴协定》 ， [...] 呼吁其余成员加快完成国内核准程序，促其早日生效。我们认 识到在进一步规范导致产能过剩和过度捕捞的渔业补贴方面取 得的进展，重申将早日完成相关谈判。 我们肯定WTO 在环境和可持续发展领域的工作，支持 WTO 贸易与环境委员会继续发挥重要作用，承诺将继续建设性 参与后续磋商。 我们对MC13 未能在农业改革方面取得实质性成果表示关 切。 我们认识到贸易在实现全球粮食安全和食品安全方面发挥 的积极作用，强调最大限度减少粮食供应链中断，促进农业贸 易投资，建设可持续和有韧性的农业食品系统，包括提升生产 力、资源利用率和系统包容性，减少粮食损耗和粮食浪费。我 们赞赏APEC 在这一领域发挥的建设性作用，包括通过《APEC 地区通过可持续农业食品系统实现粮食安全的原则》 ，将继续推 进《APEC 2030 年粮食安全路线图》目标。 亚太地区经济一体化依然是APEC 核心目标。我们重申以 市场驱动的方式推进亚太自由贸易区（FTAAP）建设的重要 性，为构建高标准和全面的区域安排做出贡献。我们肯定《北 京路线图》和《利马宣言》为推进FTAAP 议程所作的有益贡 献。 在推进FTAAP", "score": 0.47110215}, {"type": "page", "title": "亚太经合组织概况_中华人民共和国外交部", "url": "https://www.mfa.gov.cn/gjhdq_676201/gjhdqzz_681964/lhg_682278/jbqk_682280/", "content": "2023年11月，APEC第三十次领导人非正式会议在美国旧金山举行，会议围绕“为所有人创建强韧和可持续未来”的主题进行讨论，发表了《2023年APEC领导人旧金山宣言》。\n\n2024年11月，APEC第三十一次领导人非正式会议在秘鲁利马举行，会议围绕“赋能、包容、增长”的主题进行讨论，发表了《2024年APEC领导人马丘比丘宣言》、《关于亚太自由贸易区议程新展望的声明》和《关于推动向正规和全球经济转型的利马路线图》三份成果文件，并核可中方担任2026年APEC东道主。 [...] （三）高官会：每年一般举行4至5次会议，由各成员指定的高官（一般为副部级或司局级官员）组成。高官会的主要任务是负责执行领导人和部长会议的决定，审议各委员会、工作组和秘书处的活动，筹备部长级会议、领导人非正式会议及协调实施会议后续行动等事宜。\n\n（四）委员会和工作组：高官会下设4个委员会，即：贸易和投资委员会（CTI）、经济委员会（EC）、经济技术合作高官指导委员会（SCE）和预算和管理委员会（BMC）。CTI负责贸易和投资自由化方面高官会交办的工作，EC负责研究本地区经济发展趋势和问题，并协调经济结构改革工作，SCE负责指导和协调经济技术合作，BMC负责预算和行政管理等方面的问题。各委员会下设多个工作组、专家小组和分委会等机制，从事专业活动和合作。\n\n（五）秘书处：1993年1月在新加坡设立，为APEC各层次的活动提供支持与服务。秘书处负责人为执行主任，2010年起设固定任期，任期三年。现任执行主任为爱德华多·佩德罗萨（Eduardo Pedrosa ，菲律宾籍），于2025年1月就任，任期至2027年12月31日。 [...] 【成员和观察员】 APEC现有21个成员，分别是澳大利亚、文莱、加拿大、智利、中国、中国香港、印度尼西亚、日本、韩国、墨西哥、马来西亚、新西兰、巴布亚新几内亚、秘鲁、菲律宾、俄罗斯、新加坡、中国台北、泰国、美国和越南。此外，APEC还有3个观察员，分别是东盟秘书处、太平洋经济合作理事会、太平洋岛国论坛秘书处。\n\n【组织结构】 APEC共有5个层次的运作机制：\n\n（一）领导人非正式会议：1993年11月，首次APEC领导人非正式会议在美国西雅图召开，之后每年召开一次，一般于每年9月至11月间择日举行。\n\n（二）部长级会议：包括年度双部长会议以及专业部长会议。双部长会议每年在领导人会议前举行一次。专业部长会议定期或不定期举行，包括贸易部长会、财长会、中小企业部长会、能源部长会、海洋部长会、矿业部长会、电信部长会、旅游部长会、粮食安全部长会、林业部长会、结构改革部长会、交通部长会、人力资源部长会、妇女与经济高级别会议、卫生与经济高级别会议等。", "score": 0.41805586}, {"type": "page", "title": "[PDF] 2025 年亚太经合组织贸易部长会议联合声明", "url": "https://gjs.mofcom.gov.cn/api-gateway/jpaas-web-server/front/document/download?fileUrl=YW5UzzlvCwcM%2FNHHX%2FtT6O3sd%2BAp1PLMt6L1NvhKfXWKCCXhIddIKF4j3JX1AmPJY7LqS58lgTrJJUqKODzbd3Wb%2BOGPnGSbj9BL7dEe7f3SpTxJ8lMGVlWG0UJn4zw2NMEgXtOzqR06BsD1rLMw9spHUYXxbapLGr5FOLhN6oc%3D&fileName=2025%E5%B9%B4%E4%BA%9A%E5%A4%AA%E7%BB%8F%E5%90%88%E7%BB%84%E7%BB%87%E8%B4%B8%E6%98%93%E9%83%A8%E9%95%BF%E4%BC%9A%E8%AE%AE%E8%81%94%E5%90%88%E5%A3%B0%E6%98%8E%E4%B8%AD%E8%AF%91%E6%96%87.pdf", "content": "2025 年亚太经合组织贸易部长会议联合声明 韩国济州，2025 年5 月15-16 日 (中译文仅供参考) 我们，亚太经合组织（APEC）贸易部长，于2025 年5 月15 日 至16 日在韩国济州举行会议，会议由韩国贸易部长郑仁教主持。我 们欢迎世界贸易组织 （WTO） 总干事、 经济合作与发展组织 （OECD） 副秘书长、APEC 工商咨询理事会（ABAC）、东南亚国家联盟 （ASEAN）和太平洋经济合作理事会（PECC）的参与。 秉持2025 年韩国APEC 主题“构建可持续未来”的精神，我们 围绕三大重点议题推进APEC 议程： 多边贸易体制推动互联互通、 人 工智能创新促进贸易便利化、可持续贸易助力繁荣发展。 我们重申致力于落实《2040 年布特拉加亚愿景》 ，通过实施 《奥 特奥罗亚行动计划》 ， 为全体人民及子孙后代的福祉构建开放、 活力、 韧性、 和平的亚太共同体。 我们对全球贸易体系面临的根本性挑战表 示关切， 重申APEC 作为区域经济合作重要论坛的地位， 强调其在凝 聚共识应对区域经济挑战、 建设更具韧性和繁荣的亚太地区方面的重 要作用。 [...] 我们认识到世贸组织在推进贸易议题方面的重要性， 承认其既定 规则是国际贸易体系不可或缺的组成部分。 我们认识到世贸组织面临 挑战，需通过创新模式开展有意义、必要且全面的改革，以改善其职 能，使其在应对当今现实方面，更具相关性和响应性。我们赞赏在世 贸组织就当前贸易问题深化讨论的努力。 我们愿共同推动APEC 发挥 \"思想孵化器\"作用，支持成员协同合作，确保2026 年3 月在喀麦隆 举行的世贸组织第14 届部长级会议（MC14）取得成功。 我们欢迎16 个APEC 经济体接受世贸组织《渔业补贴协定》， 呼吁其他经济体完成内部程序， 鼓励所有世贸组织成员尽快完成额外 纪律谈判。 我们认识到在世贸组织就农业问题开展建设性讨论的必要 性。我们还注意到第13 届部长级会议关于电子传输暂免关税延期的 决定。 我们认识到增强数字经济可预期性的重要意义。 我们欢迎继续 重振电子商务工作计划相关工作的努力。 我们认识到世贸组织诸边谈判，包括联合声明倡议（JSIs），在 推动成员关切议题、 增强世贸组织相关性方面的积极作用。 我们欢迎 世贸组织在解决当代贸易问题、 激发新思路、 促进经济增长以及为多 [...] 相关原则、倡议及建议。我们认可原住 民（如适用）群体对经济增长所作出的宝贵贡献，并欢迎通过加强能 力建设领域的对话与合作，持续提升其参与区域及全球市场的水平。 我们感谢大韩民国成功主办本次会议，期待2025 年继续保持密 切合作。", "score": 0.3781381}, {"type": "page", "title": "亚太经济合作组织 - 维基百科", "url": "https://zh.wikipedia.org/zh-hans/%E4%BA%9A%E5%A4%AA%E7%BB%8F%E6%B5%8E%E5%90%88%E4%BD%9C%E7%BB%84%E7%BB%87", "content": "13. ^ Cabato, Luisa. Marcos skipping Apec Summit in Peru to focus on disaster response. Philippine Daily Inquirer. 7 November 2024  [7 November 2024]. （原始内容存档于2024-11-30） （英语）.\n14. ^ Metro Manila to host 2015 APEC leaders' meeting. Yahoo! News. 2013-08-29  [2014-03-26]. （原始内容存档于2014-03-27）.\n15. ^ Apec leaders' summit to be virtual. Bangkok Post. Kyodo News. 4 September 2020  [5 April 2021]. （原始内容存档于2023-04-13）. [...] | 峰会 |  1989  1990  1991  1992  1993  1994（英语：APEC Indonesia 1994）  1995（俄语：Саммит АТЭС 1995）  1996  1997  1998  1999  2000  2001  2002（英语：APEC Mexico 2002）  2003（英语：APEC Thailand 2003）  2004  2005  2006  2007（英语：APEC Australia 2007）  2008（英语：APEC Peru 2008）  2009（英语：APEC Singapore 2009）  2010  2011  2012  2013  2014  2015  2016  2017  2018  ~~2019~~  2020（英语：APEC Malaysia 2020）  2021（英语：APEC New Zealand 2021）  2022  2023  2024  2025（朝鲜语：2025년 경주 APEC 정상회의）  2026  2027 | [...] | 第31次 | 2023年 | 11月15日-17日 | 美国 | 乔·拜登 | 三藩市 |  |\n| 第32次 | 2024年 | 11月10日-16日 | 秘鲁 | 迪娜·博鲁阿尔特 | 库斯科 | ( （页面存档备份，存于互联网档案馆） |\n| 第33次（朝鲜语：2025년 경주 APEC 정상회의） | 2025年 | 10月31日-11月1日 | 韩国 | 李在明 | 庆州 |  |\n| 第34次 | 2026年 | 待公布 | 中华人民共和国 | 习近平 | 待公布 |  |\n| 第35次 | 2027年 | 待公布 | 越南 | 待公布 | 富国岛 |  |\n| 第36次 | 2028年 | 待公布 | 待公布 | 待公布 | 待公布 |  |\n| 第37次 | 2029年 | 待公布 | 待公布 | 待公布 | 待公布 |  |\n| 第38次 | 2030年 | 待公布 | 新加坡 | 黄循财 | 新加坡 |  |", "score": 0.3341118}, {"type": "page", "title": "习近平在亚太经合组织第三十一次领导人非正式会议上的 ...", "url": "https://www.mfa.gov.cn/zyxw/202411/t20241116_11527602.shtml", "content": "【_中_ _大_ _小_】\n\n打印;)\n\nImage 3Image 4Image 5\n\nImage 6\n\n共担时代责任 共促亚太发展\n\n——在亚太经合组织第三十一次领导人 非正式会议上的讲话\n\n（2024年11月16日，利马）\n\n中华人民共和国主席 习近平\n\n尊敬的博鲁阿尔特总统，\n\n各位同事：\n\n很高兴时隔8年再次来到美丽的“花园之都”利马，同大家共商亚太合作大计。感谢博鲁阿尔特总统和秘鲁政府为这次会议作出的周到安排。\n\n几十年来，亚太经合组织带动亚太地区实现大发展、大繁荣、大融通，助推亚太成为全球经济最具活力板块和主要增长引擎。当前，世界百年变局加速演进，世界经济增长乏力，世界开放指数不断下滑，亚太合作也面临地缘政治、单边主义和保护主义上升等挑战。站在历史的十字路口，亚太各国肩负更大责任。我们要团结协作，勇于担当，全面落实2040年布特拉加亚愿景，推动构建亚太命运共同体，努力开创亚太发展新时代。为此，我愿提出以下建议。 [...] 首页\n   外交部主要职责主要官员组织机构驻港、澳公署  \n   外交部长部长致辞部长简历部长活动讲话全文图片视频  \n   外交动态外事日程部领导活动业务动态例行记者会吹风会大使任免驻外报道政策解读  \n   驻外机构驻外使馆亚洲非洲欧洲北美洲南美洲大洋洲驻外总领馆亚洲非洲欧洲北美洲南美洲大洋洲驻外团、处常驻联合国代表团驻欧盟使团驻东盟使团驻非盟使团常驻联合国日内瓦办事处和瑞士其他国际组织代表团常驻维也纳联合国和其他国际组织代表团常驻联合国亚洲及太平洋经济和社会委员会代表处常驻国际海底管理局代表处驻立陶宛共和国代办处中国海地贸易发展办事处常驻世界贸易组织代表团常驻禁止化学武器组织代表团  \n   国家和组织国家（地区）国际和地区组织  \n   资料讲话全文声明公报条约文件政策文件中国外交历程中国外交人物外交史上的今天领事常识礼宾知识建交国家一览表专题  \n   政府信息公开\n   两微一端\n\n\n\n首页>重要新闻\n\n习近平在亚太经合组织第三十一次领导人非正式会议上的讲话（全文）\n===============================\n\n2024-11-16 23:43 [...] 二是培育绿色创新的亚太增长动能。我们要牢牢抓住新一轮科技革命和产业变革机遇，在人工智能、量子信息、生命健康等前沿领域加强交流合作，营造开放、公平、公正、非歧视的创新生态，推动亚太地区实现生产力跃升。要坚持生态优先、节约集约、绿色低碳发展，推进经济社会发展全面绿色转型，建设清洁美丽的亚太。要着力推动数字化绿色化协同转型发展，塑造亚太发展新动能新优势。\n\n中国正在因地制宜发展新质生产力，深化同各方绿色创新合作。中方将发布《全球数据跨境流动合作倡议》，愿同各方深化合作，共同促进高效、便利、安全的数据跨境流动。中方在亚太经合组织提出贸易单证数字化、绿色供应链能力建设、人工智能交流对话、粮食产业数字化等倡议，为亚太高质量发展贡献力量。", "score": 0.29930022}, {"type": "page", "title": "习近平主席出席亚太经合组织第三十一次领导人非正式会议 ...", "url": "http://www.cppcc.gov.cn/zxww/2024/11/25/ARTI1732496815134159.shtml", "content": "刚刚落幕的进博会，一家家亚太企业满载而归；刚刚开港的钱凯港，一些拉美邻国正准备前往考察。《区域全面经济伙伴关系协定》高质量实施，中国与东盟实质性结束自贸区3.0版升级谈判以及中秘签署自贸协定升级议定书的新消息令人振奋……\n\nImage 7\n\n11月7日，一名观众（左一）在第七届进博会国家展秘鲁馆了解秘鲁咖啡。新华社记者 张铖 摄\n\n世界瞩望着，感叹着：“中国是开放合作的真正引领者”“习主席让我们感受到中国的稳定和可靠”“全球需要像习主席这样的领导人”……\n\n此次利马会议，在中方积极推动下，各方发表了《2024年亚太经合组织领导人马丘比丘宣言》《关于亚太自由贸易区议程新展望的声明》《关于推动向正规和全球经济转型的利马路线图》三份成果文件。致力于2040年建成开放、活力、强韧、和平的亚太共同体，推进亚太自由贸易区建设，落实《亚太经合组织互联互通蓝图》……一项项重要的会议成果，饱含中国智慧的结晶。 [...] 有政府官员，有企业高管，有学者专家，7000多名代表从四面八方聚首利马。亚太经合组织经济体领导人的到来，备受瞩目。他们的抉择与担当，塑造着亚太合作的格局，也牵动着爪哇岛果农和安第斯山手艺人的生计。\n\n会旗迎风飘扬，各代表团车队接踵而至。16日上午，亚太经合组织第三十一次领导人非正式会议在利马会议中心拉开了大幕。习近平主席健步走来，博鲁阿尔特总统热情迎接，两国元首留下珍贵的“APEC瞬间”。\n\n时隔8年，为亚太发展再赴“利马之约”，是携手东道主推动开放合作的同声相应，是对“扎根亚太、建设亚太、造福亚太”承诺的信守不渝。\n\n会议中心利马厅圆形会议桌中央，镶嵌着醒目的会标。象征亚太经合组织21个经济体的21道光芒，组成一轮金色太阳，跃出层层海浪，冉冉升起。新的“利马时间”，将在亚太合作史上留下怎样的印记？世界之变、时代之变、历史之变以前所未有的方式展开，亚太合作何去何从，全球发展何去何从？\n\nImage 6\n\n这是11月8日在秘鲁利马拍摄的2024年亚太经济合作组织（APEC）会议标识。新华社记者 许睿 摄 [...] 政协工作专委会工作视察调研社情民意对外交往祖国统一书画京昆\n\n  \n   委员履职委员建言委员讲堂委员风采\n\n  \n   党派团体党派工作基层动态\n\n  \n   机关建设\n\n首页>政协要闻\n\n### “从历史长周期把握世界大势”——习近平主席出席亚太经合组织第三十一次领导人非正式会议并对秘鲁进行国事访问纪实\n\n_2024-11-25_ _来源：新华社_\n\n我要分享\n\n新浪微博QQ微信\n\nA-A+\n\n初夏的利马，一树树黄钟花簇拥绽放。太平洋的海风轻拂着“花园之都”，捎来友谊与合作的讯息。\n\n“希望您拨冗出席2024年亚太经合组织领导人非正式会议，并对秘鲁再次进行国事访问。”一年前，美国旧金山，秘鲁接棒亚太经合组织2024年东道主，博鲁阿尔特总统第一时间向习近平主席发出诚挚邀请。\n\n邀约穿越四季，情谊跨越山海。当地时间11月14日下午，习近平主席乘坐的专机抵达利马卡亚俄空军基地。在机场迎接的秘鲁部长会议主席阿德里安森紧握习近平主席的手：“谢谢习主席！您的到来令我们欢欣鼓舞。”\n\n东方大国的外交足迹，彰显着开拓与进取、格局与担当，映照着其与日俱增的国际影响力、感召力、塑造力。", "score": 0.21787827}]
