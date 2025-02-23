import os
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from langchain_core.documents import Document

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


load_dotenv("../../credentials.env")


graph = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD,
    refresh_schema=False
)

# graph.query(
#     """
#         MATCH (n) DETACH DELETE n
#     """
# )

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# text = """
# $ tree -I 'node_modules|cache|test_*' .
# .
# ├── app
# │   ├── assets
# │   │   ├── fonts
# │   │   │   ├── GeistMonoVF.woff
# │   │   │   └── GeistVF.woff
# │   │   ├── icons
# │   │   │   ├── bottom_bar_calendar.svg
# │   │   │   ├── bottom_bar_chat.svg
# │   │   │   ├── bottom_bar_geolocation.svg
# │   │   │   ├── bottom_bar_house_heart_2.svg
# │   │   │   ├── bottom_bar_house_heart.svg
# │   │   │   ├── bottom_bar_house.svg
# │   │   │   ├── bottom_bar_maplocation.svg
# │   │   │   ├── bottom_bar_more.svg
# │   │   │   ├── bottom_bar_user.svg
# │   │   │   ├── camera.svg
# │   │   │   ├── dszero.svg
# │   │   │   ├── edit.svg
# │   │   │   ├── hamburger.svg
# │   │   │   ├── help_info.svg
# │   │   │   ├── hive-home.svg
# │   │   │   ├── icon_meu_lar_appointment.svg
# │   │   │   ├── icon_meu_lar_bell.svg
# │   │   │   ├── icon_meu_lar_calendar.svg
# │   │   │   ├── icon_meu_lar_chat.svg
# │   │   │   ├── icon_meu_lar_finance.svg
# │   │   │   ├── icon_meu_lar_handshake.svg
# │   │   │   ├── icon_meu_lar_idea.svg
# │   │   │   ├── icon_meu_lar_info.svg
# │   │   │   ├── icon_meu_lar_msg.svg
# │   │   │   ├── icon_meu_lar_plan.svg
# │   │   │   ├── icon_meu_lar_settings.svg
# │   │   │   ├── icon_meu_lar.svg
# │   │   │   ├── login.svg
# │   │   │   ├── small_right_arrow.svg
# │   │   │   ├── tiktok.svg
# │   │   │   ├── tree.svg
# │   │   │   ├── twitter.svg
# │   │   │   └── youtube.svg
# │   │   └── images
# │   │       ├── 080e5ecb-2334-4b91-862e-e63a515612dc.png
# │   │       ├── 2fcb3a44-26ce-41b7-a181-f6c55f663025.png
# │   │       ├── 3e0561c9-0b3e-4c49-ad83-eda501c558c0.png
# │   │       ├── 8e96f071-f681-4fe3-b12c-6275503f963e.png
# │   │       ├── 92928c35-54cd-43a3-a7bb-cf0fe0feca07.png
# │   │       ├── card.svg
# │   │       ├── cellphone_01.gif
# │   │       ├── chat_01.png
# │   │       ├── co-parenting_01.jpg
# │   │       ├── co-parenting_01_transformed.jpeg
# │   │       ├── d381ebf2-00aa-4419-a6d8-6bb3b71880e8.png
# │   │       ├── d381ebf2-00aa-4419-a6d8-6bb3b71880e9.png
# │   │       ├── dszero_logo_02.svg
# │   │       ├── dszero_logo.svg
# │   │       ├── hand_house_vertical.jpg
# │   │       ├── hand_house_vertical_rect_2.jpg
# │   │       ├── hand_house_vertical_rect.jpg
# │   │       ├── hand_house_vertical_square.jpg
# │   │       ├── kids.svg
# │   │       ├── tree-animation.gif
# │   │       └── tree-animation.mp4
# │   ├── (auth)
# │   │   ├── login
# │   │   │   └── page.tsx
# │   │   ├── reset-password
# │   │   │   └── page.tsx
# │   │   └── signup
# │   │       ├── components
# │   │       │   ├── AccountInfoStep.tsx
# │   │       │   ├── BasicInfoStep.tsx
# │   │       │   ├── KidsInfoStep.tsx
# │   │       │   ├── ProfilePictureStep.tsx
# │   │       │   └── SignupProgress.tsx
# │   │       ├── hooks
# │   │       │   └── useSignupForm.ts
# │   │       ├── layout.tsx
# │   │       └── page.tsx
# │   ├── components
# │   │   ├── Analytics.tsx
# │   │   ├── BottomNav.tsx
# │   │   ├── ContentArea.tsx
# │   │   ├── CoparentingCalendar.tsx
# │   │   ├── friendship
# │   │   │   ├── FriendList.tsx
# │   │   │   ├── FriendRequests.tsx
# │   │   │   └── FriendSearch.tsx
# │   │   ├── layout
# │   │   │   ├── CustomTypingEffect.tsx
# │   │   │   ├── Footer.tsx
# │   │   │   ├── Header.tsx
# │   │   │   └── LoginHeader.tsx
# │   │   ├── Sidebar.tsx
# │   │   ├── ui
# │   │   │   ├── ErrorDisplay.tsx
# │   │   │   ├── LoadingPage.tsx
# │   │   │   └── NavLink.tsx
# │   │   └── UserProfileBar.tsx
# │   ├── favicon.ico
# │   ├── globals.css
# │   ├── layout.tsx
# │   ├── lib
# │   │   └── firebaseConfig.ts
# │   ├── manifest.json
# │   ├── page.tsx
# │   ├── static
# │   └── (user)
# │       └── [username]
# │           ├── calendario
# │           │   └── page.tsx
# │           ├── chat
# │           │   └── page.tsx
# │           ├── finances
# │           │   └── page.tsx
# │           ├── geolocation
# │           │   └── page.tsx
# │           ├── handshake
# │           │   └── page.tsx
# │           ├── home
# │           │   └── page.tsx
# │           ├── info
# │           │   └── page.tsx
# │           ├── lar
# │           ├── layout.tsx
# │           ├── page.tsx
# │           ├── plan
# │           │   ├── page.tsx
# │           │   └── resumo
# │           │       └── page.tsx
# │           └── settings
# │               └── page.tsx
# ├── context
# │   └── userContext.tsx
# ├── cors.json
# ├── firebase.json
# ├── LICENSE
# ├── _middleware.ts
# ├── next.config.mjs
# ├── next-env.d.ts
# ├── package.json
# ├── package-lock.json
# ├── postcss.config.mjs
# ├── public
# │   └── images
# │       ├── card.png
# │       └── kids.svg
# ├── README.md
# ├── robots.txt
# ├── tailwind.config.ts
# ├── tree.txt
# ├── tsconfig.json
# └── types
#     ├── friendship.types.ts
#     ├── shared.types.ts
#     ├── signup.types.ts
#     ├── svg.d.ts
#     ├── tailwindcss-bg-patterns.d.ts
#     └── user.types.ts

# 35 directories, 123 files
# """


# # prompt = """
# # The most important node property is the file extension. You can add this property to the nodes. It usually is the last part of the file name after the dot.
# # """
# allowed_relationships = [
#     ("Folder", "CONTAINS", "File"),
#     ("Folder", "CONTAINS", "Folder"),
# ]
# llm_transformer = LLMGraphTransformer(
#     llm = llm,
#     allowed_nodes=["Folder", "File",],
#     allowed_relationships = allowed_relationships,
#     # additional_instructions = prompt,
#     node_properties = True # ["file_extension"],
# )

# documents = [Document(page_content=text)]
# graph_documents = llm_transformer.convert_to_graph_documents(documents)

# print(f"Nodes:{graph_documents[0].nodes}")
# print(f"Relationships:{graph_documents[0].relationships}")

# graph.add_graph_documents(
#     graph_documents,
#     baseEntityLabel=True,
#     # include_source=True
# )

# subgraph = graph.query(
#     """
#         MATCH p=()-[]->() RETURN p 
#     """
# )

chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True,
    allow_dangerous_requests=True
)

response = chain.invoke({"query": '''
Use the context to answer the user's question:

For example:                         
QUESTION: how many files are in the folder "Types"?
CYPHER: MATCH (t:Folder {id: "Types"})-[:CONTAINS]->(child) RETURN count(child) AS containedNodesCount;
CONTEXT: [{'containedNodesCount': 6}]
ANSWER: The folder "Types" contains 6 files.
                                      
---

USER: how many files are in the folder "App"?

'''})

print(response['result'])