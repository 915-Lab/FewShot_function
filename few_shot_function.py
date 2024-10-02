from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pandas as pd
import csv
import openpyxl
import os

class FewShotFunction():
    def __init__(self):
        # column 변수 이름 지정
        self.user_id = "id"
        self.diary_col = "diary"
        self.summary_diary = "summary_diary"
        self.feedback = "feedback"
        self.emotion_diary = "emotion_diary"
        # 파일 이름 지정
        self.xlsx_file_name = "data.xlsx"
        self.csv_file_name = "data.csv"
        self.chroma_path = "chroma_db"
        self.file_path = "file/"

    def load_env(self):
        load_dotenv('.env')
        os.getenv("OPENAI_API_KEY")

    def make_files(self):
        excel_path = os.path.join(self.file_path, self.xlsx_file_name)
        csv_path = os.path.join(self.file_path, self.csv_file_name)
        # 파일들을 담을 디렉토리 생성
        if not os.path.exists(self.file_path):
            print(f"{self.file_path} 디렉토리를 생성합니다")
            os.mkdir(self.file_path)
        else:
            print(f"{self.file_path} 디렉토리가 존재합니다.")

        # 엑셀 파일 생성
        if not os.path.exists(excel_path):
            print(f"{self.xlsx_file_name} 이름으로된 엑셀파일을 생성합니다.")
            make_xlsx = openpyxl.Workbook()
            make_xlsx.save(excel_path)
        else:
            print(f"{self.xlsx_file_name} 파일이 존재합니다.")

        # 엑셀 파일을 csv 파일로 변환후 저장
        if not os.path.exists(csv_path):
            print(f"{self.csv_file_name} 이름으로된 csv파일을 생성합니다.")
            xlsx_data = pd.read_excel(excel_path)
            xlsx_data.to_csv(csv_path, index=False, encoding='utf-8')

        # csv 파일 칼럼 넣기
        with open(csv_path, mode='w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.user_id,
                self.diary_col,
                self.summary_diary,
                self.feedback,
                self.emotion_diary
            ])
        return excel_path, csv_path
    def utilize_prompts(self, my_diary):
        # llm 객체 생성
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        # 일기 요약 프롬프트 작성
        summary_prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             """
             Your role is to read the diary and summarize the diary based on it.
             You must read the diary and summarize it in Korean.
             The summary should be no more than three lines.
             The diary is as follows.
             diary: {diary}
             """
             ),
            ("human", "Input_diary: {input_diary}")
        ])

        # few_shot 데이터
        self.examples = [
            {
                "input":
                    """
                    오늘 회사에서 정말 기분 좋은 하루를 보냈다. 아침부터 맡았던 업무가 아주 잘 풀렸다. 특히 이번에 맡은 프로젝트가 마무리 단계에 접어들면서, 내가 준비한 자료들이 팀 내에서 큰 호응을 얻었다. 동료들이 내 자료를 보고 칭찬해줘서 기분이 한층 더 좋아졌다. 오랜만에 내 일이 빛을 발하는 것 같아서 뿌듯했다.
                    점심시간에도 분위기가 좋았다. 동료들과 함께 식사를 하며, 팀원들이 다 같이 웃고 이야기 나누는 모습이 너무 즐거웠다. 요즘 다들 바빴는데, 오늘은 조금 여유롭게 대화를 나눌 수 있어서 좋았다. 동료들과 소소한 이야기를 나누며 웃다 보니, 회사 생활도 꽤 즐거운 부분이 많다는 생각이 들었다.
                    오후에는 중요한 미팅이 있었다. 처음에는 약간 긴장이 되었지만, 내가 준비한 내용을 발표하면서 분위기가 점점 좋아졌다. 상사도 내 발표를 긍정적으로 평가해줘서 자신감이 생겼다. 이런 순간들이 회사 생활의 기쁨이구나 싶었다.
                    하루를 마무리하며 책상 정리를 하는데, 오늘 하루 정말 알차게 보낸 것 같아서 기분이 좋았다. 팀원들과 협력하며 성과를 내고, 그 과정에서 인정받는 순간들이 쌓여가면서 앞으로의 회사 생활도 기대된다. 내가 맡은 일을 잘 해냈다는 생각에 집에 가는 발걸음이 가벼웠다.
                    정말 기분 좋은 하루였다!
                    """,
                "output": "긍정"
            },
            {
                "input":
                    """
                    오늘은 정말 마음에 안 드는 하루였다. 아침부터 업무가 꼬이기 시작했다. 내가 맡은 프로젝트에서 중요한 자료를 놓치는 바람에 팀원들이 다 같이 당황하는 상황이 벌어졌다. 그동안 준비했던 부분이 제대로 반영되지 않아서 상사에게 지적을 받았을 때, 정말 속상했다. 나름대로 열심히 했다고 생각했는데, 결과가 이렇게 나와서 자존심이 상했다.
                    점심시간도 별로였다. 동료들과 점심을 먹었지만, 내내 어색한 분위기였다. 다들 바쁘고 피곤한지, 대화도 별로 없었고 내가 꺼낸 이야기들도 반응이 시원치 않았다. 혼자서만 말을 이어가다 보니, 더 기분이 처지는 느낌이었다. 그냥 혼자 조용히 밥을 먹을 걸 그랬나 싶었다.
                    오후에는 미팅이 있었는데, 여기서도 일이 잘 풀리지 않았다. 발표를 하면서 중요한 내용을 실수로 빼먹었고, 그 때문에 상사에게 다시 설명하느라 진땀을 뺐다. 다른 팀원들이 나를 바라보는 시선이 느껴지면서, 긴장해서 더 말을 꼬이기 시작했다. 끝나고 나서도 계속 그 생각이 머릿속을 떠나지 않았다. '왜 이렇게 실수만 했을까'라는 자책이 가득했다.
                    퇴근할 때는 온몸이 무거웠다. 하루 종일 뭔가 잘 안 풀리는 기분에, 회사에서의 나 자신이 한없이 작아 보였다. 집에 돌아가는 길에도 계속 오늘 있었던 일들이 떠올라서, 마음이 가라앉았다. 오늘 하루는 정말 힘들고, 기분 나쁜 날이었다.
                    빠르게 이 하루가 끝났으면 좋겠다.
                    """,
                "output": "부정"
            },
            {
                "input":
                    """
                    오늘은 정말 성과가 많은 하루였다! 아침부터 중요한 투자 미팅이 있었는데, 투자자들이 우리 사업 아이디어에 깊은 관심을 보여서 기분이 아주 좋았다. 사업 계획서를 꼼꼼히 준비한 보람이 있었던 것 같다. 미팅이 끝나고 나서 바로 긍정적인 답변을 받아, 투자 유치가 거의 확정되었다는 소식까지 들었다. 이제 우리 회사가 한 단계 더 도약할 수 있는 기회가 생긴 셈이다.
                    점심시간에는 오랜만에 창업 동료들과 함께 식사를 했다. 다들 바쁜 와중에도 시간을 내어 모여서 그동안의 성과를 나누고, 앞으로의 계획에 대해 이야기를 나눴다. 팀원들이 열정적으로 일하는 모습을 보니, 나도 힘이 난다. 정말 좋은 팀을 만난 게 큰 행운이라는 생각이 들었다.
                    오후에는 고객과의 중요한 미팅도 있었다. 우리 제품에 대해 긍정적인 피드백을 받았고, 몇 가지 추가 요구사항만 반영하면 계약이 성사될 가능성이 높아졌다. 요즘 사업이 탄력을 받으면서 조금씩 성장하는 걸 보니 정말 뿌듯하다.
                    오늘 하루는 기대했던 것 이상으로 좋은 일들로 가득했다. 사업을 시작한 지 얼마 안 됐지만, 내가 올바른 길을 가고 있다는 확신이 든다. 앞으로도 이 흐름을 유지하면서 더 많은 성과를 내야겠다.
                    """,
                "output": "긍정"
            },
            {
                "input":
                    """
                    오늘은 정말 글이 안 써지는 날이었다. 아침부터 책상 앞에 앉아 있었지만, 몇 시간 동안 거의 진전이 없었다. 새로운 소설을 쓰기 시작한 지 얼마 안 됐는데, 벌써 벽에 부딪힌 것 같다. 이야기가 머릿속에서 엉키기만 하고, 캐릭터도 제자리를 잡지 못해 답답하다. 이렇게 글이 안 풀릴 때마다 스스로가 무능력하게 느껴진다.
                    출판사에서 수정 요청도 들어왔는데, 생각했던 것보다 많은 부분을 고쳐야 한다고 해서 기분이 더 나빠졌다. 나는 나름대로 만족했는데, 그게 아니라고 하니 자존심도 상하고 자신감도 떨어졌다. 점심시간에도 머릿속에 온통 글 생각뿐이라 제대로 쉬지도 못했다.
                    오후가 되자 더 집중하려고 커피까지 마셨지만, 오히려 피곤함만 더 쌓였다. 하루 종일 쓰고 지우기를 반복하다 보니 이제는 머리가 멍해지고, 글쓰기에 대한 스트레스만 커졌다. 빨리 원고를 끝내야 한다는 압박감이 나를 짓누르는 것 같다.
                    오늘은 글이 손에 잡히지 않아 정말 힘든 하루였다. 내일은 조금 더 나아지기를 바랄 뿐이다.
                    """,
                "output": "부정"
            },
            {
                "input":
                    """
                    오늘은 대학 생활이 정말 즐거운 날이었다! 아침 수업은 조금 어려웠지만, 교수님께서 중요한 개념을 잘 설명해주셔서 이해가 쉽게 되었다. 요즘 수업이 점점 더 흥미로워지고 있어서 공부하는 게 재미있다. 수업이 끝난 후 도서관에서 팀 프로젝트를 위한 자료도 정리했는데, 오늘은 집중이 잘 되어서 많은 진전을 이뤘다.
                    점심시간에는 동기들과 캠퍼스 근처 맛집에 갔는데, 오랜만에 다 같이 모여서 먹는 점심이 정말 즐거웠다. 서로의 고민도 나누고, 이번 주말에 있을 여행 계획도 세우면서 웃고 떠들었다. 대학 생활의 이런 소소한 즐거움이 참 좋다.
                    오후에는 클럽 활동도 있었다. 동아리에서 함께 기획하는 프로젝트가 점점 형태를 갖춰가는 걸 보니 뿌듯하다. 팀원들과 협력하면서 배우는 것도 많고, 새로운 사람들과 소통하는 것도 재밌다. 앞으로도 이 활동을 통해 더 많은 경험을 쌓고 싶다.
                    하루가 정말 알차게 지나가서 기분이 좋다. 이번 학기 목표도 잘 이루고 있는 것 같아 뿌듯하고, 앞으로 더 많은 기회를 만들 수 있을 것 같다는 생각이 든다.
                    """,
                "output": "긍정"
            },
            {
                "input":
                    """
                    오늘은 학교에서 정말 별로인 하루였다. 아침에 일어나기도 힘들었고, 수업도 지루하기만 했다. 특히 수학 시간이 너무 어려워서 선생님 설명을 따라가기 힘들었다. 문제를 풀 때마다 헷갈리고 실수해서 결국 숙제도 제대로 못 끝냈다. 친구들은 다들 잘하는 것 같아서 더 속상했다.
                    점심시간에도 기분이 나빴다. 친구들이랑 같이 밥을 먹었는데, 내가 말한 농담이 별로 재미없었는지 다들 반응이 시원치 않았다. 분위기가 어색해져서 괜히 내가 잘못한 것처럼 느껴졌다. 원래 점심시간은 즐거워야 하는데, 오늘은 혼자 있는 기분이었다.
                    오후에 체육 시간에는 더 실망스러웠다. 농구를 했는데, 나는 팀에서 거의 활약을 못해서 친구들한테 미안했다. 잘하고 싶었는데, 몸이 제대로 따라주지 않아 더 위축됐다. 체육이 끝나고 나니 하루 종일 내가 쓸모없는 사람처럼 느껴졌다.
                    오늘은 정말 우울하고 기운 빠지는 날이었다. 빨리 내일이 왔으면 좋겠다.
                    """,
                "output": "부정"
            }
        ]

        # 예시 일기 데이터

        # 요약 프롬프트/ 실험용 input_diary -> 실제에서는 제거할것
        summary_prompt = summary_prompt_template.format(input_diary=my_diary, diary=my_diary)
        summary_response = llm.invoke(summary_prompt).content
        return summary_response


    def embedding_data(self, my_diary):

        # chroma database 디렉토리 생성
        if not os.path.exists(self.chroma_path):
            os.makedirs(self.chroma_path)
            print(f"{self.chroma_path}가 생성되었습니다")

        # chroma 객체 생성
        vector_store = Chroma(
            collection_name="diary_collection",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=self.chroma_path
        )

        # Chroma에 input, output 데이터 임베딩 저장
        inputs, outputs = [], []
        for example in self.examples:
            vector_store.add_texts([example["input"], example["output"]])
            inputs.append(example["input"])
            outputs.append(example["output"])

        # my_diary를 임베딩하여 유사성 검색 수행
        similar = vector_store.similarity_search(my_diary)

        # 유사 데이터의 인덱스 찾기
        most_similar_index = None
        if similar:
            most_similar = similar[0]
            most_similar_index = inputs.index(most_similar.page_content)

        emotion = outputs[most_similar_index]
        if most_similar_index is not None:
            print(f"\n가장 유사한 입력: {inputs[most_similar_index]}")
            print(f"해당 입력의 감정 분석 결과: {emotion}")

        return emotion

    def save_file(self, my_diary, summary, emotion, excel_path, csv_path):
        csv_data = pd.read_csv(self.file_path + self.csv_file_name, encoding='utf-8')

        new_data = {
            self.user_id : "test",
            self.diary_col : my_diary,
            self.summary_diary : summary,
            self.feedback : "test",
            self.emotion_diary : emotion
        }

        new_data_df = pd.DataFrame([new_data])
        # 데이터프레임에 새로운 데이터 추가
        csv_data = pd.concat([csv_data, new_data_df], ignore_index=True)
        # csv 파일에 저장
        csv_data.to_csv(csv_path, index=False, encoding='utf-8')
        # csv 파일을 excel파일에 덮어쓰기
        csv_data.to_excel(excel_path, index=False)

    def run(self):
        my_diary = """
                오늘은 평소처럼 아침 8시에 출근을 했다. 출근하자마자 책상에 앉아 오늘 처리해야 할 서류들을 살펴보았다. 
                오전에는 주로 대출 관련 상담을 진행했는데, 한 고객이 긴 상담 끝에 대출을 승인받고 매우 기뻐하는 모습을 보니 나도 뿌듯했다. 
                점심시간에는 동료들과 함께 근처 식당에서 비빔밥을 먹으며 가벼운 이야기를 나눴다. 오후에는 은행 시스템 점검이 있어 조금 더 바쁘게 움직여야 했다. 
                한참 일을 하다 보니 오후 4시가 되었고, 마감 업무를 위해 서류 정리를 시작했다. 생각보다 일이 많아 퇴근 시간이 조금 늦어졌지만, 다행히 모든 일이 잘 마무리됐다. 
                퇴근 후 집에 돌아와서는 저녁을 먹고 간단하게 스트레칭을 하며 하루의 피로를 풀었다. 오늘도 무사히 하루를 마친 것에 감사하며 잠들기 전 책을 읽기로 했다.
                """
        self.load_env()
        excel_path, csv_path = self.make_files()
        summary = self.utilize_prompts(my_diary)
        emotion = self.embedding_data(my_diary)
        self.save_file(my_diary, summary, emotion, excel_path, csv_path)