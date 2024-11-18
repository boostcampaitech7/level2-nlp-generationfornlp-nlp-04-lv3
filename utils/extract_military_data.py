import pdfplumber
import pandas as pd
import re
import os
import ast

# 1. DataExtractor Class
class MilitaryKoreanDataExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = ''
    
    def extract_text(self):
        # PDF 파일 열기
        with pdfplumber.open(self.pdf_path) as pdf:
            # 모든 페이지에서 텍스트 추출
            for page in pdf.pages[2:-1]:
                self.text += page.extract_text()


    @staticmethod
    def replace_unwanted_sentences(text):
        text = re.sub(r'다음은 수업 중 학생의 발표이다\. 물음에 답하시오\.', '다음 글을 읽고 물음에 답하시오.', text)
        text = re.sub(r'다음은 강연의 일부이다\. 물음에 답하시오\.', '다음 글을 읽고 물음에 답하시오.', text)
        text = re.sub(r'(가)는 학교에 게시할 안내문을 작성하기 위한 학생회 학생들의 회의이고, (나)는 이를 바탕으로 작\n성한 글의 초고이다\. 물음에 답하시오\.', '다음 글을 읽고 물음에 답하시오.', text)
        text = re.sub(r'(가)는 학생들의 토의이고, (나)는 이를 바탕으로 작성한 글의 초고이다. 물음에 답하시오.', '다음 글을 읽고 물음에 답하시오.', text)
        text = re.sub(r'다음은 협상의 일부이다. 물음에 답하시오.', '다음 글을 읽고 물음에 답하시오.', text)
        text = re.sub(r'다음은 학생이 수업 시간에 한 발표이다. 물음에 답하시오.', '다음 글을 읽고 물음에 답하시오.', text)
        text = re.sub(r'다음을 읽고 물음에 답하시오.', '다음 글을 읽고 물음에 답하시오.', text)
        text = re.sub(r'다음은 학급 임원 3명이 한 회의의 일부이다. 물음에 답하시오.', '다음 글을 읽고 물음에 답하시오.', text)
        return text

    # 지문 추출
    def extract_paragraphs(self, text):
        # "다음 글을 읽고 물음에 답하시오."와 그 다음 문제 부분 사이에 있는 텍스트를 지문으로 추출
        pattern = r'다음 글을 읽고 물음에 답하시오\.\s*(.*?)\s*(?=\d+\.)'  # 문제 번호(예: 1., 2.) 전에 위치한 텍스트 추출
        paragraphs = re.findall(pattern, text, re.DOTALL)  # re.DOTALL을 사용하여 줄바꿈을 포함한 모든 문자 처리
        # 각 지문에 인덱스를 붙이기
        paragraph_data = [{'index': i, 'passage': paragraph.strip()} for i, paragraph in enumerate(paragraphs)]
        
        return paragraph_data


    # text에서 지문을 하나씩 찾아서 제거하기
    def remove_paragraphs_from_text(self, text):
        paragraph_data = self.extract_paragraphs(text)
        for paragraph in paragraph_data:
            text = text.replace(paragraph['passage'], '')  # 지문에 해당하는 텍스트 삭제
        return text


    # 문제 추출
    def extract_problems(self, text):
        problem_text = self.remove_paragraphs_from_text(text)
        pattern = r'(\d+\..*?)(?=\d+\.|$)'  # 문제 번호를 기준으로 묶음
        problems = re.findall(pattern, problem_text, re.DOTALL)
        all_problems = []
        
        for problem in problems:
            # 문제 부분과 선택지를 분리
            # 선택지 번호가 있는 부분을 기준으로 문제와 선택지를 나눔
            question_match = re.match(r'(\d+\..*?)(?=\s*①)', problem)
            if question_match:
                question = question_match.group(1).strip()  # 질문 부분 추출
                # [A], [B]와 같은 텍스트가 포함된 문제를 제외
                if re.search(r'\[A\]|\[B\]|㉮|㉯|㉰|㉱|㉲|㉳', question):
                    continue  # 문제를 건너뛰고 다음 문제로 넘어갑니다.
                
                # 선택지 부분을 추출
                choices = re.findall(r'[①②③④⑤].*', problem)
                choices = [choice.strip() for choice in choices]  # 선택지 리스트
                all_problems.append({
                    'question': question,
                    'choices': choices,
                    'answer': ''  # 정답 부분은 나중에 채울 수 있습니다
                })  
        
        return all_problems


    # 지문-문제 매칭
    def match_paragraph_question(self, text):
        pattern = r'\[(\d+)∼(\d+)\] 다음 글을 읽고 물음에 답하시오\.' # "[숫자∼숫자] 다음 글을 읽고 물음에 답하시오." 형태를 찾는 정규 표현식
        ranges = re.findall(pattern, text)  # 해당 패턴에 맞는 문장들을 찾음
        return ranges

    @staticmethod
    def remove_unwanted_patterns(text):
        # '국 어 영 역 공 통'과 '공 통 국 어 영 역' 패턴 제거
        text = re.sub(r'\d+ 국 어 영 역 공 통|공 통 국 어 영 역 \d+|\d+ 국 어 영 역 A 형|A 형 국 어 영 역 \d+', '', text)
        return text

    # <그림>이 포함된 데이터 제거
    def remove_paragraphs_with_images(df):
        # Remove rows where the 'paragraph' contains '<그림>, <표>'
        df = df[~df['paragraph'].str.contains('<그림>', na=False)]
        df = df[~df['paragraph'].str.contains('<표>', na=False)]
        return df

    def extract_and_save_to_csv(self, output_csv_path):
        text = self.replace_unwanted_sentences(self.text)
        matching_ranges = self.match_paragraph_question(text)

        data = []
        paragraph_data = self.extract_paragraphs(text)
        problem_data = self.extract_problems(text)

        for idx, (start, end) in enumerate(matching_ranges):
            paragraph = paragraph_data[idx]['passage']
            paragraph = self.remove_unwanted_patterns(paragraph)
            problems_in_range = [problem for problem in problem_data if int(start) <= int(problem['question'].split('.')[0]) <= int(end)]
            for problem in problems_in_range:
                question = problem['question']
                choices = problem['choices']
                question = self.remove_unwanted_patterns(question)
                choices = [self.remove_unwanted_patterns(choice) for choice in choices]

                problem_number = problem['question'].split('.')[0]
                data.append({
                    'id': f'generation-for-nlp-military-2012-korean-{problem_number}',
                    'paragraph': paragraph,
                    'problems': {'question': question, 'choices': choices, 'answer': problem['answer']},
                    'question_plus': '',
                    'is_exist': 1
                })

        df = pd.DataFrame(data)
        df = self.remove_paragraphs_with_images(df)
        df.to_csv(output_csv_path, index=False)
        print("사관학교 국어영역이 CSV 파일로 변환되었습니다.")


# 2. AnswerExtractor Class
class MilitaryKoreanAnswerExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.ans = ''

    def extract_text(self):
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                self.ans += page.extract_text()
    
    @staticmethod
    def map_answer_to_number(answer):
        answer_map = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5}
        return answer_map.get(answer, None)  # 기본값으로 None 반환

    
    # 정답을 추출하기 위한 함수
    def extract_answers(self):
        # 문항 번호와 정답을 추출하는 정규 표현식
        question_pattern = r'문항\s*(\d+(?:\s+\d+)*)\s*번호\s*정답\s*((?:①|②|③|④|⑤)(?:\s+(?:①|②|③|④|⑤))*)'
        # 여러 구간에 대한 매칭을 처리하기 위해 findall 사용
        matches = re.findall(question_pattern, self.text)
        
        # 문항 번호와 정답을 매칭하여 딕셔너리로 저장
        answer_dict = {}
        for match in matches:
            questions = match[0].split()
            answers = match[1].split()

            # 문항 번호와 정답을 매칭하여 딕셔너리에 저장
            for question, answer in zip(questions, answers):  
                answer_dict[question] = self.map_answer_to_number(answer)
        return answer_dict



    # 문제 CSV 파일에 정답을 넣는 함수
    def add_answers_to_csv(self, csv_file, answers_dict, output_file):
        # CSV 파일 읽기
        df = pd.read_csv(csv_file)
        
        # 각 문제에 대해 정답을 채워넣기
        for index, row in df.iterrows():
            # id에서 문제 번호 추출
            problem_number = row['id'].split('-')[-1]  # id에서 'generation-for-nlp-military-년도-korean-1'에서 1 추출
            
            # 문제 번호에 해당하는 정답을 answers_dict에서 찾아서 'answer' 열에 추가
            if problem_number in answers_dict:
                print(problem_number, '번 정답:')
                problem_data = ast.literal_eval(row['problems'])
                problem_data['answer'] = answers_dict[problem_number]
                print(problem_data['answer'])
                df.at[index, 'problems'] = str(problem_data)
        
        # 수정된 DataFrame을 새로운 CSV 파일로 저장
        df.to_csv(output_file, index=False)
        print(f"정답이 추가된 CSV 파일이 '{output_file}'로 저장되었습니다.")


# 3. CSVMerger Class
class MilitaryKoreanCSVMerger:
    def __init__(self, input_folder, output_file):
        self.input_folder = input_folder
        self.output_file = output_file

    def merge_csv_files(self):
        # 모든 CSV 파일의 경로를 가져옵니다.
        files = [f for f in os.listdir(self.input_folder) if f.endswith('.csv')]
        
        # 파일들을 읽어 DataFrame 목록에 저장합니다.
        df_list = []
        for file in files:
            file_path = os.path.join(self.input_folder, file)
            df = pd.read_csv(file_path)
            df_list.append(df)
        
        # 모든 DataFrame을 병합합니다.
        merged_df = pd.concat(df_list, ignore_index=True)
        # 병합된 DataFrame을 새로운 CSV 파일로 저장합니다.
        merged_df.to_csv(self.output_file, index=False)
        print(f"CSV 파일들이 병합되어 '{self.output_file}'로 저장되었습니다.")



def main():
    # MilitaryKoreanDataExtractor - PDF to CSV
    input_pdf_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military_2025_korean.pdf' # 원본 pdf 파일 경로
    output_csv_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military_2025_korean.csv' # output CSV 파일 경로
    extractor = MilitaryKoreanDataExtractor(input_pdf_path)
    extractor.extract_text()
    extractor.extract_and_save_to_csv(output_csv_path)

    # MilitaryKoreanAnswerExtractor - Answer Extraction and CSV update
    input_pdf_path_answers = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military/military_2025_answer.pdf' # 정답지 pdf 파일
    input_csv_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military_2025_korean.csv' # 문제지에서 추출한 데이터 CSV 파일
    output_file = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military_2025_korean_output.csv' # output CSV 파일
    answer_extractor = MilitaryKoreanAnswerExtractor(input_pdf_path_answers)
    answer_extractor.extract_text()
    answers_dict = answer_extractor.extract_answers()
    answer_extractor.add_answers_to_csv(input_csv_path, answers_dict, output_file)

    # MilitaryKoreanCSVMerger - Merge all CSV files
    input_folder = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military'
    output_file = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military/military_2025-2014_korean.csv'
    merger = MilitaryKoreanCSVMerger(input_folder, output_file)
    merger.merge_csv_files()

if __name__ == "__main__":
    main()