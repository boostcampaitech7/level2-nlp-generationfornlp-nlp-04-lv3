import pdfplumber
import pandas as pd
import re
import ast


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




def main():
    input_pdf_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military/military_2017_answer.pdf' # 정답지 pdf 파일
    input_csv_path = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military_2017_korean.csv' # 문제지에서 추출한 데이터 CSV 파일
    output_file = '/data/ephemeral/home/level2-nlp-generationfornlp-nlp-04-lv3/data/military_2017_korean_output.csv' # output CSV 파일
    
    extractor = MilitaryKoreanAnswerExtractor(input_pdf_path)
    extractor.extract_text()
    answers_dict = extractor.extract_answers()
    extractor.add_answers_to_csv(input_csv_path, answers_dict, output_file)


if __name__ == "__main__":
    main()

# answer 파일이 pdf가 아닌 경우 다음을 answers_dict에 대입하면 됩니다.
''' 2014년 answers
answers_dict = {
    '1': 4, '2': 3, '3': 2, '4': 2, '5': 1, '6': 5, '7': 1, '8': 4, '9': 5, '10': 1,
    '11': 3, '12': 1, '13': 4, '14': 1, '15': 5, '16': 3, '17': 5, '18': 2, '19': 4, '20': 3,
    '21': 5, '22': 3, '23': 3, '24': 4, '25': 4, '26': 4, '27': 3, '28': 5, '29': 2, '30': 5,
    '31': 2, '32': 1, '33': 2, '34': 1, '35': 1, '36': 5, '37': 3, '38': 4, '39': 3, '40': 2,
    '41': 2, '42': 5, '43': 2, '44': 2, '45': 3
}'''
''' 2015년 answers
answers_dict = {
  '1': 2, '2': 4, '3': 2, '4': 2, '5': 5, '6': 3, '7': 4, '8': 5, '9': 4, '10': 1,
  '11': 2, '12': 5, '13': 3, '14': 5, '15': 5, '16': 4, '17': 3, '18': 3, '19': 4,
  '20': 5, '21': 2, '22': 2, '23': 4, '24': 2, '25': 2, '26': 4, '27': 4, '28': 4,
  '29': 3, '30': 4, '31': 4, '32': 4, '33': 2, '34': 3, '35': 3, '36': 5, '37': 4, 
  '38': 2, '39': 4, '40': 4, '41': 1, '42': 5, '43': 3, '44': 4, '45': 2
}'''
''' 2016년 answers
answers_dict = {
    '1': 3, '2': 3, '3': 4, '4': 5, '5': 2, '6': 4, '7': 2, '8': 3, '9': 4, '10': 5,
    '11': 5, '12': 1, '13': 5, '14': 5, '15': 2, '16': 2, '17': 3, '18': 3, '19': 4, '20': 1,
    '21': 4, '22': 5, '23': 3, '24': 5, '25': 5, '26': 2, '27': 5, '28': 3, '29': 4, '30': 3,
    '31': 1, '32': 2, '33': 3, '34': 2, '35': 1, '36': 3, '37': 1, '38': 4, '39': 3, '40': 2,
    '41': 1, '42': 5, '43': 4, '44': 4, '45': 4
}'''

