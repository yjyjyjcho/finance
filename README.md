# finance

고로시아 블로그
https://dataplay.tistory.com/5?category=845492

코드_gan으로 주식 매매 예측 
https://colab.research.google.com/drive/1WXG3cohwO6_0mbmB9CdT37cc1jfE2Zon

금융 데이터 from 퀀들(quandl)

퀀들 인스톨 : pip install quandl

https://www.quandl.com/

퀀들 파이썬 사용법 : https://www.quandl.com/tools/python


* local csv 파일 업로드 코드
from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])
    ))
    
    저장 경로 : content 폴더


DB그룹 자산배분투자대회 : https://github.com/yjyjyjcho/BIGDataCampus_project_Remotion-/files/5660539/default.pdf
