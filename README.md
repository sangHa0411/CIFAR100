# ImageNet

## Goal
  1. ResNet
  2. DenseNet
  3. VIT
  4. Mobile Net & Efficient Net

## Using
  1. PyTorch
  2. Tensorboard
  3. unittest
  4. AutoML

## Git
  * Start
    * 시작 : git init
    * 깃허브 연결 : git remote add origin <github 주소>
  * Branch
    * 브랜치 확인 : git branch
    * 브랜치 생성 : git branch <새 브랜치>
    * 브랜치 이동 : git checkout <이동할 브랜치>
    * 브랜치 로그 확인 : git log --oneline
    * 브랜치 병합
      1. git checkout master
      2. git merge <병합할 브랜치> 
  * Git 활용
    * 상태 확인 : git status
    * 스테이지에 이동 : git add <작업트리에 넣을 파일> 
    * 저장소에 저장 : git commit -m "Message"
    * 되돌리기
      1. git log로 commit Hash 확인
      2. git revert 복사한 Hash
  * 원격저장소 
    * 원격저장소 전체 복제 : git clone 주소 <로컬 디렉토리>
    * 원격저장소에 저장 
      1. git commit -m "Message"
      2. git push 
    * 원격저장소로부터 받기 
      1. 디렉토리 이동
      2. git pull 

---

## Git 활용 구조(경험)

  0. register local ssh public key to github
  1. git clone <repo ssh>
  2. git branch <my branch>
  3. programming in local workspace
  4. git add "file"
  5. git commit -m "message"
  6. git push
  7. Make Pull Request
      1. Pull Requset가 생성이 되면 같으 팀원들에게 Message가 전달된다. (구체적으로 메시지를 작성해야)
      2. Message를 받은 팀원들으 Code Review를 하고 Merge여부를 결정한다.
      3. Merge 된 code를 git pull 명령을 통해서 동기화를 진행한다.
      4. 목적을 달성한 branch는 삭제한다.
    


