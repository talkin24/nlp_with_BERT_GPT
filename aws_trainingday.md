# AWS Training day

## AWS VPC
- 사용자 정의 가상 네트워크 공간(논리적인 가상의 컴포넌트)
- 완전한 네트워크 제어 가능
  - IP 범위
  - Subnet
  - Route table
  - Network ACL, 보안그룹
  - 다양한 게이트웨이

- VPC 내의 모든 EC2 인스턴스는 사설 IP가 부여됨
- 개별 인스턴스에 공인 IP 할당 가능

- 생성 순서
  1. 리전 선택 및 IP 주소 범위 설정
    - 4개 리전
    - CIDR 블록설정
    - VPC CIDR은 생성 후 변경 불가능
    - 네트워크 범위는 /16 ~ /28
      - 16bit 기준 65,000개 가량 블록 설정
      - 블록 설정 후 그 중 실제 사용하는 서브넷에 IP 매핑
  2. 인터넷 게이트웨이 생성
    - IPv4 0.0.0.0/0의 목적지는 인터넷 게이트 웨이
    - 
  3. 가용 영역 내에 필요한 서브넷 정의
  4. 라우팅 구성
  5. VPC로 송수신 되는 트래픽 제어

- EC2 인스턴스를 위한 공인 IP 주소
  - 공인 주소 자동 할당
  - Elastic IP Addresses(EIP)
    - AWS가 가지고 있는 BYOIP Pool에서 제공(단, public IP는 미사용시 탄력적으로 변경되게 됨)
    - Subnet에 Private IP, Public IP가 따로 발급됨


## EC2
- 타입 400개 정도로 굉장히 다양
  - C-type: 컴퓨팅 집약적인 워크로드
  - R, X-type: 메모리 집약적인 워크로드
    - VM 당 1~2TB
    - ex. Spark cluster?
  - P, F, G-types: GPU 사용
  - H, I, D-types: 스토리지 집약적인 워크로드
    - ex. HDFS, Kafka, NoSQLs

- M5d.xlarge
  - M/5/d.xlarge
  - 인스턴스패밀리/세대/추가기능.인스턴스사이즈
  - 인스턴스 세대가 높은데도 더 가격이 쌀 수 있음

- 어떤 워크로드?
  - GPU 사용 여부
  - 대용량 저장 공간
  - 디스크 I/O
    - 캐쉬여부
      - 캐쉬를 위한 메모리 고려
    - 밸런스고려

- 구매 옵션
  - 온디맨드
  - 예약인스턴스
  - 세이빙플랜
  - 스팟인스턴스(베타, 먼저 사용하는 사람이 임자)

## DB
- 데이터 트렌드
  - 데이터 증가
  - Micro-services
  - DevOps

- 완전 관리형 데이터베이스
  - 스키마 디자인/쿼리구성 등만 하면 나머지는 aws가

- Amazon RDS
  - 많이 쓰이는 관계형 DB 다 사용가능
  - 모니터링: CloudWatch
  - 경보
  - 성능 개선 도우미(Performance Insights)
    - 로드를 유발하는 SQL문과 그 이유를 알 수 있음
  - 가용성 및 내구성
    - 다중 AZ 배포(동기식 복제) - 높은 내구성
    - 읽기전용 복제본(비동기식 복제) - 높은 가용성
    - 2가지 동시에 사용하는 워크로드 추천
  - 자동 백업
    - 스냅샷 - 증분 식 백업
  - 확장성
    - 다양한 CPU/메모리 옵션을 가진 데이터베이스 인스턴스 제공
  - 빠른 성능과 보안
    - VPC를 통한 네트워크 격리
    - IAM 기반 리소스 수준 권한 제어

- Amazon Aurora
  - 완전 관리형 데이터 베이스
  - MySQL, PostgreSQL만 호환
  - 최대 15개의 읽기 전용 복제본
  - 확장, 분산형 아키텍처
  - 데이터 복제 방식에서 RDS와 차이
  - 빠른 계정 간 데이터베이스 복제
    - 프로덕변 DB를 테스트 계정에 생성하여 테스트
  - DMS(데이터베이스 마이그레이션 서비스)
    - 동종 및 이기종 데이터 복제 지원


## Storage
- 스토리지 타입
  - 블록 스토리지
    - 데이터를 블록으로 나누어 저장
    - SAN
- 파일 스토리지
  - 디렉토리 구조
  - NAS
- 오브젝트 스토리지
  - API 호출을 통해 접근

- Amazon EBS
  - 블록 스토리지
  - EC2에서 사용하도록 설계
  - 영구 지속 블록 스토리지
  - 볼륨 타입
    - SSD
      - gp2/gp3: General-purpose SSD/NoSQL
      - io1/io2: Provisioned IOPS SSD/RDB - I/O가 많은 DB 애플리케이션
    - HDD
      - st1: Throughput-optimized HDD/빅데이터 분석
      - sc1: Cold HDD/ 파일, 미디어
    - 선택 방법
      - IOPS > 80,000? /Throughput?
      - Latency
      - 비용 vs 성능

  - EBS Multi-Attach
    - EBS가 여러 instance로 attach

- Amazon EFS
  - 파일 스토리지
  - 리눅스 기반 공유 스토리지
  - 비용 효과적
  - 다양한 서비스와 호환
    - Compute(ECS, EKS, Fargate...)
    - Automation(Lambda, Auto Scaling)
    - Machine Learning(SageMaker)

- Amazon FSx
  - 파일스토리지
  - 윈도우 기반 공유 스토리지(윈도우 파일시스템과 호환)
  - EFS와 유사

- Amazon S3
  - 무제한에 가까운 용량
  - 데이터 레이크 구축
  - HTTP 통신을 통해 읽고 씀. 따라서 서버에 직접 붙이기는 어려움
  - 3곳 이상의 분리된 가용 영역에 저장
  - 접근 빈도에 따라 life cycle 관리 가능(스토리지 클래스 자동 변경)
  - 스토리지 비용은 계속해서 감소 중
  - 클래스도 다양화
  - 