# RF Capacitor Surrogate Designer (Prototype)

**격자형 RF 캐패시터 설계를 빠르게 실험·스크리닝하는 연구용 도구**  
PDK(유전율·두께·유닛사이즈) 기반으로 구조를 정의하고, RLC 교사 모델과 MLP 서로게이트를 활용하여  
EM 시뮬레이션 대비 수백 배 빠른 주파수 응답 예측과 기초적 역설계 기능을 제공합니다.  

---

## ✨ Features
- **PDK 기반 정의**  
  - 유전율, 두께, 유닛 사이즈 등 공정 파라미터 반영  
- **격자형 설계**  
  - 격자 크기 (Nx, Ny) 입력 → 정전용량 추정  
- **Wideband Dataset 생성**  
  - RLC 교사 모델로 전 주파수대역 S-파라미터 곡선 생성  
- **MLP Surrogate 학습**  
  - 입력 격자 파라미터 → 전대역 S11/S21 응답 매핑  
  - CUDA / DirectML / CPU 자동 선택 (Cross-platform)  
- **즉시 추론**  
  - 학습된 모델로 주파수 응답 곡선 즉시 예측  
  - Forward EM 대비 수백 배 빠른 대량 스크리닝  
- **Inverse Design (옵션)**  
  - 목표 응답 스펙을 주면 격자 탐색·최적화로 구조 제안  
- **CLI & GUI 제공**  
  - 명령어 기반 실행 및 간단한 GUI 인터페이스  
- **결과 저장·내보내기**  

---

## 🧩 Workflow
1. **Define PDK** → Material, thickness, unit size  
2. **Set grid (Nx, Ny)**  
3. **Generate S-parameters (teacher RLC model)**  
4. **Train MLP surrogate**  
5. **Predict full-band S11/S21 instantly**  
6. (Optional) **Inverse design via optimization**  
7. **Validate with EM simulation if needed**

---

## 📂 Project Structure
