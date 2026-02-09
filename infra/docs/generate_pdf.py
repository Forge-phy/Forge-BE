"""
Forge v2.0 AI Module 문서 PDF 생성
깔끔한 템플릿으로 LLM + LSTM 구현 설명
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# 색상 정의
BLUE = colors.HexColor('#4A86E8')
GREEN = colors.HexColor('#76B900')
ORANGE = colors.HexColor('#E69138')
GRAY = colors.HexColor('#666666')
LIGHT_GRAY = colors.HexColor('#F5F5F5')
LIGHT_BLUE = colors.HexColor('#DAE8FC')
LIGHT_GREEN = colors.HexColor('#D5E8D4')
LIGHT_ORANGE = colors.HexColor('#FFE6CC')


def create_styles():
    """스타일 정의"""
    styles = getSampleStyleSheet()

    # 제목 스타일
    styles.add(ParagraphStyle(
        'Title_Custom',
        parent=styles['Title'],
        fontSize=28,
        textColor=colors.HexColor('#333333'),
        spaceAfter=10,
        alignment=TA_CENTER
    ))

    # 부제목
    styles.add(ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=GRAY,
        spaceAfter=30,
        alignment=TA_CENTER
    ))

    # 섹션 제목
    styles.add(ParagraphStyle(
        'Section',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=BLUE,
        spaceBefore=20,
        spaceAfter=10,
        borderColor=BLUE,
        borderWidth=1,
        borderPadding=5
    ))

    # 소제목
    styles.add(ParagraphStyle(
        'SubSection',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#333333'),
        spaceBefore=15,
        spaceAfter=8
    ))

    # 본문
    styles.add(ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=16
    ))

    # 코드
    styles.add(ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontSize=9,
        textColor=colors.HexColor('#333333'),
        backColor=LIGHT_GRAY,
        borderColor=colors.HexColor('#CCCCCC'),
        borderWidth=1,
        borderPadding=8,
        spaceAfter=10
    ))

    # 강조
    styles.add(ParagraphStyle(
        'Highlight',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        backColor=LIGHT_ORANGE,
        borderPadding=10,
        spaceAfter=10
    ))

    return styles


def create_table(data, col_widths=None, header_color=BLUE):
    """테이블 생성"""
    table = Table(data, colWidths=col_widths)

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ])
    table.setStyle(style)
    return table


def generate_pdf():
    """PDF 생성"""
    output_path = '/Users/yangjeong-u/projects/forge/docs/Forge_AI_Modules.pdf'
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = create_styles()
    story = []

    # ==================== 표지 ====================
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("Forge v2.0", styles['Title_Custom']))
    story.append(Paragraph("AI Module Implementation", styles['Title_Custom']))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("LLM + LSTM 기술 구현 상세", styles['Subtitle']))
    story.append(Spacer(1, 2*cm))

    # 핵심 요약 박스
    summary_data = [
        ['Module', 'Role', 'Technology'],
        ['LLM', '자연어 → Isaac Sim 파라미터', 'OpenAI / Qwen'],
        ['LSTM', 'Sim2Real 갭 예측', 'PyTorch LSTM'],
        ['LLM+LSTM', '보고서 자동 생성', 'Pipeline'],
    ]
    story.append(create_table(summary_data, col_widths=[3*cm, 6*cm, 4*cm]))

    story.append(PageBreak())

    # ==================== LLM Section ====================
    story.append(Paragraph("1. LLM Module", styles['Section']))
    story.append(Paragraph("자연어 → Isaac Sim 파라미터 변환", styles['Subtitle']))

    # 왜 LLM이 필요한가?
    story.append(Paragraph("1.1 왜 LLM이 필요한가?", styles['SubSection']))

    story.append(Paragraph(
        "<b>문제 상황:</b> 사용자가 자연어로 시뮬레이션 환경을 요청합니다.",
        styles['Body_Custom']
    ))

    story.append(Paragraph(
        '"로봇 50대가 들어갈 수 있는 대형 창고에서 AGV 5대와 포크리프트 2대를 배치하고, 선반은 4열로 구성해줘"',
        styles['Code_Custom']
    ))

    story.append(Paragraph(
        "<b>규칙 기반(Rule-based)으로는 불가능한 이유:</b>",
        styles['Body_Custom']
    ))

    reason_data = [
        ['요청 표현', '필요한 추론'],
        ['"50대가 들어갈 수 있는"', '창고 크기 계산 (50×100m²=5,000m²→70m×70m)'],
        ['"대형 창고"', '문맥에 따른 크기 조정'],
        ['"AGV 5대와 포크리프트 2대"', '다중 로봇 타입 파싱, 각각 다른 파라미터'],
        ['"선반 4열"', 'count=4, arrangement="row" 매핑'],
    ]
    story.append(create_table(reason_data, col_widths=[5*cm, 8*cm], header_color=ORANGE))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph(
        "<b>LLM의 장점:</b> 자연어 이해 + 추론 능력, 문맥에 따른 유연한 해석, 새로운 표현도 학습 없이 대응 가능",
        styles['Highlight']
    ))

    # 추론 가이드라인
    story.append(Paragraph("1.2 추론 가이드라인", styles['SubSection']))

    story.append(Paragraph(
        "LLM에게 제공하는 추론 규칙 (prompts.py):",
        styles['Body_Custom']
    ))

    guideline_data = [
        ['추론 유형', '규칙', '예시'],
        ['창고 크기', 'N대 × 100m² = 총 면적', '50대 → 5,000m² → 70m×70m'],
        ['로봇 밀도', '최적: 100m² 당 1대', '"여유있게" → 150m² 당 1대'],
        ['선반 배치', 'N열 → count=N', '"4열 선반" → count: 4'],
        ['환경 조건', '키워드 → 파라미터', '"더운 환경" → temp: 35'],
    ]
    story.append(create_table(guideline_data, col_widths=[3*cm, 5*cm, 5*cm]))

    # 민감도 분기
    story.append(Paragraph("1.3 민감도 기반 LLM 분기", styles['SubSection']))

    security_data = [
        ['민감도', '데이터 예시', '사용 LLM'],
        ['PUBLIC', '일반 환경 설정', 'OpenAI (빠름)'],
        ['INTERNAL', '시뮬레이션 결과, 생산량', 'Qwen 로컬 (보안)'],
        ['CONFIDENTIAL', '고객사 정보, 비밀번호', 'Qwen 로컬 (보안)'],
    ]
    story.append(create_table(security_data, col_widths=[3.5*cm, 5*cm, 4.5*cm]))

    story.append(PageBreak())

    # ==================== LSTM Section ====================
    story.append(Paragraph("2. LSTM Module", styles['Section']))
    story.append(Paragraph("Sim2Real 갭 예측", styles['Subtitle']))

    # 왜 LSTM이 필요한가?
    story.append(Paragraph("2.1 왜 LSTM이 필요한가?", styles['SubSection']))

    story.append(Paragraph(
        "<b>Sim2Real Gap (시뮬레이션 ↔ 현실 격차):</b>",
        styles['Body_Custom']
    ))

    story.append(Paragraph(
        "• 시뮬레이션은 이상적 환경<br/>"
        "• 현실에는 온도, 습도, 장비 노후화 등 변수 존재<br/>"
        "• 동일 설정이어도 현장 성능은 10~30% 낮을 수 있음",
        styles['Body_Custom']
    ))

    story.append(Paragraph(
        "<b>LSTM의 장점:</b> 시계열 패턴 학습, 복합 변수 간 상호작용 자동 학습, 신뢰구간 제공",
        styles['Highlight']
    ))

    # 모델 구조
    story.append(Paragraph("2.2 모델 구조 (v3)", styles['SubSection']))

    model_data = [
        ['항목', '값'],
        ['모델 타입', 'LSTM (2 layers)'],
        ['입력 특성', '10개'],
        ['Hidden Size', '64'],
        ['학습 데이터', '10년 (87,600 샘플)'],
        ['Test R²', '0.826'],
    ]
    story.append(create_table(model_data, col_widths=[4*cm, 6*cm], header_color=GREEN))

    # 입력 특성
    story.append(Paragraph("2.3 입력 특성 (10개)", styles['SubSection']))

    feature_data = [
        ['특성', '설명', '영향'],
        ['sim_throughput', '시뮬레이션 처리량', '기준값'],
        ['temperature', '현장 온도 (°C)', '고온 시 성능↓'],
        ['humidity', '현장 습도 (%)', '고습 시 성능↓'],
        ['robot_count', '로봇 대수', '과밀 시 성능↓'],
        ['operation_hours', '연속 가동 시간', '장시간 시 피로↑'],
        ['days_since_maintenance', '정비 후 경과일', '노후화 영향'],
        ['hour', '시간대', '일간 패턴'],
        ['day_of_week', '요일', '주간 패턴'],
        ['is_weekend', '주말 여부', '운영 패턴'],
        ['day_of_year', '연중 일수', '계절성'],
    ]
    story.append(create_table(feature_data, col_widths=[4*cm, 5*cm, 4*cm], header_color=GREEN))

    # 출력
    story.append(Paragraph("2.4 출력 예시", styles['SubSection']))

    story.append(Paragraph(
        '{"sim_throughput": 100, "predicted_real_throughput": 73.26, '
        '"gap_percent": -26.74, "confidence": 0.83, "factors": ["고온", "고습도"]}',
        styles['Code_Custom']
    ))

    story.append(PageBreak())

    # ==================== Pipeline Section ====================
    story.append(Paragraph("3. LSTM → LLM Pipeline", styles['Section']))
    story.append(Paragraph("예측 결과를 자연어 보고서로 자동 변환", styles['Subtitle']))

    story.append(Paragraph("3.1 변환 예시", styles['SubSection']))

    story.append(Paragraph("<b>LSTM 출력 (숫자):</b>", styles['Body_Custom']))
    story.append(Paragraph(
        '{"gap": -15%, "confidence": 0.89, "factors": ["temp", "humidity"]}',
        styles['Code_Custom']
    ))

    story.append(Paragraph("<b>LLM 보고서 (자연어):</b>", styles['Body_Custom']))
    story.append(Paragraph(
        '"현장 적용 시 약 15% 성능 저하가 예상됩니다. (신뢰도 89%)<br/>'
        '주요 원인은 온도와 습도입니다.<br/>'
        '권장: 환기 시스템 점검 후 적용하세요."',
        styles['Highlight']
    ))

    # 핵심 가치
    story.append(Paragraph("3.2 v2.0 핵심 가치", styles['SubSection']))

    value_data = [
        ['가치', '설명'],
        ['전문가 불필요', '예측 결과를 누구나 이해 가능'],
        ['즉시 활용', '의사결정에 바로 활용 가능한 형태'],
        ['권장사항 포함', '다음 단계 액션까지 자동 생성'],
    ]
    story.append(create_table(value_data, col_widths=[4*cm, 9*cm], header_color=colors.HexColor('#9673A6')))

    # 요약
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("요약", styles['Section']))

    summary_final = [
        ['Module', 'Role', 'Why AI?'],
        ['LLM', '자연어 → 파라미터', '규칙으로 불가능한 추론'],
        ['LSTM', '갭 예측', '복합 변수 상호작용 학습'],
        ['Pipeline', '보고서 생성', '비전문가도 이해 가능'],
    ]
    story.append(create_table(summary_final, col_widths=[3*cm, 5*cm, 5*cm]))

    # 빌드
    doc.build(story)
    print(f'✅ PDF 생성 완료: {output_path}')
    return output_path


if __name__ == '__main__':
    generate_pdf()
