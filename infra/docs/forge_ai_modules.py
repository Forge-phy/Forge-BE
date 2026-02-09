"""
Forge v2.0 AI 모듈 프레젠테이션 생성 스크립트
LLM + LSTM 슬라이드 PPTX 생성
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import nsmap
from pptx.oxml import parse_xml

# 색상 정의
BLUE = RGBColor(74, 134, 232)       # LLM 메인
GREEN = RGBColor(118, 185, 0)       # LSTM 메인 (NVIDIA)
LIGHT_BLUE = RGBColor(218, 232, 252)
LIGHT_GREEN = RGBColor(213, 232, 212)
YELLOW = RGBColor(255, 242, 204)
RED = RGBColor(248, 206, 204)
PURPLE = RGBColor(225, 213, 231)
GRAY = RGBColor(102, 102, 102)
WHITE = RGBColor(255, 255, 255)
BLACK = RGBColor(51, 51, 51)


def add_title_slide(prs, title, subtitle):
    """타이틀 슬라이드 추가"""
    slide_layout = prs.slide_layouts[6]  # 빈 슬라이드
    slide = prs.slides.add_slide(slide_layout)

    # 배경색
    background = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
    )
    background.fill.solid()
    background.fill.fore_color.rgb = RGBColor(45, 45, 45)
    background.line.fill.background()

    # 제목
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(1))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    # 부제목
    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.5), Inches(9), Inches(0.5))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(20)
    p.font.color.rgb = RGBColor(180, 180, 180)
    p.alignment = PP_ALIGN.CENTER

    return slide


def add_llm_slide(prs):
    """LLM 모듈 슬라이드"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    # 제목 영역
    title_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.2)
    )
    title_shape.fill.solid()
    title_shape.fill.fore_color.rgb = BLUE
    title_shape.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "LLM Module"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE

    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.8), Inches(9), Inches(0.4))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = "OpenAI + Qwen 하이브리드 (민감도 기반 분기)"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(200, 220, 255)

    y_pos = 1.4

    # 입력 섹션
    input_label = slide.shapes.add_textbox(Inches(0.5), Inches(y_pos), Inches(1), Inches(0.3))
    tf = input_label.text_frame
    p = tf.paragraphs[0]
    p.text = "입력"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLACK

    input_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos + 0.3), Inches(9), Inches(0.6)
    )
    input_box.fill.solid()
    input_box.fill.fore_color.rgb = YELLOW
    input_box.line.color.rgb = RGBColor(214, 182, 86)

    input_text = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.4), Inches(8.6), Inches(0.4))
    tf = input_text.text_frame
    p = tf.paragraphs[0]
    p.text = '자연어 요청: "30m x 20m 창고에 AGV 3대, 선반 2열"'
    p.font.size = Pt(14)
    p.alignment = PP_ALIGN.CENTER

    y_pos += 1.1

    # 민감도 분류기 섹션
    classifier_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos), Inches(9), Inches(1.3)
    )
    classifier_box.fill.solid()
    classifier_box.fill.fore_color.rgb = PURPLE
    classifier_box.line.color.rgb = RGBColor(150, 115, 166)

    classifier_title = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.1), Inches(8.6), Inches(0.3))
    tf = classifier_title.text_frame
    p = tf.paragraphs[0]
    p.text = "민감도 분류기 (SensitivityClassifier)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    flow_text = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.4), Inches(8.6), Inches(0.3))
    tf = flow_text.text_frame
    p = tf.paragraphs[0]
    p.text = "1. 도메인 판단 → 2. 패턴 매칭 → 3. 키워드 점수 → 4. 레벨 결정"
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.CENTER

    # 민감도 레벨 박스들
    levels = [
        ("PUBLIC", LIGHT_GREEN, 0.7),
        ("INTERNAL", YELLOW, 3.5),
        ("CONFIDENTIAL", RED, 6.3)
    ]
    for name, color, x in levels:
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y_pos + 0.8), Inches(2.5), Inches(0.4)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = color
        txt = slide.shapes.add_textbox(Inches(x), Inches(y_pos + 0.85), Inches(2.5), Inches(0.3))
        tf = txt.text_frame
        p = tf.paragraphs[0]
        p.text = name
        p.font.size = Pt(12)
        p.font.bold = True
        p.alignment = PP_ALIGN.CENTER

    y_pos += 1.5

    # 프로바이더 섹션
    provider_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos), Inches(9), Inches(1.1)
    )
    provider_box.fill.solid()
    provider_box.fill.fore_color.rgb = LIGHT_BLUE
    provider_box.line.color.rgb = RGBColor(108, 142, 191)

    provider_title = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.1), Inches(8.6), Inches(0.3))
    tf = provider_title.text_frame
    p = tf.paragraphs[0]
    p.text = "LLM 프로바이더 선택"
    p.font.size = Pt(14)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    # OpenAI 박스
    openai_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(y_pos + 0.5), Inches(4), Inches(0.5)
    )
    openai_box.fill.solid()
    openai_box.fill.fore_color.rgb = LIGHT_GREEN
    openai_txt = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.55), Inches(4), Inches(0.4))
    tf = openai_txt.text_frame
    p = tf.paragraphs[0]
    p.text = "OpenAI (gpt-4o-mini) - 빠름, 외부 API"
    p.font.size = Pt(11)
    p.alignment = PP_ALIGN.CENTER

    # Qwen 박스
    qwen_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.3), Inches(y_pos + 0.5), Inches(4), Inches(0.5)
    )
    qwen_box.fill.solid()
    qwen_box.fill.fore_color.rgb = RED
    qwen_txt = slide.shapes.add_textbox(Inches(5.3), Inches(y_pos + 0.55), Inches(4), Inches(0.4))
    tf = qwen_txt.text_frame
    p = tf.paragraphs[0]
    p.text = "Qwen 로컬 (qwen2.5:7b) - 보안, 데이터 보호"
    p.font.size = Pt(11)
    p.alignment = PP_ALIGN.CENTER

    y_pos += 1.3

    # 주요 기능 섹션
    func_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos), Inches(9), Inches(1.4)
    )
    func_box.fill.solid()
    func_box.fill.fore_color.rgb = LIGHT_GREEN
    func_box.line.color.rgb = RGBColor(130, 179, 102)

    func_title = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.1), Inches(8.6), Inches(0.3))
    tf = func_title.text_frame
    p = tf.paragraphs[0]
    p.text = "주요 기능"
    p.font.size = Pt(14)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    functions = [
        ("parse_environment_request()", "자연어 → Isaac Sim 파라미터", 0.7, y_pos + 0.5),
        ("modify_parameters()", "대화형 파라미터 수정", 5, y_pos + 0.5),
        ("analyze_simulation_result()", "시뮬레이션 결과 분석", 0.7, y_pos + 0.95),
        ("generate_report_from_lstm()", "LSTM → 자연어 보고서", 5, y_pos + 0.95),
    ]

    for func_name, desc, x, y in functions:
        fbox = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(4), Inches(0.4)
        )
        fbox.fill.solid()
        fbox.fill.fore_color.rgb = WHITE
        ftxt = slide.shapes.add_textbox(Inches(x + 0.1), Inches(y + 0.05), Inches(3.8), Inches(0.35))
        tf = ftxt.text_frame
        p = tf.paragraphs[0]
        p.text = f"{func_name}\n{desc}"
        p.font.size = Pt(9)

    y_pos += 1.6

    # 출력 섹션
    output_label = slide.shapes.add_textbox(Inches(0.5), Inches(y_pos), Inches(1), Inches(0.3))
    tf = output_label.text_frame
    p = tf.paragraphs[0]
    p.text = "출력"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLACK

    output_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos + 0.3), Inches(9), Inches(0.6)
    )
    output_box.fill.solid()
    output_box.fill.fore_color.rgb = LIGHT_GREEN
    output_box.line.color.rgb = RGBColor(130, 179, 102)

    output_text = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.4), Inches(8.6), Inches(0.4))
    tf = output_text.text_frame
    p = tf.paragraphs[0]
    p.text = 'Isaac Sim 파라미터 (JSON): {environment: {...}, robots: [...], simulation: {...}}'
    p.font.size = Pt(12)
    p.alignment = PP_ALIGN.CENTER

    return slide


def add_lstm_slide(prs):
    """LSTM 모듈 슬라이드"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    # 제목 영역
    title_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.2)
    )
    title_shape.fill.solid()
    title_shape.fill.fore_color.rgb = GREEN
    title_shape.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "LSTM Module"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE

    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.8), Inches(9), Inches(0.4))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Sim2Real 갭 예측 (PyTorch MLP/LSTM/GRU)"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(200, 255, 200)

    y_pos = 1.4

    # 입력 특성 섹션
    input_label = slide.shapes.add_textbox(Inches(0.5), Inches(y_pos), Inches(2), Inches(0.3))
    tf = input_label.text_frame
    p = tf.paragraphs[0]
    p.text = "입력 특성 (6개)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLACK

    input_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos + 0.3), Inches(9), Inches(0.9)
    )
    input_box.fill.solid()
    input_box.fill.fore_color.rgb = YELLOW
    input_box.line.color.rgb = RGBColor(214, 182, 86)

    features = [
        ("sim_throughput", 0.6), ("temperature", 2.1), ("humidity", 3.6),
        ("robot_count", 5.1), ("operation_hours", 6.6), ("warehouse_size", 8.1)
    ]
    for feat, x in features:
        fbox = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(y_pos + 0.45), Inches(1.4), Inches(0.25)
        )
        fbox.fill.solid()
        fbox.fill.fore_color.rgb = WHITE
        ftxt = slide.shapes.add_textbox(Inches(x), Inches(y_pos + 0.47), Inches(1.4), Inches(0.2))
        tf = ftxt.text_frame
        p = tf.paragraphs[0]
        p.text = feat
        p.font.size = Pt(9)
        p.alignment = PP_ALIGN.CENTER

    # 두 번째 줄
    features2 = [
        ("robot_count", 0.6), ("operation_hours", 2.1), ("warehouse_size", 3.6)
    ]
    for feat, x in features2:
        fbox = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(y_pos + 0.8), Inches(1.4), Inches(0.25)
        )
        fbox.fill.solid()
        fbox.fill.fore_color.rgb = WHITE
        ftxt = slide.shapes.add_textbox(Inches(x), Inches(y_pos + 0.82), Inches(1.4), Inches(0.2))
        tf = ftxt.text_frame
        p = tf.paragraphs[0]
        p.text = feat
        p.font.size = Pt(9)
        p.alignment = PP_ALIGN.CENTER

    y_pos += 1.4

    # 모델 아키텍처 섹션
    model_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos), Inches(9), Inches(1.8)
    )
    model_box.fill.solid()
    model_box.fill.fore_color.rgb = LIGHT_BLUE
    model_box.line.color.rgb = RGBColor(108, 142, 191)

    model_title = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.1), Inches(8.6), Inches(0.3))
    tf = model_title.text_frame
    p = tf.paragraphs[0]
    p.text = "모델 아키텍처 (Sim2RealMLP)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    # 레이어들
    layers = [
        ("Linear(6→64)\n+ ReLU + BN", 0.7),
        ("Linear(64→32)\n+ ReLU + BN", 2.9),
        ("Linear(32→16)\n+ ReLU", 5.1),
        ("Linear(16→1)", 7.3)
    ]
    for layer, x in layers:
        lbox = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y_pos + 0.5), Inches(1.8), Inches(0.7)
        )
        lbox.fill.solid()
        lbox.fill.fore_color.rgb = WHITE
        ltxt = slide.shapes.add_textbox(Inches(x), Inches(y_pos + 0.55), Inches(1.8), Inches(0.6))
        tf = ltxt.text_frame
        p = tf.paragraphs[0]
        p.text = layer
        p.font.size = Pt(10)
        p.alignment = PP_ALIGN.CENTER

    # 화살표 텍스트
    for x in [2.5, 4.7, 6.9]:
        arr = slide.shapes.add_textbox(Inches(x), Inches(y_pos + 0.75), Inches(0.4), Inches(0.3))
        tf = arr.text_frame
        p = tf.paragraphs[0]
        p.text = "→"
        p.font.size = Pt(16)
        p.font.bold = True
        p.alignment = PP_ALIGN.CENTER

    # 모델 옵션
    opt_txt = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 1.3), Inches(8.6), Inches(0.25))
    tf = opt_txt.text_frame
    p = tf.paragraphs[0]
    p.text = "모델 선택: MLP (기본) | LSTM | GRU     |     Dropout: 0.2     |     파라미터: ~5,000개 (경량)"
    p.font.size = Pt(10)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.CENTER

    y_pos += 2.0

    # 영향 요인 분석 섹션
    factor_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos), Inches(9), Inches(0.9)
    )
    factor_box.fill.solid()
    factor_box.fill.fore_color.rgb = PURPLE
    factor_box.line.color.rgb = RGBColor(150, 115, 166)

    factor_title = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.1), Inches(8.6), Inches(0.3))
    tf = factor_title.text_frame
    p = tf.paragraphs[0]
    p.text = "영향 요인 분석 (_analyze_factors)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    factors = [
        ("고온 (>30°C)", RED, 0.7),
        ("고습도 (>70%)", RED, 2.9),
        ("로봇혼잡 (>6대)", YELLOW, 5.1),
        ("장시간가동 (>8h)", YELLOW, 7.3)
    ]
    for factor, color, x in factors:
        fbox = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y_pos + 0.45), Inches(1.8), Inches(0.35)
        )
        fbox.fill.solid()
        fbox.fill.fore_color.rgb = color
        ftxt = slide.shapes.add_textbox(Inches(x), Inches(y_pos + 0.5), Inches(1.8), Inches(0.25))
        tf = ftxt.text_frame
        p = tf.paragraphs[0]
        p.text = factor
        p.font.size = Pt(10)
        p.alignment = PP_ALIGN.CENTER

    y_pos += 1.1

    # 출력 섹션
    output_label = slide.shapes.add_textbox(Inches(0.5), Inches(y_pos), Inches(2.5), Inches(0.3))
    tf = output_label.text_frame
    p = tf.paragraphs[0]
    p.text = "출력 (PredictionResult)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = BLACK

    output_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos + 0.3), Inches(9), Inches(1.3)
    )
    output_box.fill.solid()
    output_box.fill.fore_color.rgb = LIGHT_GREEN
    output_box.line.color.rgb = RGBColor(130, 179, 102)

    outputs = [
        "sim_throughput: 100 개/h",
        "predicted_real_throughput: 85 개/h  ← 핵심 예측값",
        "gap_percent: -15%",
        "confidence: 0.89",
        "confidence_interval: [83, 87]",
        'factors: ["고온", "고습도"]'
    ]

    for i, out in enumerate(outputs):
        y = y_pos + 0.4 + (i * 0.18)
        x = 0.7 if i < 3 else 5
        if i >= 3:
            y = y_pos + 0.4 + ((i - 3) * 0.18)

        otxt = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(4), Inches(0.2))
        tf = otxt.text_frame
        p = tf.paragraphs[0]
        p.text = out
        p.font.size = Pt(11)
        if "predicted_real" in out:
            p.font.bold = True
            p.font.color.rgb = RGBColor(184, 84, 80)

    return slide


def add_pipeline_slide(prs):
    """LSTM → LLM 파이프라인 슬라이드"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    # 제목 영역
    title_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.2)
    )
    title_shape.fill.solid()
    title_shape.fill.fore_color.rgb = RGBColor(150, 115, 166)
    title_shape.line.fill.background()

    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "LSTM → LLM 파이프라인"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE

    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.8), Inches(9), Inches(0.4))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = "예측 결과를 경영진이 이해할 수 있는 보고서로 자동 변환"
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(230, 220, 240)

    y_pos = 1.6

    # LSTM 출력 박스
    lstm_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos), Inches(4), Inches(2)
    )
    lstm_box.fill.solid()
    lstm_box.fill.fore_color.rgb = LIGHT_GREEN

    lstm_title = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.1), Inches(3.6), Inches(0.3))
    tf = lstm_title.text_frame
    p = tf.paragraphs[0]
    p.text = "LSTM 출력 (숫자)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    lstm_content = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.5), Inches(3.6), Inches(1.4))
    tf = lstm_content.text_frame
    p = tf.paragraphs[0]
    p.text = """{
  "gap": 15%,
  "confidence": 0.89,
  "factors": ["temp", "humidity"]
}"""
    p.font.size = Pt(11)
    p.font.name = "Courier New"

    # 화살표
    arrow_txt = slide.shapes.add_textbox(Inches(4.5), Inches(y_pos + 0.8), Inches(1), Inches(0.5))
    tf = arrow_txt.text_frame
    p = tf.paragraphs[0]
    p.text = "→"
    p.font.size = Pt(48)
    p.font.bold = True
    p.font.color.rgb = RGBColor(150, 115, 166)
    p.alignment = PP_ALIGN.CENTER

    # LLM 출력 박스
    llm_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.5), Inches(y_pos), Inches(4), Inches(2)
    )
    llm_box.fill.solid()
    llm_box.fill.fore_color.rgb = LIGHT_BLUE

    llm_title = slide.shapes.add_textbox(Inches(5.7), Inches(y_pos + 0.1), Inches(3.6), Inches(0.3))
    tf = llm_title.text_frame
    p = tf.paragraphs[0]
    p.text = "LLM 보고서 (자연어)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    llm_content = slide.shapes.add_textbox(Inches(5.7), Inches(y_pos + 0.5), Inches(3.6), Inches(1.4))
    tf = llm_content.text_frame
    p = tf.paragraphs[0]
    p.text = '''"현장 적용 시 약 15% 성능 저하가
예상됩니다. (신뢰도 89%)

주요 원인: 온도와 습도
권장: 환기 시스템 점검 후 적용"'''
    p.font.size = Pt(11)

    y_pos += 2.3

    # 핵심 가치 박스
    value_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y_pos), Inches(9), Inches(1.5)
    )
    value_box.fill.solid()
    value_box.fill.fore_color.rgb = YELLOW

    value_title = slide.shapes.add_textbox(Inches(0.7), Inches(y_pos + 0.1), Inches(8.6), Inches(0.3))
    tf = value_title.text_frame
    p = tf.paragraphs[0]
    p.text = "v2.0 핵심 가치"
    p.font.size = Pt(16)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    values = [
        "✓ 전문가 없이도 예측 결과 이해 가능",
        "✓ 의사결정에 바로 활용 가능한 형태",
        "✓ 권장사항까지 자동 생성"
    ]
    for i, val in enumerate(values):
        vtxt = slide.shapes.add_textbox(Inches(2), Inches(y_pos + 0.5 + i * 0.3), Inches(6), Inches(0.3))
        tf = vtxt.text_frame
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(14)
        p.alignment = PP_ALIGN.CENTER

    return slide


def create_presentation():
    """프레젠테이션 생성"""
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # 슬라이드 추가
    add_title_slide(prs, "Forge v2.0 AI Modules", "LLM + LSTM 기술 구현 상세")
    add_llm_slide(prs)
    add_lstm_slide(prs)
    add_pipeline_slide(prs)

    # 저장
    output_path = "/Users/yangjeong-u/projects/forge/docs/Forge_AI_Modules.pptx"
    prs.save(output_path)
    print(f"✅ 프레젠테이션 저장 완료: {output_path}")
    return output_path


if __name__ == "__main__":
    create_presentation()
