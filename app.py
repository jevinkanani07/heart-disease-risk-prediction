import streamlit as st
import pandas as pd
import joblib
import random

from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GENERAL
   ========================================================== */

.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}

.main .block-container {
    max-width: 1100px;
    padding-top: 35px;
    padding-bottom: 50px;
}


/* ==========================================================
   HEADINGS
   ========================================================== */

.main-title {
    text-align: center;
    color: #FFFFFF !important;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #FFFFFF !important;
    font-size: 18px;
    font-weight: 500;
    margin-bottom: 42px;
}

.section-title {
    color: #FFFFFF !important;
    font-size: 30px;
    font-weight: 800;
    margin-top: 35px;
    margin-bottom: 22px;
}


/* ==========================================================
   LABELS
   ========================================================== */

label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: #FFFFFF !important;
    font-weight: 650 !important;
}


/* ==========================================================
   SELECTBOX
   ========================================================== */

[data-baseweb="select"] > div {
    background-color: #252B36 !important;
    border: 1px solid #667085 !important;
    border-radius: 10px !important;
    color: #FFFFFF !important;
}

[data-baseweb="select"] span {
    color: #FFFFFF !important;
}

[data-baseweb="select"] svg {
    fill: #FFFFFF !important;
}

[data-baseweb="popover"] {
    background-color: #1F2937 !important;
}

[data-baseweb="menu"] {
    background-color: #1F2937 !important;
}

[role="option"] {
    background-color: #1F2937 !important;
    color: #FFFFFF !important;
}

[role="option"]:hover {
    background-color: #374151 !important;
}


/* ==========================================================
   NUMBER INPUT
   ========================================================== */

[data-testid="stNumberInput"] input {
    background-color: #252B36 !important;
    color: #FFFFFF !important;
    border: 1px solid #667085 !important;
    font-size: 17px !important;
    font-weight: 650 !important;
}


/* ==========================================================
   PLUS / MINUS BUTTONS
   ALWAYS RED
   ========================================================== */

[data-testid="stNumberInput"] button {
    background-color: #FF2B2B !important;
    color: #FFFFFF !important;
    border: none !important;
}

[data-testid="stNumberInput"] button:hover {
    background-color: #FF2B2B !important;
    color: #FFFFFF !important;
}

[data-testid="stNumberInput"] button:active {
    background-color: #FF2B2B !important;
    color: #FFFFFF !important;
}

[data-testid="stNumberInput"] button:focus {
    background-color: #FF2B2B !important;
    color: #FFFFFF !important;
    box-shadow: none !important;
}

[data-testid="stNumberInput"] button svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}


/* ==========================================================
   SLIDER
   ========================================================== */

[data-testid="stSlider"] label,
[data-testid="stSlider"] p,
[data-testid="stSlider"] span {
    color: #FFFFFF !important;
}

[data-testid="stSlider"] [role="slider"] {
    background-color: #FF2B2B !important;
    border-color: #FF2B2B !important;
}


/* ==========================================================
   MAIN RED BUTTON
   ========================================================== */

.stButton > button {
    width: 100%;
    min-height: 55px;
    background-color: #FF2B2B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 17px !important;
    font-weight: 750 !important;
}

.stButton > button:hover {
    background-color: #FF2B2B !important;
    color: #FFFFFF !important;
}

.stButton > button:active {
    background-color: #FF2B2B !important;
    color: #FFFFFF !important;
}


/* ==========================================================
   DOWNLOAD BUTTON
   ========================================================== */

.stDownloadButton > button {
    width: 100%;
    min-height: 52px;
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

.stDownloadButton > button:hover {
    background-color: #F3F4F6 !important;
    color: #111827 !important;
}


/* ==========================================================
   RESULT CARD
   ========================================================== */

.result-card {
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 25px;
}

.result-card-low {
    background-color: #DCFCE7;
    border: 3px solid #22C55E;
}

.result-card-moderate {
    background-color: #FFF1D6;
    border: 3px solid #F59E0B;
}

.result-card-high {
    background-color: #FEE2E2;
    border: 3px solid #EF4444;
}


/* ==========================================================
   ABOUT CARDS
   ========================================================== */

.about-card {
    background-color: #171D29;
    border: 1px solid #3B4555;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 20px;
}

.about-card h3 {
    color: #FFFFFF !important;
    margin-top: 0;
}

.about-card p,
.about-card li {
    color: #FFFFFF !important;
    line-height: 1.75;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}


/* ==========================================================
   MOBILE
   ========================================================== */

@media (max-width: 768px) {

    .main-title {
        font-size: 32px;
    }

    .subtitle {
        font-size: 16px;
    }

    .section-title {
        font-size: 25px;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "heart_disease_model.pkl"
    )


try:

    model = load_model()

except Exception as error:

    st.error(
        "The trained machine learning model could not be loaded."
    )

    st.write("Technical details:")

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_NAME = "Random Forest"

MODEL_METRICS = {
    "Accuracy": "90.22%",
    "Precision": "89.62%",
    "Recall": "93.14%",
    "F1 Score": "91.35%",
    "ROC-AUC": "93.31%"
}


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ============================================================
# PDF REPORT FUNCTION
# ============================================================

def create_pdf_report(
    assignment_id,
    patient_data,
    predicted_class,
    probability,
    risk_label
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17324D"),
        spaceAfter=7
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#17324D"),
        spaceBefore=8,
        spaceAfter=7
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    result_title_style = ParagraphStyle(
        "ResultTitleStyle",
        parent=normal_style,
        fontName="Helvetica-Bold",
        fontSize=17,
        alignment=TA_CENTER
    )

    result_percentage_style = ParagraphStyle(
        "ResultPercentageStyle",
        parent=normal_style,
        fontName="Helvetica-Bold",
        fontSize=25,
        alignment=TA_CENTER
    )

    result_text_style = ParagraphStyle(
        "ResultTextStyle",
        parent=normal_style,
        fontSize=9,
        alignment=TA_CENTER
    )

    story = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Heart Disease Risk Prediction",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Machine Learning Based Heart Disease Risk Assessment",
            subtitle_style
        )
    )

    # --------------------------------------------------------
    # ASSIGNMENT ID AND DATE
    # --------------------------------------------------------

    date_time = datetime.now().strftime(
        "%d-%m-%Y %I:%M:%S %p"
    )

    meta_data = [
        ["Assignment ID", str(assignment_id)],
        ["Date & Time", date_time]
    ]

    meta_table = Table(
        meta_data,
        colWidths=[
            55 * mm,
            115 * mm
        ]
    )

    meta_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#E2E8F0")
            ),
            (
                "BACKGROUND",
                (1, 0),
                (1, -1),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#CBD5E1")
            ),
            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (1, 0),
                (1, -1),
                "Helvetica"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(meta_table)

    story.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )

    patient_rows = [
        ["Parameter", "Value"]
    ]

    for parameter, value in patient_data.items():

        patient_rows.append(
            [
                parameter,
                str(value)
            ]
        )

    patient_table = Table(
        patient_rows,
        colWidths=[
            75 * mm,
            95 * mm
        ],
        repeatRows=1
    )

    patient_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#2563EB")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (0, 1),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#CBD5E1")
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#F8FAFC")
                ]
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(patient_table)

    story.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Prediction Result",
            heading_style
        )
    )

    if risk_label == "Low Risk":

        result_color = colors.HexColor("#15803D")
        result_background = colors.HexColor("#DCFCE7")

    elif risk_label == "Moderate Risk":

        result_color = colors.HexColor("#C2410C")
        result_background = colors.HexColor("#FFEDD5")

    else:

        result_color = colors.HexColor("#B91C1C")
        result_background = colors.HexColor("#FEE2E2")

    result_title_style.textColor = result_color
    result_percentage_style.textColor = result_color
    result_text_style.textColor = result_color

    result_rows = [
        [
            Paragraph(
                risk_label,
                result_title_style
            )
        ],
        [
            Paragraph(
                f"{probability:.1f}%",
                result_percentage_style
            )
        ],
        [
            Paragraph(
                "Estimated Heart Disease Probability",
                result_text_style
            )
        ],
        [
            Paragraph(
                f"Predicted Class: <b>{predicted_class}</b>",
                result_text_style
            )
        ]
    ]

    result_table = Table(
        result_rows,
        colWidths=[
            170 * mm
        ]
    )

    result_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                result_background
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                1.5,
                result_color
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(result_table)

    story.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Model Information",
            heading_style
        )
    )

    model_rows = [
        ["Parameter", "Value"],
        ["Selected Model", MODEL_NAME],
        ["Accuracy", MODEL_METRICS["Accuracy"]],
        ["Precision", MODEL_METRICS["Precision"]],
        ["Recall", MODEL_METRICS["Recall"]],
        ["F1 Score", MODEL_METRICS["F1 Score"]],
        ["ROC-AUC", MODEL_METRICS["ROC-AUC"]]
    ]

    model_table = Table(
        model_rows,
        colWidths=[
            75 * mm,
            95 * mm
        ],
        repeatRows=1
    )

    model_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#2563EB")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (0, 1),
                (0, -1),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#CBD5E1")
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#F8FAFC")
                ]
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(model_table)

    story.append(
        Spacer(1, 12)
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This application provides a "
            "machine learning based risk estimation. The result "
            "is not a medical diagnosis and should not replace "
            "professional medical advice, diagnosis or treatment.",
            normal_style
        )
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    def add_footer(canvas, doc):

        canvas.saveState()

        page_width, page_height = A4

        canvas.setStrokeColor(
            colors.HexColor("#CBD5E1")
        )

        canvas.line(
            15 * mm,
            10 * mm,
            page_width - 15 * mm,
            10 * mm
        )

        canvas.setFont(
            "Helvetica-Bold",
            8
        )

        canvas.setFillColor(
            colors.HexColor("#64748B")
        )

        canvas.drawString(
            15 * mm,
            5 * mm,
            "Developed by Jevin Kanani"
        )

        canvas.drawRightString(
            page_width - 15 * mm,
            5 * mm,
            "Heart Disease Risk Prediction"
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_footer,
        onLaterPages=add_footer
    )

    buffer.seek(0)

    return buffer


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Heart Disease Risk Prediction"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Risk Prediction",
        "Prediction History",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Developed by Jevin Kanani"
)


# ============================================================
# RISK PREDICTION PAGE
# ============================================================

if page == "Risk Prediction":

    st.markdown(
        '<div class="main-title">'
        'Heart Disease Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning Based Heart Disease Risk Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Patient Information'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # PATIENT INPUT FORM
    # ONE FIELD PER ROW
    # ========================================================

    age = st.slider(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )


    sex = st.selectbox(
        "Sex",
        ["M", "F"],
        format_func=lambda x: (
            "M - Male"
            if x == "M"
            else
            "F - Female"
        )
    )


    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"],
        format_func=lambda x: {
            "ATA": "ATA - Atypical Angina",
            "NAP": "NAP - Non-Anginal Pain",
            "ASY": "ASY - Asymptomatic",
            "TA": "TA - Typical Angina"
        }[x]
    )


    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )


    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=200,
        step=1
    )


    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1],
        format_func=lambda x: (
            "Normal (0) - <=120 mg/dL"
            if x == 0
            else
            "High (1) - >120 mg/dL"
        )
    )


    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        format_func=lambda x: {
            "Normal": "Normal",
            "ST": "ST - ST-T Wave Abnormality",
            "LVH": "LVH - Left Ventricular Hypertrophy"
        }[x]
    )


    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )


    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"],
        format_func=lambda x: (
            "N - No"
            if x == "N"
            else
            "Y - Yes"
        )
    )


    oldpeak = st.slider(
        "Oldpeak",
        min_value=-3.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        format="%.1f"
    )


    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        format_func=lambda x: {
            "Up": "Up - Upsloping",
            "Flat": "Flat",
            "Down": "Down - Downsloping"
        }[x]
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_clicked = st.button(
        "Predict Heart Disease Risk",
        use_container_width=True
    )


    if predict_clicked:

        try:

            # =================================================
            # CREATE MODEL INPUT
            # =================================================

            input_data = pd.DataFrame(
                {
                    "Age": [int(age)],
                    "Sex": [sex],
                    "ChestPainType": [chest_pain],
                    "RestingBP": [int(resting_bp)],
                    "Cholesterol": [int(cholesterol)],
                    "FastingBS": [int(fasting_bs)],
                    "RestingECG": [resting_ecg],
                    "MaxHR": [int(max_hr)],
                    "ExerciseAngina": [exercise_angina],
                    "Oldpeak": [float(oldpeak)],
                    "ST_Slope": [st_slope]
                }
            )


            # =================================================
            # MODEL PREDICTION
            # =================================================

            prediction = model.predict(
                input_data
            )[0]


            # =================================================
            # PROBABILITY
            # =================================================

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

            else:

                raise ValueError(
                    "The loaded model does not support probability prediction."
                )


            # =================================================
            # GET CLASS 1 PROBABILITY
            # =================================================

            if hasattr(model, "classes_"):

                classes = list(
                    model.classes_
                )

                if 1 in classes:

                    class_1_index = classes.index(1)

                    probability = (
                        probabilities[class_1_index]
                        * 100
                    )

                else:

                    probability = (
                        probabilities[-1]
                        * 100
                    )

            else:

                probability = (
                    probabilities[-1]
                    * 100
                )


            probability = float(
                probability
            )


            # =================================================
            # PREDICTED CLASS
            # =================================================

            try:

                prediction_number = int(
                    prediction
                )

            except Exception:

                prediction_number = (
                    1
                    if str(prediction).lower()
                    in ["1", "yes", "true", "heart disease"]
                    else 0
                )


            if prediction_number == 1:

                predicted_class = (
                    "Heart Disease"
                )

            else:

                predicted_class = (
                    "No Heart Disease"
                )


            # =================================================
            # RISK CATEGORY
            # =================================================

            if probability < 30:
                risk_label = "Low Risk"

                # Dark green background
                card_background = "#14532D"
                card_border = "#22C55E"

            elif probability < 60:
                risk_label = "Moderate Risk"

                # Dark orange background
                card_background = "#7C2D12"
                card_border = "#F59E0B"

            else:
                risk_label = "High Risk"

                # Dark red background
                card_background = "#7F1D1D"
                card_border = "#EF4444"

                # Text will always remain white
                card_text = "#FFFFFF"

            # =================================================
            # PREDICTION RESULT
            #
            # IMPORTANT:
            # NO HTML DIV RESULT HERE.
            # This prevents the previous rendering bug.
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Prediction Result'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # RESULT CARD
            # =================================================

            if risk_label == "Low Risk":

                st.success(
                    f"""
### Low Risk

# {probability:.1f}%

**Estimated Heart Disease Probability**

Predicted Class: **{predicted_class}**
"""
                )

            elif risk_label == "Moderate Risk":

                st.warning(
                    f"""
### Moderate Risk

# {probability:.1f}%

**Estimated Heart Disease Probability**

Predicted Class: **{predicted_class}**
"""
                )

            else:

                st.error(
                    f"""
### High Risk

# {probability:.1f}%

**Estimated Heart Disease Probability**

Predicted Class: **{predicted_class}**
"""
                )


            # =================================================
            # PATIENT SUMMARY
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Patient Summary'
                '</div>',
                unsafe_allow_html=True
            )


            patient_summary = pd.DataFrame(
                [
                    ["Age", age],
                    ["Sex", sex],
                    [
                        "Chest Pain Type",
                        chest_pain
                    ],
                    [
                        "Resting Blood Pressure",
                        resting_bp
                    ],
                    [
                        "Cholesterol",
                        cholesterol
                    ],
                    [
                        "Fasting Blood Sugar",
                        fasting_bs
                    ],
                    [
                        "Resting ECG",
                        resting_ecg
                    ],
                    [
                        "Maximum Heart Rate",
                        max_hr
                    ],
                    [
                        "Exercise Angina",
                        exercise_angina
                    ],
                    [
                        "Oldpeak",
                        f"{oldpeak:.1f}"
                    ],
                    [
                        "ST Slope",
                        st_slope
                    ]
                ],
                columns=[
                    "Parameter",
                    "Value"
                ]
            )


            # SINGLE COMPLETE TABLE

            st.dataframe(
                patient_summary,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # MODEL INFORMATION
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Model Information'
                '</div>',
                unsafe_allow_html=True
            )


            model_information = pd.DataFrame(
                [
                    [
                        "Selected Model",
                        MODEL_NAME
                    ],
                    [
                        "Accuracy",
                        MODEL_METRICS["Accuracy"]
                    ],
                    [
                        "Precision",
                        MODEL_METRICS["Precision"]
                    ],
                    [
                        "Recall",
                        MODEL_METRICS["Recall"]
                    ],
                    [
                        "F1 Score",
                        MODEL_METRICS["F1 Score"]
                    ],
                    [
                        "ROC-AUC",
                        MODEL_METRICS["ROC-AUC"]
                    ]
                ],
                columns=[
                    "Parameter",
                    "Value"
                ]
            )


            # SINGLE COMPLETE TABLE

            st.dataframe(
                model_information,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # PREDICTION EXPLANATION
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Prediction Explanation'
                '</div>',
                unsafe_allow_html=True
            )


            st.info(
                "The prediction is generated using a Random Forest "
                "machine learning model trained on heart disease "
                "patient data. The model evaluates multiple patient "
                "features together to estimate the probability of "
                "heart disease."
            )


            st.caption(
                "Low, Moderate and High risk ranges are "
                "application-defined presentation categories. "
                "They are not clinical thresholds."
            )


            # =================================================
            # SAVE PREDICTION HISTORY
            # =================================================

            history_record = {
                "Date & Time": datetime.now().strftime(
                    "%d-%m-%Y %I:%M %p"
                ),
                "Age": age,
                "Sex": sex,
                "Chest Pain Type": chest_pain,
                "Resting BP": resting_bp,
                "Cholesterol": cholesterol,
                "Fasting BS": fasting_bs,
                "Resting ECG": resting_ecg,
                "Maximum HR": max_hr,
                "Exercise Angina": exercise_angina,
                "Oldpeak": oldpeak,
                "ST Slope": st_slope,
                "Prediction": predicted_class,
                "Probability": f"{probability:.1f}%",
                "Risk": risk_label
            }


            st.session_state.prediction_history.append(
                history_record
            )


            # =================================================
            # PROFESSIONAL PDF REPORT
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Professional PDF Report'
                '</div>',
                unsafe_allow_html=True
            )


            # 8 DIGIT ASSIGNMENT ID

            assignment_id = str(
                random.randint(
                    10000000,
                    99999999
                )
            )


            # IMPORTANT:
            # THESE ARE EXACTLY THE SAME VALUES USED
            # FOR THE MACHINE LEARNING PREDICTION.

            pdf_patient_data = {
                "Age": age,
                "Sex": sex,
                "Chest Pain Type": chest_pain,
                "Resting Blood Pressure": resting_bp,
                "Cholesterol": cholesterol,
                "Fasting Blood Sugar": fasting_bs,
                "Resting ECG": resting_ecg,
                "Maximum Heart Rate": max_hr,
                "Exercise Angina": exercise_angina,
                "Oldpeak": f"{oldpeak:.1f}",
                "ST Slope": st_slope
            }


            pdf_file = create_pdf_report(
                assignment_id=assignment_id,
                patient_data=pdf_patient_data,
                predicted_class=predicted_class,
                probability=probability,
                risk_label=risk_label
            )


            st.download_button(
                label="Download Professional PDF Report",
                data=pdf_file.getvalue(),
                file_name="heart_disease_risk_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )


        except Exception as error:

            # =================================================
            # PREDICTION ERROR
            # =================================================

            st.error(
                "Prediction could not be completed."
            )

            st.write(
                "Technical details:"
            )

            st.code(
                str(error)
            )

            st.info(
                "Please make sure that heart_disease_model.pkl "
                "is the trained pipeline containing the same "
                "11 features used by this application."
            )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.markdown(
        '<div class="main-title">'
        'Prediction History'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Previous predictions from the current session'
        '</div>',
        unsafe_allow_html=True
    )


    if len(
        st.session_state.prediction_history
    ) == 0:

        st.info(
            "No prediction history is available yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        if st.button(
            "Clear Prediction History",
            use_container_width=True
        ):

            st.session_state.prediction_history = []

            st.rerun()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">'
        'Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Evaluation results of the machine learning models'
        '</div>',
        unsafe_allow_html=True
    )


    performance_data = pd.DataFrame(
        [
            [
                "Random Forest",
                "90.22%",
                "89.62%",
                "93.14%",
                "91.35%",
                "93.31%"
            ],
            [
                "SVM",
                "89.67%",
                "88.79%",
                "93.14%",
                "90.91%",
                "94.94%"
            ],
            [
                "KNN",
                "89.67%",
                "91.09%",
                "90.20%",
                "90.64%",
                "94.31%"
            ],
            [
                "Logistic Regression",
                "88.59%",
                "87.16%",
                "93.14%",
                "90.05%",
                "92.99%"
            ],
            [
                "Decision Tree",
                "79.35%",
                "81.37%",
                "81.37%",
                "81.37%",
                "79.10%"
            ]
        ],
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    )


    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        '<div class="section-title">'
        'Selected Model'
        '</div>',
        unsafe_allow_html=True
    )


    st.success(
        "Random Forest was selected as the final model "
        "because it achieved the highest F1 Score of 91.35%."
    )


    st.markdown(
        '<div class="section-title">'
        'Evaluation Metrics'
        '</div>',
        unsafe_allow_html=True
    )


    metric_data = pd.DataFrame(
        [
            [
                "Accuracy",
                "Percentage of total predictions classified correctly."
            ],
            [
                "Precision",
                "Percentage of predicted positive cases that were actually positive."
            ],
            [
                "Recall",
                "Percentage of actual positive cases correctly identified."
            ],
            [
                "F1 Score",
                "Combined measure of Precision and Recall."
            ],
            [
                "ROC-AUC",
                "Measures how effectively the model separates the two classes."
            ]
        ],
        columns=[
            "Metric",
            "Meaning"
        ]
    )


    st.dataframe(
        metric_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">'
        'About Project'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning Based Heart Disease Risk Assessment System'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>Project Overview</h3>

<p>
Heart Disease Risk Prediction is an interactive machine
learning application developed to estimate the probability
of heart disease from selected patient health attributes.
</p>

<p>
The application combines data preprocessing, supervised
machine learning, probability estimation, interactive
visualization and professional reporting into one system.
</p>

<p>
Users enter patient information through the prediction form.
The trained machine learning pipeline processes the input
and produces a predicted class along with an estimated
probability of heart disease.
</p>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # OBJECTIVES
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>Project Objectives</h3>

<ul>
<li>Develop a machine learning based heart disease risk estimation system.</li>
<li>Process both numerical and categorical patient features.</li>
<li>Compare multiple supervised classification algorithms.</li>
<li>Select the most suitable model using evaluation metrics.</li>
<li>Provide an interactive and easy-to-use prediction interface.</li>
<li>Display estimated probability along with a prediction category.</li>
<li>Maintain prediction history during the active application session.</li>
<li>Generate a professional PDF report from prediction results.</li>
<li>Provide detailed information about the project and its input features.</li>
</ul>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MACHINE LEARNING METHODOLOGY
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>Machine Learning Methodology</h3>

<p>
The system follows a supervised machine learning
classification approach. Patient information is used as
input features and the trained model predicts whether the
patient belongs to the heart disease class.
</p>

<p>
Numerical features are standardized using StandardScaler,
while categorical features are transformed using
OneHotEncoder. The preprocessing and trained model are
stored together in a single machine learning pipeline.
</p>

<p>
Five classification algorithms were evaluated:
</p>

<ul>
<li>Random Forest</li>
<li>Support Vector Machine</li>
<li>K-Nearest Neighbors</li>
<li>Logistic Regression</li>
<li>Decision Tree</li>
</ul>

<p>
Random Forest was selected as the final model because it
achieved the highest F1 Score of 91.35% among the evaluated
models.
</p>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>Input Features</h3>

<ul>
<li><b>Age:</b> Patient age in completed years.</li>

<li><b>Sex:</b> Patient sex represented by M or F.</li>

<li><b>Chest Pain Type:</b> Category describing the patient's chest pain pattern.</li>

<li><b>Resting Blood Pressure:</b> Blood pressure measured while the patient is at rest.</li>

<li><b>Cholesterol:</b> Recorded cholesterol level.</li>

<li><b>Fasting Blood Sugar:</b> Binary category representing fasting blood sugar level.</li>

<li><b>Resting ECG:</b> Category describing the resting electrocardiogram result.</li>

<li><b>Maximum Heart Rate:</b> Maximum heart rate achieved during the recorded test.</li>

<li><b>Exercise Angina:</b> Indicates whether exercise-related angina is present.</li>

<li><b>Oldpeak:</b> Numerical ST-segment depression measurement.</li>

<li><b>ST Slope:</b> Category describing the slope of the ST segment during exercise.</li>
</ul>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SYSTEM WORKFLOW
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>System Workflow</h3>

<ol>

<li>
Patient information is entered through the interactive form.
</li>

<li>
The application creates a structured DataFrame containing
the 11 required input features.
</li>

<li>
The saved preprocessing pipeline transforms numerical and
categorical features.
</li>

<li>
The Random Forest model processes the transformed data.
</li>

<li>
The model generates the predicted class.
</li>

<li>
The model calculates the probability associated with the
heart disease class.
</li>

<li>
The application displays the prediction and probability.
</li>

<li>
A presentation risk category is assigned according to the
application-defined probability ranges.
</li>

<li>
The prediction is stored in the active session history.
</li>

<li>
A professional PDF report can be generated using the exact
same patient data and prediction result.
</li>

</ol>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>Technology Stack</h3>

<ul>

<li><b>Python:</b> Main programming language.</li>

<li><b>Pandas:</b> Data manipulation and DataFrame operations.</li>

<li><b>NumPy:</b> Numerical computing.</li>

<li><b>Scikit-learn:</b> Machine learning and preprocessing.</li>

<li><b>Joblib:</b> Model serialization and loading.</li>

<li><b>Streamlit:</b> Interactive web application framework.</li>

<li><b>ReportLab:</b> PDF report generation.</li>

<li><b>Jupyter Notebook:</b> Data analysis and model development.</li>

<li><b>GitHub:</b> Source code management and deployment.</li>

</ul>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # APPLICATION FEATURES
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>Application Features</h3>

<ul>

<li>Interactive patient information form.</li>

<li>Full-width patient input fields.</li>

<li>Red increment and decrement controls.</li>

<li>Descriptive labels for coded categorical values.</li>

<li>Machine learning based probability estimation.</li>

<li>Low, Moderate and High presentation categories.</li>

<li>Complete patient summary table.</li>

<li>Complete model information table.</li>

<li>Prediction history for the current session.</li>

<li>Professional PDF report generation.</li>

<li>Model performance comparison.</li>

<li>Detailed project documentation inside the application.</li>

</ul>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>Final Model Performance</h3>

<ul>

<li><b>Selected Model:</b> Random Forest</li>

<li><b>Accuracy:</b> 90.22%</li>

<li><b>Precision:</b> 89.62%</li>

<li><b>Recall:</b> 93.14%</li>

<li><b>F1 Score:</b> 91.35%</li>

<li><b>ROC-AUC:</b> 93.31%</li>

</ul>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>Limitations</h3>

<ul>

<li>
The model's performance depends on the quality and
characteristics of the training dataset.
</li>

<li>
The prediction is an estimated machine learning output.
</li>

<li>
The system does not replace professional medical evaluation.
</li>

<li>
Application-defined risk categories are not clinical thresholds.
</li>

<li>
Prediction history is maintained only during the active session.
</li>

</ul>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FUTURE SCOPE
    # --------------------------------------------------------

    st.markdown(
        """
<div class="about-card">

<h3>Future Scope</h3>

<ul>

<li>Integration with a secure database.</li>

<li>Permanent patient record management.</li>

<li>User authentication and access control.</li>

<li>Advanced analytics dashboard.</li>

<li>Model explainability using SHAP or similar methods.</li>

<li>Automated model retraining.</li>

<li>API based prediction services.</li>

<li>Cloud database integration.</li>

<li>Administrative dashboard.</li>

<li>Advanced reporting and visualization.</li>

</ul>

</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        "This application is a machine learning based risk "
        "estimation system and is not a medical diagnostic system. "
        "The prediction should not be considered a substitute for "
        "professional medical advice, diagnosis or treatment."
    )


    st.markdown(
        """
<div style="
    text-align:right;
    color:#FFFFFF;
    font-size:15px;
    font-weight:700;
    margin-top:30px;
    padding-bottom:20px;
">
Developed by Jevin Kanani
</div>
""",
        unsafe_allow_html=True
    )
