# ============================================================
# HEART DISEASE RISK PREDICTION
# Developed by Jevin Kanani
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import random

from datetime import datetime
from io import BytesIO

# PDF Libraries
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
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
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .low-risk {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }

    .moderate-risk {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }

    .high-risk {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }

    .result-title {
        font-size: 30px;
        font-weight: bold;
    }

    .result-percentage {
        font-size: 45px;
        font-weight: bold;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")


model = load_model()


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
# CREATE PROFESSIONAL PDF REPORT
# ============================================================

def create_pdf_report(
    assignment_id,
    report_datetime,
    patient_data,
    risk_level,
    probability
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1f2937"),
        spaceBefore=8,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14
    )

    center_style = ParagraphStyle(
        "CenterStyle",
        parent=normal_style,
        alignment=TA_CENTER
    )

    disclaimer_style = ParagraphStyle(
        "DisclaimerStyle",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#555555")
    )

    story = []

    # ========================================================
    # REPORT HEADER
    # ========================================================

    story.append(
        Paragraph(
            "HEART DISEASE RISK PREDICTION",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Machine Learning Based Risk Assessment Report",
            subtitle_style
        )
    )

    # ========================================================
    # ASSIGNMENT INFORMATION
    # ========================================================

    assignment_data = [
        [
            Paragraph("<b>Assignment ID</b>", normal_style),
            Paragraph(str(assignment_id), normal_style)
        ],
        [
            Paragraph("<b>Date & Time</b>", normal_style),
            Paragraph(report_datetime, normal_style)
        ]
    ]

    assignment_table = Table(
        assignment_data,
        colWidths=[150, 340]
    )

    assignment_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, colors.grey),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#eeeeee")
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
            ]
        )
    )

    story.append(assignment_table)

    story.append(Spacer(1, 18))

    # ========================================================
    # 1. PATIENT INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "1. Patient Information",
            heading_style
        )
    )

    patient_table_data = [
        [
            Paragraph("<b>Parameter</b>", normal_style),
            Paragraph("<b>Value</b>", normal_style)
        ]
    ]

    for parameter, value in patient_data.items():

        patient_table_data.append(
            [
                Paragraph(str(parameter), normal_style),
                Paragraph(str(value), normal_style)
            ]
        )

    patient_table = Table(
        patient_table_data,
        colWidths=[250, 240],
        repeatRows=1
    )

    patient_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#e9ecef")
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
            ]
        )
    )

    story.append(patient_table)

    story.append(Spacer(1, 18))

    # ========================================================
    # 2. PREDICTION RESULT
    # ========================================================

    story.append(
        Paragraph(
            "2. Prediction Result",
            heading_style
        )
    )

    if risk_level == "Low Risk":

        result_background = colors.HexColor("#d4edda")
        result_border = colors.HexColor("#28a745")

    elif risk_level == "Moderate Risk":

        result_background = colors.HexColor("#fff3cd")
        result_border = colors.HexColor("#ffc107")

    else:

        result_background = colors.HexColor("#f8d7da")
        result_border = colors.HexColor("#dc3545")

    risk_style = ParagraphStyle(
        "RiskStyle",
        parent=normal_style,
        fontSize=18,
        alignment=TA_CENTER
    )

    probability_style = ParagraphStyle(
        "ProbabilityStyle",
        parent=normal_style,
        fontSize=25,
        alignment=TA_CENTER
    )

    result_table = Table(
        [
            [
                Paragraph(
                    f"<b>{risk_level.upper()}</b>",
                    risk_style
                )
            ],
            [
                Paragraph(
                    f"<b>{probability:.2f}%</b>",
                    probability_style
                )
            ],
            [
                Paragraph(
                    "Estimated Heart Disease Probability",
                    center_style
                )
            ]
        ],
        colWidths=[490]
    )

    result_table.setStyle(
        TableStyle(
            [
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
                    2,
                    result_border
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
            ]
        )
    )

    story.append(result_table)

    story.append(Spacer(1, 18))

    # ========================================================
    # 3. MODEL INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "3. Model Information",
            heading_style
        )
    )

    model_table_data = [
        [
            Paragraph("<b>Metric</b>", normal_style),
            Paragraph("<b>Performance</b>", normal_style)
        ],
        [
            Paragraph("<b>Selected Model</b>", normal_style),
            Paragraph(MODEL_NAME, normal_style)
        ]
    ]

    for metric, value in MODEL_METRICS.items():

        model_table_data.append(
            [
                Paragraph(metric, normal_style),
                Paragraph(value, normal_style)
            ]
        )

    model_table = Table(
        model_table_data,
        colWidths=[250, 240]
    )

    model_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#e9ecef")
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
            ]
        )
    )

    story.append(model_table)

    story.append(Spacer(1, 18))

    # ========================================================
    # 4. ASSESSMENT NOTE
    # ========================================================

    story.append(
        Paragraph(
            "4. Assessment Note",
            heading_style
        )
    )

    assessment_text = (
        f"The machine learning model classified the provided "
        f"patient information as <b>{risk_level}</b> with an "
        f"estimated probability of <b>{probability:.2f}%</b>. "
        f"The prediction was generated using the trained "
        f"Random Forest machine learning model."
    )

    story.append(
        Paragraph(
            assessment_text,
            normal_style
        )
    )

    story.append(Spacer(1, 18))

    # ========================================================
    # 5. DISCLAIMER
    # ========================================================

    story.append(
        Paragraph(
            "5. Disclaimer",
            heading_style
        )
    )

    disclaimer_text = (
        "<b>Important:</b> This report is generated for "
        "educational and academic purposes only. The prediction "
        "is produced by a machine learning model and should not "
        "be considered a medical diagnosis, medical advice, or "
        "a substitute for professional healthcare consultation. "
        "Always consult a qualified healthcare professional for "
        "medical decisions."
    )

    disclaimer_table = Table(
        [
            [
                Paragraph(
                    disclaimer_text,
                    disclaimer_style
                )
            ]
        ],
        colWidths=[490]
    )

    disclaimer_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#f8f9fa")
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    colors.grey
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
            ]
        )
    )

    story.append(disclaimer_table)

    story.append(Spacer(1, 25))

    # ========================================================
    # FOOTER
    # ========================================================

    footer_table = Table(
        [
            [
                Paragraph(
                    "<b>Developed by</b><br/>Jevin Kanani",
                    normal_style
                ),
                Paragraph(
                    "<b>Heart Disease Risk Prediction</b>",
                    normal_style
                )
            ]
        ],
        colWidths=[245, 245]
    )

    footer_table.setStyle(
        TableStyle(
            [
                (
                    "LINEABOVE",
                    (0, 0),
                    (-1, 0),
                    0.7,
                    colors.grey
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP")
            ]
        )
    )

    story.append(footer_table)

    document.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Heart Risk")

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

st.sidebar.info(
    "Machine Learning Based Heart Disease Risk Prediction"
)

st.sidebar.markdown(
    "**Developed by:**\n\n"
    "Jevin Kanani"
)


# ============================================================
# RISK PREDICTION PAGE
# ============================================================

if page == "Risk Prediction":

    st.markdown(
        '<div class="main-title">Heart Disease Risk Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning Based Heart Disease Risk Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Patient Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.st.slider(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            ["M", "F"]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "ASY", "TA"]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120
        )

    with col2:

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=0,
            max_value=700,
            value=200
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            [0, 1],
            format_func=lambda x:
            "Normal (0)" if x == 0 else "High (1)"
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            ["Normal", "ST", "LVH"]
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

    with col3:

        exercise_angina = st.selectbox(
            "Exercise Angina",
            ["N", "Y"]
        )

        oldpeak = st.st.slider(
            "Oldpeak",
            min_value=-3.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "ST Slope",
            ["Up", "Flat", "Down"]
        )

    st.markdown("---")

    predict_button = st.button(
        "Predict Heart Disease Risk",
        type="primary",
        use_container_width=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        input_data = pd.DataFrame(
            {
                "Age": [age],
                "Sex": [sex],
                "ChestPainType": [chest_pain],
                "RestingBP": [resting_bp],
                "Cholesterol": [cholesterol],
                "FastingBS": [fasting_bs],
                "RestingECG": [resting_ecg],
                "MaxHR": [max_hr],
                "ExerciseAngina": [exercise_angina],
                "Oldpeak": [oldpeak],
                "ST_Slope": [st_slope]
            }
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        # Find probability of class 1
        if hasattr(model, "classes_"):

            classes = list(model.classes_)

            if 1 in classes:

                class_1_index = classes.index(1)

                probability = probabilities[class_1_index] * 100

            else:

                probability = probabilities[-1] * 100

        else:

            probability = probabilities[-1] * 100


        # ====================================================
        # RISK LEVEL
        # ====================================================

        if probability < 30:

            risk_level = "Low Risk"
            risk_class = "low-risk"

        elif probability < 60:

            risk_level = "Moderate Risk"
            risk_class = "moderate-risk"

        else:

            risk_level = "High Risk"
            risk_class = "high-risk"


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )


        # IMPORTANT:
        # No indentation before HTML.
        # This fixes HTML appearing as plain text.

        result_html = f"""
<div class="result-box {risk_class}">
    <div class="result-title">{risk_level}</div>
    <div class="result-percentage">{probability:.2f}%</div>
    <div>Estimated Heart Disease Probability</div>
</div>
"""

        st.markdown(
            result_html,
            unsafe_allow_html=True
        )


        st.progress(
            min(max(int(probability), 0), 100)
        )


        # ====================================================
        # PATIENT SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">Patient Summary</div>',
            unsafe_allow_html=True
        )


        summary_col1, summary_col2 = st.columns(2)


        # Left table
        left_summary = pd.DataFrame(
            {
                "Parameter": [
                    "Age",
                    "Sex",
                    "Chest Pain Type",
                    "Resting Blood Pressure",
                    "Cholesterol",
                    "Fasting Blood Sugar"
                ],

                "Value": [
                    age,
                    sex,
                    chest_pain,
                    resting_bp,
                    cholesterol,
                    fasting_bs
                ]
            }
        )


        # Right table
        right_summary = pd.DataFrame(
            {
                "Parameter": [
                    "Resting ECG",
                    "Maximum Heart Rate",
                    "Exercise Angina",
                    "Oldpeak",
                    "ST Slope"
                ],

                "Value": [
                    resting_ecg,
                    max_hr,
                    exercise_angina,
                    oldpeak,
                    st_slope
                ]
            }
        )


        with summary_col1:

            st.dataframe(
                left_summary,
                use_container_width=True,
                hide_index=True
            )


        with summary_col2:

            st.dataframe(
                right_summary,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.markdown(
            '<div class="section-title">Model Information</div>',
            unsafe_allow_html=True
        )


        model_col1, model_col2 = st.columns(2)


        model_left = pd.DataFrame(
            {
                "Parameter": [
                    "Selected Model",
                    "Accuracy",
                    "Precision"
                ],

                "Value": [
                    MODEL_NAME,
                    MODEL_METRICS["Accuracy"],
                    MODEL_METRICS["Precision"]
                ]
            }
        )


        model_right = pd.DataFrame(
            {
                "Parameter": [
                    "Recall",
                    "F1 Score",
                    "ROC-AUC"
                ],

                "Value": [
                    MODEL_METRICS["Recall"],
                    MODEL_METRICS["F1 Score"],
                    MODEL_METRICS["ROC-AUC"]
                ]
            }
        )


        with model_col1:

            st.dataframe(
                model_left,
                use_container_width=True,
                hide_index=True
            )


        with model_col2:

            st.dataframe(
                model_right,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # PREDICTION EXPLANATION
        # ====================================================

        st.markdown(
            '<div class="section-title">Prediction Explanation</div>',
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
            "Note: Low, Moderate and High risk ranges are "
            "application-defined presentation categories. "
            "They are not clinical thresholds."
        )


        # ====================================================
        # PREDICTION HISTORY
        # ====================================================

        if "history" not in st.session_state:

            st.session_state.history = []


        prediction_time = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )


        history_record = {
            "Date & Time": prediction_time,
            "Age": age,
            "Sex": sex,
            "Risk Level": risk_level,
            "Probability": f"{probability:.2f}%"
        }


        st.session_state.history.append(
            history_record
        )


        # ====================================================
        # PROFESSIONAL PDF REPORT
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Professional Report</div>',
            unsafe_allow_html=True
        )


        assignment_id = random.randint(
            10000000,
            99999999
        )


        report_datetime = datetime.now().strftime(
            "%d %B %Y, %I:%M:%S %p"
        )


        patient_report_data = {

            "Age": age,

            "Sex": sex,

            "Chest Pain Type": chest_pain,

            "Resting Blood Pressure": resting_bp,

            "Cholesterol": cholesterol,

            "Fasting Blood Sugar": fasting_bs,

            "Resting ECG": resting_ecg,

            "Maximum Heart Rate": max_hr,

            "Exercise Angina": exercise_angina,

            "Oldpeak": oldpeak,

            "ST Slope": st_slope
        }


        pdf_file = create_pdf_report(

            assignment_id=assignment_id,

            report_datetime=report_datetime,

            patient_data=patient_report_data,

            risk_level=risk_level,

            probability=probability
        )


        st.download_button(

            label="Download Professional PDF Report",

            data=pdf_file.getvalue(),

            file_name="heart_disease_risk_report.pdf",

            mime="application/pdf",

            use_container_width=True
        )


        st.success(
            f"PDF report generated successfully. "
            f"Assignment ID: {assignment_id}"
        )


# ============================================================
# PREDICTION HISTORY PAGE
# ============================================================

elif page == "Prediction History":

    st.title("Prediction History")


    if (
        "history" not in st.session_state
        or len(st.session_state.history) == 0
    ):

        st.info(
            "No prediction history available yet."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


        st.markdown("---")


        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.title("Model Performance")


    st.write(
        "Five machine learning classification algorithms "
        "were evaluated during model development."
    )


    performance_data = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "SVM",
                "KNN",
                "Logistic Regression",
                "Decision Tree"
            ],

            "Accuracy": [
                "90.22%",
                "89.67%",
                "89.67%",
                "88.59%",
                "79.35%"
            ],

            "Precision": [
                "89.62%",
                "88.79%",
                "91.09%",
                "87.16%",
                "81.37%"
            ],

            "Recall": [
                "93.14%",
                "93.14%",
                "90.20%",
                "93.14%",
                "81.37%"
            ],

            "F1 Score": [
                "91.35%",
                "90.91%",
                "90.64%",
                "90.05%",
                "81.37%"
            ],

            "ROC-AUC": [
                "93.31%",
                "94.94%",
                "94.31%",
                "92.99%",
                "79.10%"
            ]
        }
    )


    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True
    )


    st.markdown("---")


    st.success(
        "Random Forest was selected as the final model "
        "because it achieved the highest F1 Score of 91.35%."
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About Project":

    st.title("About Project")


    st.markdown(
        """
        ## Heart Disease Risk Prediction

        This application uses Machine Learning to estimate
        the probability of heart disease based on patient
        health parameters.

        ### Project Objectives

        - Predict heart disease risk using Machine Learning
        - Compare multiple classification algorithms
        - Select the best-performing model
        - Provide a web-based prediction interface
        - Generate a professional PDF report

        ### Machine Learning Models

        1. Random Forest
        2. Support Vector Machine
        3. K-Nearest Neighbors
        4. Logistic Regression
        5. Decision Tree

        ### Final Model

        **Random Forest**

        ### Performance

        - Accuracy: **90.22%**
        - Precision: **89.62%**
        - Recall: **93.14%**
        - F1 Score: **91.35%**
        - ROC-AUC: **93.31%**

        ### Technologies Used

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        - Streamlit
        - ReportLab

        ### Disclaimer

        This application is developed for educational purposes
        only.

        The prediction generated by this application should
        not be considered a medical diagnosis or professional
        medical advice.

        Please consult a qualified healthcare professional
        for actual medical decisions.
        """
    )


    st.markdown("---")


    st.caption(
        "Developed by Jevin Kanani"
    )
