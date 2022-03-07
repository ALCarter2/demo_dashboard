# Import packages
import streamlit as st
import streamlit.components.v1 as components
import datetime
import gspread
from gspread.exceptions import SpreadsheetNotFound
from gspread_pandas import Spread
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from math import nan

# from oauth2client.service_account import ServiceAccountCredentials
from oauth2client import service_account

# from google.oauth2 import service_account
import numpy as np
import pandas as pd
from datetime import date
from datetime import datetime
from dateutil import parser

st.set_page_config(
    layout="wide",
    page_title="Demo Beta App",
    initial_sidebar_state="expanded",
    page_icon="🚀",
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


key = {
    "type": st.secrets["type"],
    "project_id": st.secrets["project_id"],
    "private_key_id": st.secrets["private_key_id"],
    "private_key": st.secrets["private_key"],
    "client_email": st.secrets["client_email"],
    "client_id": st.secrets["client_id"],
    "auth_uri": st.secrets["auth_uri"],
    "token_uri": st.secrets["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["client_x509_cert_url"],
}


scope = [
    st.secrets["entry1"],
    st.secrets["entry2"],
]

plt.rcParams.update(
    {
        "axes.titlesize": 4,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.labelsize": 4,
        "legend.fontsize": 4,
        "legend.title_fontsize": 4,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
    }
)

# Import data
@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def grabDF(sheet_url, sheet, query, count=st.session_state.get("count", 0)):
    # st.write(f"grabDF count: {count}")
    creds = service_account.ServiceAccountCredentials.from_json_keyfile_dict(key, scope)
    client = gspread.authorize(creds)
    gc = client.open_by_url(sheet_url)
    worksheet = gc.worksheet(sheet)
    df = pd.DataFrame(worksheet.get_all_records())
    return df if query == "all" else df.query(query)


def updateDF(df, fields, id_patient):
    """
    fields is a dictionary of column names and values.
    The function updates the row of id_patient with the values in fields.
    """
    for key in fields:
        df.loc[df["id_patient"] == id_patient, key] = fields[key][0]
    return df


def writeDF(sheet_url, sheet, df):
    """
    This function writes the first row of the dataframe to the sheet using gspread.
    """
    creds = service_account.ServiceAccountCredentials.from_json_keyfile_dict(key, scope)
    client = gspread.authorize(creds)
    gc = client.open_by_url(sheet_url)
    worksheet = gc.worksheet(sheet)
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())


@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def getPatients(count=st.session_state.get("count", 0)):
    """
    This function uses grabDF() to grab the data from the sheet.
    The function then goes through each row and returns a list of Last Name,
    First Name, and ID.
    """
    df = grabDF(sheet_url, "Patients", '`First Name` != ""')
    return [
        f"{row['Last Name']}, {row['First Name']} : {row['id_patient']}"
        for index, row in df.iterrows()
    ]


@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def getClinician(count=st.session_state.get("count", 0)):
    """
    This function uses grabDF() to grab the data from the Clinician sheet.
    The function then goes through each row and returns a list of Last Name,
    First Name, and ID.
    """
    df = grabDF(sheet_url, "Clinicians", '`First Name` != ""')
    # st.write(df)
    return [f"{row['First Name']} {row['Last Name']}" for index, row in df.iterrows()]


def create_new_id():
    """
    This function uses grabDF() to grab data from the Patients sheet.
    The function then returns the max ID number + 1.
    """
    df = grabDF(sheet_url, "Patients", '`First Name` != ""')
    return df["id_patient"].max() + 1


def get_id_fields(input_string):

    # -- Split the input string into a list of strings
    split_list = input_string.split(":")
    # -- Get the last name
    last_name = split_list[0].split(",")[0].lstrip()
    # -- Get the first name
    first_name = split_list[0].split(",")[1].strip()
    # -- Get the id
    id_patient = int(split_list[1].lstrip())

    return last_name, first_name, id_patient


@st.cache(allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def format_client_record(df, count=st.session_state.get("count", 0)):
    last_name = df["Last Name"].values[0]
    first_name = df["First Name"].values[0]
    intake_date = parser.parse(df["Intake Date"].values[0])
    clinician = df["Clinician"].values[0]
    gender = df["Gender"].values[0]
    diet = df["Diet"].values[0]
    id_patient = int(df["id_patient"].values[0])
    return (
        last_name,
        first_name,
        intake_date,
        clinician,
        gender,
        diet,
        id_patient,
    )


def evaluate_hasValue(fields):
    """
    This function checks to see if every field in list fields has a length greater than 0.
    If every field has a length greater than 0, return True.
    If not, return False.
    """
    return all(str(fields[field][0]) for field in fields)


def update_fields(fields, row):
    """
    This function updates the fields with the values in row.
    """
    for key in fields:
        fields[key][0] = row[key][0]
    return fields


sheet_url = "https://docs.google.com/spreadsheets/d/1DYK7Yvh-aS4SlHqMXLn-C5TqQZcaSvuJUxryGxxwdQA"
# sheet = "Patients"

# Streamlit Components
options = ["➕ Add Client", "📈 Existing Client", "📊 Metrics"]
clients = getPatients()
forms = ["Clinical Intake", "Release of Information"]
new_client_req = False

# If no, then initialize count to 0
# If count is already initialized, don't do anything
if "count" not in st.session_state:
    st.session_state.count = random.uniform(1, 100)

if "submit" not in st.session_state:
    st.session_state.submit = 0


# st.write(st.session_state.count, st.session_state.submit)
st.subheader("Navigation")
list_options = st.selectbox("Select an option", options)
st.markdown("""---""")

if list_options == "➕ Add Client":
    st.session_state.submit = 0
    st.sidebar.header("")
    clinicians = [""] + getClinician()
    new_id = create_new_id()
    genders = ["", "Female", "Male", "Non-Binary"]
    with st.form("new_client_form"):
        # define multiple columns,
        row1col1, row1col2 = st.columns(2)
        row2col1, row2col2, row2col3 = st.columns(3)
        row3col1, row3col2, row3col3 = st.columns((1, 1, 2))
        row4col1 = st.columns(1)

        # -- First Row
        with row1col2:
            clinician = st.selectbox("Clinician", clinicians)

        # -- Second Row
        with row2col1:
            new_last_name = st.text_input("Last Name")

        with row2col2:
            new_first_name = st.text_input("First Name")

        with row2col3:
            new_gender = st.selectbox("Gender", genders)

        with row3col2:
            row = {
                "Last Name": [new_last_name],
                "First Name": [new_first_name],
                "Intake Date": [str(date.today())],
                "Clinician": [clinician],
                "Gender": [new_gender],
                "Diet": [""],
                "id_patient": new_id,
            }

            df_row = pd.DataFrame(row)

        btn_newForm = st.form_submit_button("✔️ Save Client")
        if new_last_name != "" and new_first_name != "":
            new_client_req = True
        if btn_newForm:
            if new_client_req:
                df = pd.concat([grabDF(sheet_url, "Patients", "all"), df_row])
                writeDF(sheet_url, "Patients", df)
                st.success("New Client Added! 🤗")
                new_client_req = False
                st.session_state.count += 1
            else:
                st.error("❌ Please enter all name fields.")
        st.write(st.session_state.count, st.session_state.submit)

elif list_options == "📈 Existing Client":
    st.sidebar.header("Menu")
    with st.sidebar.form(key="inputs_form"):

        list_clients = st.selectbox("Client:", clients)
        client_lastName, client_firstName, client_id = get_id_fields(list_clients)
        # get all records for the selected client
        df = grabDF(sheet_url, "Patients", f"id_patient == {client_id}")

        # format the dataframe
        (
            client_lastName,
            client_firstName,
            client_intake_date,
            client_clinician,
            client_gender,
            client_diet,
            client_id,
        ) = format_client_record(df)
        fields = {
            "Last Name": [client_lastName],
            "First Name": [client_firstName],
            "Intake Date": [client_intake_date],
            "Clinician": [client_clinician],
            "Gender": [client_gender],
            "Diet": [client_diet],
        }

        list_forms = st.selectbox("Form:", forms)
        submit_btn = st.form_submit_button(label="Submit")
        if submit_btn:
            st.session_state.submit += 1

    if list_forms == "Clinical Intake":
        if st.session_state.submit > 0:
            st.subheader("Clinical Intake")
            df = grabDF(sheet_url, "Patients", query=f"id_patient == {client_id}")

            with st.form("client_record_form"):
                # define multiple columns,
                row1col1, row1col2 = st.columns(2)
                row2col1, row2col2, row2col3 = st.columns(3)
                row3col1, row3col2, row3col3 = st.columns(3)
                row4col1, row4col2, row4col3 = st.columns(3)

                # -- First Row
                with row1col1:
                    client_intake_date = st.date_input(
                        "Intake Date", client_intake_date
                    )
                with row1col2:
                    values_clinicians = [""] + getClinician()
                    default_ix_clinician = values_clinicians.index(client_clinician)
                    client_clinician = st.selectbox(
                        "Clinician",
                        values_clinicians,
                        index=default_ix_clinician,
                    )

                # -- Second Row
                with row2col1:
                    client_lastName = st.text_input(
                        "Client Last Name", value=client_lastName
                    )
                    values_gender = ["", "Female", "Male", "Non-Binary"]
                    default_ix_gender = values_gender.index(client_gender)
                    client_gender = st.selectbox(
                        "Gender", values_gender, index=default_ix_gender
                    )

                with row2col2:
                    client_firstName = st.text_input(
                        "Client First Name", value=client_firstName
                    )
                    values_diet = ["", "Carnivore", "Herbivore", "Omnivore"]
                    default_ix_diet = values_diet.index(client_diet)
                    client_diet = st.selectbox(
                        "Diet", values_diet, index=default_ix_diet
                    )

                with row3col2:
                    row = {
                        "Last Name": [client_lastName],
                        "First Name": [client_firstName],
                        "Intake Date": [str(client_intake_date)],
                        "Clinician": [client_clinician],
                        "Gender": [client_gender],
                        "Diet": [client_diet],
                    }

                btn_saveForm = st.form_submit_button("✔️ Update Client")

                if btn_saveForm:
                    fields = update_fields(fields, row)
                    save_client_req = evaluate_hasValue(row)
                    if save_client_req:
                        df = updateDF(
                            grabDF(sheet_url, "Patients", "all"), row, client_id
                        )
                        writeDF(sheet_url, "Patients", df)
                        st.success("Client profile updated. 😊")
                        st.balloons()
                        st.session_state.count += 1
                        st.session_state.submit = 0
                    else:
                        df = updateDF(
                            grabDF(sheet_url, "Patients", "all"), row, client_id
                        )
                        writeDF(sheet_url, "Patients", df)
                        st.success("Client profile updated. 😊")
                        st.session_state.count += 1
                        st.session_state.submit = 0
                st.write(st.session_state.count, st.session_state.submit)
    elif list_forms == "Release of Information":
        st.subheader("Release of Information")
        st.checkbox(
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip"
        )
else:
    st.session_state.count += 1
    st.session_state.submit = 0
    st.sidebar.header("")
    df = grabDF(sheet_url, "Patients", query=f"all")
    st.write(df)
    st.markdown("""---""")
    row1col1, row1col2, row1col3 = st.columns(3)
    with row1col1:
        plt.figure()
        plt.rcParams.update({"font.size": 22})

        plt.title("Client Count by Clinician", fontsize=10)
        ax1 = (
            df["Clinician"]
            .dropna()
            .value_counts()
            .plot.bar(
                rot=0,
                fontsize=10,
                sort_columns=True,
            )
        )
        ax1.set_ylabel("Count", fontdict={"fontsize": 10})
        ax1.set_xlabel("Clinician", fontdict={"fontsize": 10})

        # Get your current y-ticks (loc is an array of your current y-tick elements)
        loc, labels = plt.yticks()
        # This sets your y-ticks to the specified range at whole number intervals
        plt.yticks(np.arange(0, max(loc), step=1))

        st.pyplot(plt)

    with row1col2:
        plt.figure()
        plt.rcParams.update({"font.size": 22})

        plt.title("Client Count by Diet", fontsize=10)
        ax2 = (
            df["Diet"]
            .dropna()
            .value_counts()
            .plot.bar(
                rot=0,
                fontsize=10,
                sort_columns=True,
                xlabel="Diet",
            )
        )
        ax2.set_ylabel("Count", fontdict={"fontsize": 10})
        ax2.set_xlabel("Diet", fontdict={"fontsize": 10})

        # Get your current y-ticks (loc is an array of your current y-tick elements)
        loc, labels = plt.yticks()
        # This sets your y-ticks to the specified range at whole number intervals
        plt.yticks(np.arange(0, max(loc), step=1))

        st.pyplot(plt)

    with row1col3:
        plt.figure()
        plt.rcParams.update({"font.size": 22})

        plt.title("Client Count by Gender", fontsize=10)
        ax1 = (
            df["Gender"]
            .dropna()
            .value_counts()
            .plot.bar(
                rot=0,
                fontsize=10,
                sort_columns=True,
                xlabel="Gender",
            )
        )
        ax1.set_ylabel("Count", fontdict={"fontsize": 10})
        ax1.set_xlabel("Gender", fontdict={"fontsize": 10})

        # Get your current y-ticks (loc is an array of your current y-tick elements)
        loc, labels = plt.yticks()
        # This sets your y-ticks to the specified range at whole number intervals
        plt.yticks(np.arange(0, max(loc), step=1))

        st.pyplot(plt)

    st.write(st.session_state.count, st.session_state.submit)

# Streamlit Outputs 2
