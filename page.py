import streamlit as st

create_page = st.Page("streamlit_SICU.py", title="SICU Patient Monitoring", icon="📋")
delete_page = st.Page("CT_segmentation.py", title="CT_segmentation", icon="🩻")

pg = st.navigation([create_page, delete_page])
st.set_page_config(page_title="Save the SICU", page_icon="🩺")
pg.run()