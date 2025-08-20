# -*- coding: utf-8 -*-
import streamlit as st
from .reconciliation_ui import render_reconciliation_ui

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_reconciliation_ui()
