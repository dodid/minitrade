

def check_streamlit():
    ''' Return True if code is run within streamlit, else False '''
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ModuleNotFoundError:
        return False


if check_streamlit():
    import streamlit as st
    print = st.write
    display = st.write

