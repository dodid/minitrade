import streamlit as st

from minitrade.trader import StrategyManager

st.set_page_config(page_title='Strategy', layout='wide')


def show_strategy_uploader():
    ''' Create strategy uploader control '''
    uploaded_file = st.sidebar.file_uploader('Upload strategy file', type=['py'], accept_multiple_files=False)
    if uploaded_file is not None:
        StrategyManager.save(uploaded_file.name, uploaded_file.getvalue().decode("utf-8"))
        return uploaded_file.name


def load_strategy(strategy_file: str) -> str:
    try:
        StrategyManager.load(strategy_file)
    except Exception as e:
        st.error(e)
    return StrategyManager.read(strategy_file)


def confirm_delete_trade_strategy(strategy_file: str) -> None:
    def confirm_delete():
        if st.session_state.delete_confirm_textinput == strategy_file:
            StrategyManager.delete(strategy_file)
    st.text_input(f'Type "{strategy_file}" and press Enter to delete',
                  on_change=confirm_delete, key='delete_confirm_textinput')


def show_strategy_code_and_controls(strategy_file):
    c1, c2, c3 = st.columns([6, 1, 1])
    c1.subheader(strategy_file)
    strategy_code = load_strategy(strategy_file)
    c2.download_button('Download', data=strategy_code, file_name=strategy_file)
    if c3.button('Delete', type="primary"):
        confirm_delete_trade_strategy(strategy_file)
    st.code(strategy_code, language='python')


def show_strategy_selector(uploaded_file: str) -> str | None:
    strategy_lst = StrategyManager.list()
    selected_index = strategy_lst.index(uploaded_file) if uploaded_file else 0
    return st.sidebar.radio('Strategy', strategy_lst, index=selected_index)


uploaded_file = show_strategy_uploader()
strategy_file = show_strategy_selector(uploaded_file)
if strategy_file:
    show_strategy_code_and_controls(strategy_file)
