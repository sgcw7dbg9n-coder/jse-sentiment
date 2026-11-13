import streamlit as st

from views import companies, headlines, method, models, overview, predictions, returns, scorer

st.set_page_config(page_title="JSE news sentiment", page_icon=":material/monitoring:", layout="wide")

navigation = st.navigation({
    "Findings": [
        st.Page(overview.render, title="Overview", icon=":material/home:", default=True),
        st.Page(models.render, title="Model comparison", icon=":material/leaderboard:", url_path="models"),
        st.Page(returns.render, title="Sentiment and returns", icon=":material/show_chart:", url_path="returns"),
    ],
    "Explore": [
        st.Page(companies.render, title="Company explorer", icon=":material/business:", url_path="companies"),
        st.Page(headlines.render, title="Headlines and scorers", icon=":material/newspaper:", url_path="headlines"),
        st.Page(predictions.render, title="Test-period predictions", icon=":material/query_stats:",
                url_path="predictions"),
        st.Page(scorer.render, title="Score a headline", icon=":material/edit_note:", url_path="score"),
    ],
    "About": [
        st.Page(method.render, title="Data and method", icon=":material/description:", url_path="method"),
    ],
})

with st.sidebar:
    st.caption("Research prototype for an honours project. Not investment advice.")

navigation.run()
