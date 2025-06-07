from sqlalchemy import URL, create_engine
import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
from langchain.agents import AgentExecutor 
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq


# pg_uri = f"postgresql+psycopg2://postgres:@localhost:5432/STUDENT"


PSQL = "USE_PSQL"

st.set_page_config(page_title="LangChain: Chat with PostgreSQL")
st.title("LangChain: Chat with SQL DB")


st.sidebar.title("Enter the PSQL DB details")

db_uri = PSQL

psql_host = st.sidebar.text_input("Provide PostgreSQL Host")
psql_port = st.sidebar.text_input("Provide Port Number ")
psql_user = st.sidebar.text_input("Provide PostgreSQL User")
psql_password = st.sidebar.text_input("PostgreSQL Password",type="password")
psql_db = st.sidebar.text_input("PostgreSQL Database")


api_key = st.sidebar.text_input(label="GROQ API KEY",type="password")

if not db_uri:
    st.info("Please enter the database info and uri ")
if not api_key:
    st.info("Please enter the GROQ api key ")
    
llm = ChatGroq(groq_api_key = api_key,model="Llama3-8b-8192",streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(Psql_host=None,Psql_user=None,Psql_password=None,Psql_db=None,Psql_port=None):
    if not (psql_host and psql_db and psql_password and psql_port and psql_user):
        st.error("Please enter all PSQL connection details")
        st.stop()
    url = URL.create(
        drivername = "postgresql+psycopg2",
        username = Psql_user,
        password=Psql_password,
        host=Psql_host,
        port=Psql_port,
        database=Psql_db
    )
    return SQLDatabase(create_engine(url))


db = configure_db(psql_host,psql_user,psql_password,psql_db,psql_port)


#* TOOLKIT

toolkit = SQLDatabaseToolkit(db=db,llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose = True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role":"assistant","content":"How can i help you?"}]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    

user_query = st.chat_input(placeholder="Ask anything about DB")

if user_query:
    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query,callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response) 